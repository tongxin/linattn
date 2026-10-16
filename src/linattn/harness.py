"""The comparison harness: one entry point that runs every variant through the
same measurements, so that quality and cost numbers are directly comparable.

The harness deliberately does *not* try to be a serving engine.  It answers the
three questions that the survey makes claims about:

1. does the variant reproduce dense attention when its budget is unrestricted?
2. how much error does it introduce at a realistic budget, and how much of the
   attention mass does its selection recover?
3. what does that buy in state size and KV reads?

Real-model behaviour (perplexity, needle retrieval, head classification) is
handled by ``experiments/`` because it needs a pretrained checkpoint.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from linattn.core import AttnConfig, AttentionVariant, dense_attention
from linattn.metrics import max_abs_error, rel_error

__all__ = ["make_inputs", "run_variant", "compare", "structured_inputs"]


def make_inputs(
    batch: int,
    heads: int,
    seq: int,
    dim: int,
    *,
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    kv_heads: Optional[int] = None,
):
    """Random q/k/v with unit-scale entries (no planted structure)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    kv_heads = kv_heads or heads
    q = torch.randn(batch, heads, seq, dim, generator=g, dtype=torch.float32)
    k = torch.randn(batch, kv_heads, seq, dim, generator=g, dtype=torch.float32)
    v = torch.randn(batch, kv_heads, seq, dim, generator=g, dtype=torch.float32)
    return q.to(device=device, dtype=dtype), k.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype)


def structured_inputs(
    batch: int,
    heads: int,
    seq: int,
    dim: int,
    *,
    n_spikes: int = 4,
    spike_width: int = 32,
    seed: int = 0,
    kv_heads: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
):
    """Random inputs plus planted "retrieval" structure.

    Long-context attention in trained models is *not* uniform: a handful of
    distant blocks collect most of the mass (retrieval heads), while the rest of
    the mass is local.  We plant that structure so that selection quality can be
    measured against a realistic target rather than against uniform noise, where
    any selection looks fine.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    kv_heads = kv_heads or heads
    q = torch.randn(batch, heads, seq, dim, generator=g, dtype=torch.float32)
    k = torch.randn(batch, kv_heads, seq, dim, generator=g, dtype=torch.float32)
    v = torch.randn(batch, kv_heads, seq, dim, generator=g, dtype=torch.float32)
    # plant spikes: pick random key positions, give them large key norms so that
    # queries with a matching random direction hit them hard.
    for b in range(batch):
        for h in range(kv_heads):
            starts = torch.randint(0, max(1, seq - spike_width), (n_spikes,), generator=g)
            for s in starts:
                s = int(s)
                k[b, h, s : s + spike_width] *= 3.0
            # local mass: recent keys get larger norms (recency bias of trained LMs)
            k[b, h] *= torch.linspace(0.6, 1.4, seq).unsqueeze(-1)
    return q.to(dtype=dtype), k.to(dtype=dtype), v.to(dtype=dtype)


@dataclass
class VariantRecord:
    variant: str
    family: str
    seq: int
    rel_err: float = 0.0
    max_abs_err: float = 0.0
    mass_recall: float = float("nan")
    kv_reads: float = 0.0
    state_bytes: float = 0.0
    latency_ms: float = 0.0
    peak_mib: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        out = dict(self.__dict__)
        out["extra"] = dict(self.extra)
        return out


def run_variant(
    variant: AttentionVariant,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    reference: Optional[torch.Tensor] = None,
    chunk: Optional[int] = None,
    measure_latency: bool = True,
    dtype: torch.dtype = torch.float64,
) -> VariantRecord:
    """Run one variant end-to-end and measure quality plus cost."""
    qq, kk, vv = q.to(dtype), k.to(dtype), v.to(dtype)
    if reference is None:
        reference = dense_attention(qq, kk, vv)
    t0 = time.perf_counter()
    out, state, stats = variant.forward(qq, kk, vv, chunk=chunk, collect_stats=True)
    latency = (time.perf_counter() - t0) * 1e3
    rec = VariantRecord(
        variant=variant.name,
        family=variant.family,
        seq=qq.shape[-2],
        rel_err=rel_error(out, reference),
        max_abs_err=max_abs_error(out, reference),
        kv_reads=variant.read_budget(qq.shape[-2], qq.shape[0]),
        state_bytes=float(variant.memory_bytes(qq.shape[-2], qq.shape[0])),
        latency_ms=latency if measure_latency else 0.0,
    )
    merged: Dict[str, float] = {}
    for s in stats:
        for key, value in s.items():
            if isinstance(value, (int, float)):
                merged.setdefault(key, 0.0)
                merged[key] += float(value)
    rec.extra = {k: v / max(1, len(stats)) for k, v in merged.items()}
    if "reads_per_query" in rec.extra:
        rec.kv_reads = rec.extra["reads_per_query"]
    masks = getattr(variant, "_masks", None)
    if masks:
        from linattn.metrics import attention_mass, mass_recall

        block = variant.cfg.block_size
        # each chunk's mask covers the keys cached at that point, so assemble them
        # into the full (T_q, T_k) grid rather than concatenating
        mask = torch.zeros(qq.shape[0], qq.shape[1], qq.shape[-2], qq.shape[-2], dtype=torch.bool,
                           device=qq.device)
        pos = 0
        for m in masks:
            t_qc, t_kc = m.shape[-2], m.shape[-1]
            mask[..., pos : pos + t_qc, :t_kc] = m
            pos += t_qc
        t_q, t_k = mask.shape[-2], mask.shape[-1]
        n_qb, n_kb = t_q // block, t_k // block
        if n_qb and n_kb:
            reduced = mask[..., : n_qb * block, : n_kb * block]
            reduced = reduced.view(*mask.shape[:-2], n_qb, block, n_kb, block).any(-3).any(-1)
            mass = attention_mass(qq, kk, vv, block=block)
            rec.extra["mass_recall"] = float(mass_recall(mass, reduced).mean())
    return rec


def compare(
    variants: Sequence[AttentionVariant],
    *,
    seq: int = 1024,
    batch: int = 1,
    heads: int = 4,
    kv_heads: int = 4,
    dim: int = 64,
    structured: bool = True,
    chunk: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
) -> List[Dict]:
    """Run a list of variants on one shared input and return comparable records."""
    if structured:
        q, k, v = structured_inputs(batch, heads, seq, dim, kv_heads=kv_heads)
    else:
        q, k, v = make_inputs(batch, heads, seq, dim, kv_heads=kv_heads)
    reference = dense_attention(q.to(dtype), k.to(dtype), v.to(dtype))
    records = []
    for variant in variants:
        try:
            rec = run_variant(variant, q, k, v, reference=reference, chunk=chunk, dtype=dtype)
        except Exception as exc:  # keep the sweep going; surface the failure in the table
            rec = VariantRecord(variant=variant.name, family=variant.family, seq=seq, extra={"error": str(exc)})
        records.append(rec.as_dict())
    return records


def save_records(records: Sequence[Dict], path: str, **meta) -> None:
    payload = {"meta": meta, "records": list(records)}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
