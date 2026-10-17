"""Tiered cost model for the KV layer.

Reported as a breakdown, because the question the repo asks is *which term
dominates*:

======================  ==========================================================
``attn_ms``             max(attention FLOPs / peak, KV bytes read / bandwidth)
``mlp_ms``              the non-attention transformer work (MLP + projections)
``kv_write_ms``         new KV written to the hot tier
``promote_ms``          cached blocks pulled from the tier they live in
``recompute_ms``        (1 - hit_rate) of the prompt's prefill work, on a miss
``lookup_ms``           prefix-cache block lookups
======================  ==========================================================

``hit_rate`` is an explicit input -- the "artificial feature parameter" -- which can
be measured with :mod:`linattn.kv.cache` or swept. ``kv_read_fraction`` is how the
sparse-attention variants enter: it scales attention reads and FLOPs, not storage.
``compute_kind`` separates *algorithms* from *formats*: a recurrent layer and a dense
layer can cache comparable bytes while having linear vs quadratic FLOP curves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from linattn.kv.format import KVFormat

__all__ = ["Hardware", "ModelShape", "Workload", "CostBreakdown", "breakdown", "sweep_hit_rate"]


@dataclass
class Hardware:
    """Bandwidths in bytes/s, compute in FLOP/s."""

    hbm_bw: float = 3.35e12          # H100 SXM
    dram_bw: float = 4.0e10          # ~40 GB/s per socket, conservative
    ssd_bw: float = 6.0e9            # NVMe, sequential
    flops: float = 9.9e14            # dense fp16 tensor cores
    kv_tier: str = "hbm"             # where newly computed KV is written
    cached_tier: str = "hbm"         # where a cached prefix is served from
    lookup_us_per_block: float = 0.4  # hash + probe, per 16-token block

    def bandwidth(self, tier: str) -> float:
        try:
            return {"hbm": self.hbm_bw, "dram": self.dram_bw, "ssd": self.ssd_bw}[tier]
        except KeyError as exc:
            raise ValueError(f"unknown tier {tier!r}; use hbm | dram | ssd") from exc


@dataclass
class ModelShape:
    n_layers: int = 61
    n_heads: int = 32
    head_dim: int = 128
    d_model: int = 4096
    mlp_flops_multiple: float = 12.0   # MLP + projections, in units of d_model^2
    state_dk: int = 128
    state_dv: int = 128

    def attention_flops(self, seq_len: int, batch: int = 1, decode_tokens: int = 0) -> float:
        """Causal prefill attention plus ``decode_tokens`` decode steps (the core only)."""
        per_layer_prefill = 2.0 * 2.0 * self.n_heads * self.head_dim * (seq_len * (seq_len + 1) / 2)
        per_layer_decode = 2.0 * 2.0 * self.n_heads * self.head_dim * seq_len * decode_tokens
        return (per_layer_prefill + per_layer_decode) * self.n_layers * batch

    def recurrent_flops(self, seq_len: int, batch: int = 1, decode_tokens: int = 0) -> float:
        """State read + update per token; four dk x dv matvecs per head is the right order."""
        tokens = seq_len + decode_tokens
        core = 2.0 * 4.0 * self.n_heads * self.state_dk * self.state_dv * tokens
        return core * self.n_layers * batch

    def mlp_flops(self, seq_len: int, batch: int = 1, decode_tokens: int = 0) -> float:
        tokens = seq_len + decode_tokens
        return 2.0 * self.mlp_flops_multiple * self.d_model**2 * tokens * self.n_layers * batch


@dataclass
class Workload:
    prompt_tokens: int = 32768
    output_tokens: int = 512
    batch: int = 1
    requests: int = 1
    hit_rate: float = 0.0            # fraction of the prompt served from cache
    kv_read_fraction: float = 1.0    # sparse attention: fraction of KV actually read
    gap_s: float = 0.0               # idle gap per request (amortised separately)
    compute_kind: str = "dense"      # dense | sparse | recurrent


@dataclass
class CostBreakdown:
    attn_ms: float
    mlp_ms: float
    kv_write_ms: float
    promote_ms: float
    recompute_ms: float
    lookup_ms: float
    total_ms: float
    dominant: str
    kv_bytes: float
    cached_bytes: float
    components: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        out = dict(self.__dict__)
        out["components"] = dict(self.components)
        return out


def _core_flops(shape: ModelShape, kind: str, tokens: int, batch: int, decode: int = 0) -> float:
    if kind == "recurrent":
        return shape.recurrent_flops(tokens, batch, decode)
    return shape.attention_flops(tokens, batch, decode)


def breakdown(
    fmt: KVFormat,
    shape: ModelShape,
    hw: Hardware,
    work: Workload,
) -> CostBreakdown:
    """Per-request cost breakdown, in milliseconds."""
    if not 0.0 <= work.hit_rate <= 1.0:
        raise ValueError("hit_rate must be in [0, 1]")
    if not 0.0 <= work.kv_read_fraction <= 1.0:
        raise ValueError("kv_read_fraction must be in [0, 1]")
    if work.compute_kind not in ("dense", "sparse", "recurrent"):
        raise ValueError("compute_kind must be dense | sparse | recurrent")

    prompt, out_tok, batch = work.prompt_tokens, work.output_tokens, work.batch
    total_tokens = prompt + out_tok
    sparse = 1.0 if work.compute_kind == "dense" else work.kv_read_fraction

    kv_bytes = fmt.bytes_for(total_tokens, shape.n_layers, batch)
    cached_bytes = fmt.bytes_for(prompt, shape.n_layers, batch) * work.hit_rate

    # 1. attention core: roofline over compute and KV reads
    core_flops = _core_flops(shape, work.compute_kind, prompt, batch, out_tok) * sparse
    read_bytes = fmt.bytes_for(total_tokens, shape.n_layers, batch) * work.kv_read_fraction
    attn_s = max(core_flops / hw.flops, read_bytes / hw.bandwidth(hw.cached_tier))

    # 2. the rest of the transformer (MLP + projections)
    mlp_s = shape.mlp_flops(prompt, batch, out_tok) / hw.flops

    # 3. writing the newly computed KV (prompt part not served from cache + generation)
    new_prompt_bytes = fmt.bytes_for(prompt, shape.n_layers, batch) * (1.0 - work.hit_rate)
    kv_write_s = (
        new_prompt_bytes + fmt.bytes_for(out_tok, shape.n_layers, batch)
    ) / hw.bandwidth(hw.kv_tier)

    # 4. promoting the cached prefix from whichever tier holds it
    promote_s = cached_bytes / hw.bandwidth(hw.cached_tier)

    # 5. recomputing the missed part of the prompt (core + MLP)
    missing = prompt * (1.0 - work.hit_rate)
    if missing:
        recompute_s = (
            _core_flops(shape, work.compute_kind, int(missing), batch) * sparse
            + shape.mlp_flops(int(missing), batch)
        ) / hw.flops
    else:
        recompute_s = 0.0

    # 6. prefix-cache metadata lookups
    lookup_s = max(1, total_tokens // 16) * hw.lookup_us_per_block * 1e-6

    parts = {
        "attn": attn_s * 1e3,
        "mlp": mlp_s * 1e3,
        "kv_write": kv_write_s * 1e3,
        "promote": promote_s * 1e3,
        "recompute": recompute_s * 1e3,
        "lookup": lookup_s * 1e3,
    }
    return CostBreakdown(
        attn_ms=parts["attn"],
        mlp_ms=parts["mlp"],
        kv_write_ms=parts["kv_write"],
        promote_ms=parts["promote"],
        recompute_ms=parts["recompute"],
        lookup_ms=parts["lookup"],
        total_ms=sum(parts.values()),
        dominant=max(parts, key=parts.get),
        kv_bytes=kv_bytes,
        cached_bytes=cached_bytes,
        components=parts,
    )


def sweep_hit_rate(
    fmt: KVFormat,
    shape: ModelShape,
    hw: Hardware,
    work: Workload,
    hit_rates: Optional[List[float]] = None,
) -> List[Dict[str, object]]:
    """Cost vs hit rate -- the sensitivity that motivates cache work."""
    rows: List[Dict[str, object]] = []
    for hr in (hit_rates or [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]):
        w = Workload(**{**work.__dict__, "hit_rate": hr})
        b = breakdown(fmt, shape, hw, w)
        rows.append({
            "format": fmt.name,
            "hit_rate": hr,
            "total_ms": b.total_ms,
            "attn_ms": b.attn_ms,
            "mlp_ms": b.mlp_ms,
            "recompute_ms": b.recompute_ms,
            "storage_ms": b.kv_write_ms + b.promote_ms,
            "dominant": b.dominant,
        })
    return rows
