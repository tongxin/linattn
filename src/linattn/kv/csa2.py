"""End-to-end CSA2-style stack: compression + low precision + cross-layer reuse.

The single-layer mechanism lives in :class:`linattn.variants.csa2.CSA2Attention`.
This module wires it into a *stack* so the cross-layer part can be measured rather
than assumed:

* layers are partitioned by :class:`~linattn.kv.reuse.CrossLayerPlan`;
* the representative ("full") layer of each group computes and stores the compressed,
  quantised KV;
* the other layers in the group **read that copy** instead of their own;
* the report compares the shared stack against the per-layer baseline on the same
  inputs, so the cost of sharing is a measured divergence, not a claim.

The inputs are per-layer ``(q, k, v)`` tensors.  Real layers have correlated but not
identical KV -- sharing is an *architectural* choice made during pretraining (as in
CSA2 / IndexCache), which is why the experiment sweeps the correlation between
layers instead of pretending random layers can share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from linattn.core import AttnConfig
from linattn.kv.format import KVSpec
from linattn.kv.quant import QuantSpec
from linattn.kv.reuse import CrossLayerPlan
from linattn.variants.csa2 import CSA2Attention

__all__ = ["CSA2StackConfig", "CSA2Stack", "csa2_spec"]


@dataclass
class CSA2StackConfig:
    n_layers: int = 8
    layer_group: int = 4          # layers sharing one stored KV copy
    compress_ratio: int = 32      # pooling stride m
    bits: int = 4                 # 4 | 8 | 16
    quant_group: int = 32
    topk_groups: int = 8
    window: int = 128
    share: bool = True
    seed: int = 0


def csa2_spec(cfg: CSA2StackConfig, attn: AttnConfig) -> KVSpec:
    """Storage spec implied by an attention config (values before compression)."""
    return KVSpec(
        name=f"csa2(group={cfg.layer_group},bits={cfg.bits},m={cfg.compress_ratio})",
        values_per_token_per_layer=2 * attn.n_kv_heads * attn.head_dim,
        quant=QuantSpec(bits=cfg.bits, group_size=cfg.quant_group),
        compress_group=cfg.compress_ratio,
        cross_layer_group=cfg.layer_group if cfg.share else 1,
    )


class CSA2Stack:
    """A minimal multi-layer harness that exercises cross-layer KV reuse."""

    def __init__(self, cfg: CSA2StackConfig, attn: AttnConfig):
        self.cfg = cfg
        self.attn_cfg = attn
        self.plan = CrossLayerPlan(n_layers=cfg.n_layers, group_size=cfg.layer_group)
        extras = {
            "compress_ratio": cfg.compress_ratio,
            "bits": cfg.bits,
            "quant_group": cfg.quant_group,
            "topk_groups": cfg.topk_groups,
            "window": cfg.window,
            "cross_layer_group": cfg.layer_group if cfg.share else 1,
        }
        self.layer = CSA2Attention(attn.with_(extras=extras))

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Dict[str, object]:
        """``q,k,v``: ``(n_layers, B, H, T, D)`` (K/V may be ``n_kv_heads``)."""
        n_layers = q.shape[0]
        if n_layers != self.cfg.n_layers:
            raise ValueError(f"expected {self.cfg.n_layers} layers, got {n_layers}")
        outs: List[torch.Tensor] = []
        reads: List[float] = []
        with torch.no_grad():
            for layer in range(n_layers):
                src = self.plan.representative(layer) if self.cfg.share else layer
                out, _, stats = self.layer.forward(q[layer], k[src], v[src], chunk=q.shape[2], collect_stats=True)
                outs.append(out)
                reads.append(float(stats[-1]["reads_per_query"]))
        stacked = torch.stack(outs)
        spec = csa2_spec(self.cfg, self.attn_cfg)
        seq = q.shape[-2]
        per_seq_bytes = spec.bytes_for(seq, n_layers) / max(1, q.shape[1])
        return {
            "out": stacked,
            "reads_per_query": sum(reads) / len(reads),
            "copies": self.plan.copies if self.cfg.share else n_layers,
            "bytes_per_token": per_seq_bytes / seq,
            "spec": spec.describe(),
        }


def compare_share(
    cfg: CSA2StackConfig,
    attn: AttnConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    baseline_bits: int = 16,
) -> Dict[str, float]:
    """Stack with sharing vs without, and vs a full-precision per-layer baseline."""
    shared = CSA2Stack(cfg, attn)
    per_layer_cfg = CSA2StackConfig(**{**cfg.__dict__, "share": False})
    per_layer = CSA2Stack(per_layer_cfg, attn)
    ref_cfg = CSA2StackConfig(**{**per_layer_cfg.__dict__, "bits": baseline_bits})
    reference = CSA2Stack(ref_cfg, attn)

    ref = reference.run(q, k, v)
    base = per_layer.run(q, k, v)
    sh = shared.run(q, k, v)
    rel = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-12))
    return {
        "divergence_shared_vs_ref": rel(sh["out"], ref["out"]),
        "divergence_unshared_vs_ref": rel(base["out"], ref["out"]),
        "sharing_delta": rel(sh["out"], base["out"]),
        "bytes_per_token_shared": sh["bytes_per_token"],
        "bytes_per_token_unshared": base["bytes_per_token"],
        "copies_shared": sh["copies"],
        "copies_unshared": base["copies"],
        "reads_per_query": sh["reads_per_query"],
    }
