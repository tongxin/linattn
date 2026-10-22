"""Validate the approximation metric hierarchy on representative variants.

This is the bridge between theory and measurement: every variant is evaluated on
the same inputs and budget, but mask-based selectors expose a distribution while
Performer/Linformer expose only an output. Missing cells are intentional.
"""

from __future__ import annotations

import sys
import statistics
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from approximation_metrics import (  # noqa: E402
    attention_operator_metrics,
    cosine_distance,
    relative_l2,
)
from linattn.core import AttnConfig, dense_attention  # noqa: E402
from linattn.harness import make_inputs, structured_inputs  # noqa: E402
from linattn.registry import build  # noqa: E402
from linattn.reporting import save_report  # noqa: E402


def _masked_distribution(weights: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    selected = weights * support
    return selected / selected.sum(-1, keepdim=True).clamp_min(1e-12)


def _evaluate(name: str, variant, q, k, v, input_kind: str) -> Dict:
    causal = bool(getattr(variant, "causal", True))
    o_ref, p_ref = dense_attention(q, k, v, causal=causal, return_weights=True)
    out, _, stats = variant.forward(q, k, v, chunk=q.shape[-2], collect_stats=True)
    row: Dict = {
        "variant": name,
        "input": input_kind,
        "reference_mode": "causal" if causal else "noncausal",
        "output_rel_l2": relative_l2(o_ref, out),
        "output_cosine_distance": cosine_distance(o_ref, out),
        "max_abs_error": float((o_ref - out).abs().max()),
        "reads_per_query": float(stats[-1].get("reads_per_query", 0.0)),
    }
    masks = getattr(variant, "_masks", [])
    if masks:
        support = masks[-1].to(q.device)
        p_hat = _masked_distribution(p_ref, support)
        dist = attention_operator_metrics(p_ref, p_hat, o_ref, out, k=8)
        row.update({f"distribution_{key}": value for key, value in dist.items()})
        row["support_fraction"] = float(support.float().mean())
    return row


def run(input_kind: str, cfg: AttnConfig, seed: int = 11) -> list[Dict]:
    t = 128
    q, k, v = (
        structured_inputs(1, cfg.n_heads, t, cfg.head_dim, kv_heads=cfg.n_kv_heads, seed=seed)
        if input_kind == "structured"
        else make_inputs(1, cfg.n_heads, t, cfg.head_dim, kv_heads=cfg.n_kv_heads, seed=seed)
    )
    q, k, v = q.double(), k.double(), v.double()
    configs = {
        "quest": cfg,
        "xattention": cfg,
        "moba": cfg,
        "minference": cfg.with_(extras={"pattern": "block_sparse"}),
        "performer": cfg.with_(extras={"features": 32}),
        "linformer": cfg.with_(extras={"linformer_k": 32}),
    }
    rows = []
    for name, local_cfg in configs.items():
        rows.append(_evaluate(name, build(name, local_cfg), q, k, v, input_kind))
    return rows


def main() -> int:
    cfg = AttnConfig(
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        block_size=16,
        window_size=32,
        topk_blocks=4,
        topk_tokens=32,
        cache_budget=64,
        sink_size=2,
    )
    raw = []
    for kind in ("random", "structured"):
        for seed in range(3):
            raw.extend(run(kind, cfg, seed=seed))
    rows = []
    for variant in sorted({r["variant"] for r in raw}):
        for input_kind in ("random", "structured"):
            subset = [r for r in raw if r["variant"] == variant and r["input"] == input_kind]
            rows.append({
                "variant": variant,
                "input": input_kind,
                "reference_mode": subset[0]["reference_mode"],
                "n_seeds": len(subset),
                "output_rel_l2_mean": statistics.mean(r["output_rel_l2"] for r in subset),
                "output_rel_l2_std": statistics.pstdev(r["output_rel_l2"] for r in subset),
                "output_cosine_distance_mean": statistics.mean(r["output_cosine_distance"] for r in subset),
                "max_abs_error_mean": statistics.mean(r["max_abs_error"] for r in subset),
                "reads_per_query_mean": statistics.mean(r["reads_per_query"] for r in subset),
                "support_fraction_mean": (
                    statistics.mean(r["support_fraction"] for r in subset)
                    if "support_fraction" in subset[0] else None
                ),
                "attention_js_mean": (
                    statistics.mean(r["distribution_attention_js"] for r in subset)
                    if "distribution_attention_js" in subset[0] else None
                ),
                "topk_agreement_mean": (
                    statistics.mean(r["distribution_topk_agreement"] for r in subset)
                    if "distribution_topk_agreement" in subset[0] else None
                ),
            })
    path = save_report(
        "approximation_validation",
        rows,
        title="Metric hierarchy validation on representative attention variants",
        columns=["variant", "input", "reference_mode", "n_seeds", "output_rel_l2_mean",
                 "output_rel_l2_std", "output_cosine_distance_mean", "max_abs_error_mean",
                 "reads_per_query_mean", "support_fraction_mean", "attention_js_mean",
                 "topk_agreement_mean"],
        notes=(
            "T=128, H=4, H_kv=2, d=32, four 16-token blocks, 32-token token budget. "
            "Mask-based selectors expose both support and a renormalized attention "
            "distribution; Performer/Linformer expose only an output, so distribution "
            "cells are blank. Linformer is compared with a noncausal dense reference "
            "because its sequence projection mixes positions; other variants use the "
            "causal reference. This is a metric-validation experiment, not a quality "
            "leaderboard. `structured` plants retrieval spikes; `random` is the control."
        ),
    )
    print(f"wrote {path}")
    for r in rows:
        print(f"{r['input']:<10} {r['variant']:<12} "
              f"out={r['output_rel_l2_mean']:.4f}±{r['output_rel_l2_std']:.4f} "
              f"support={r['support_fraction_mean'] if r['support_fraction_mean'] is not None else float('nan'):.3f} "
              f"JS={r['attention_js_mean'] if r['attention_js_mean'] is not None else float('nan'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
