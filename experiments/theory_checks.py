"""Empirical checks of the approximation bounds and predictions in the report.

Three mechanisms are tested separately:
* sparse support: the value-weighted error should be bounded by dropped mass and
  the value norm;
* random features: variance should fall as feature count grows, even if a single
  seed is noisy;
* precision: FP16 should be exact, FP8 should be closer than FP4.
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from linattn.core import AttnConfig, dense_attention, repeat_kv  # noqa: E402
from linattn.harness import make_inputs, structured_inputs  # noqa: E402
from linattn.kv.quant import QuantSpec, roundtrip  # noqa: E402
from linattn.metrics import rel_error  # noqa: E402
from linattn.registry import build  # noqa: E402
from linattn.reporting import save_report  # noqa: E402


def _inputs(kind: str, seed: int, t: int = 128):
    if kind == "retrieval":
        return structured_inputs(1, 4, t, 32, n_spikes=1, spike_width=8, seed=seed, kv_heads=2)
    if kind == "dispersed":
        q, k, v = make_inputs(1, 4, t, 32, seed=seed, kv_heads=2)
        for start in (12, 44, 76, 108):
            k[:, :, start : start + 8] *= 2.5
        return q, k, v
    return make_inputs(1, 4, t, 32, seed=seed, kv_heads=2)


def _mask_support(variant, q, k, v):
    out, _, _ = variant.forward(q, k, v, chunk=q.shape[-2], collect_stats=True)
    masks = getattr(variant, "_masks", [])
    return out, masks[-1] if masks else None


def selection_bound_rows() -> list[Dict]:
    cfg = AttnConfig(
        n_heads=4, n_kv_heads=2, head_dim=32, block_size=16,
        window_size=16, topk_blocks=2, topk_tokens=32, sink_size=0,
    )
    rows: list[Dict] = []
    for kind in ("random", "retrieval", "dispersed"):
        for seed in range(3):
            q, k, v = _inputs(kind, seed)
            q, k, v = q.double(), k.double(), v.double()
            o_ref, p_ref = dense_attention(q, k, v, causal=True, return_weights=True)
            v_rep = repeat_kv(v, cfg.group_size)
            vmax = v_rep.norm(dim=-1).amax(-1)                # (B,H)
            for name in ("quest", "moba", "minference", "xattention"):
                local = cfg.with_(extras={"pattern": "block_sparse"}) if name == "minference" else cfg
                out, support = _mask_support(build(name, local), q, k, v)
                if support is None:
                    continue
                selected = (p_ref * support).sum(-1, keepdim=True).clamp_min(1e-12)
                p_sel = p_ref * support / selected
                o_sel = torch.matmul(p_sel, v_rep)
                err = float((o_ref - o_sel).norm())
                retained = (p_ref * support).sum(-1)             # (B,H,Tq)
                dropped_per_row = 1.0 - retained
                dropped = float(dropped_per_row.mean())
                bound_rows = 2.0 * dropped_per_row * vmax.unsqueeze(-1)
                bound = float(torch.sqrt(bound_rows.square().sum()))
                rows.append({
                    "study": "selection_bound",
                    "input": kind,
                    "seed": seed,
                    "variant": name,
                    "dropped_mass": dropped,
                    "mass_recall": 1.0 - dropped,
                    "value_norm_max": float(vmax.max()),
                    "output_l2_error": err,
                    "bound_2_delta_vmax": bound,
                    "bound_ratio": err / max(bound, 1e-12),
                    "satisfied": float(err <= bound + 1e-10),
                    "output_rel_l2": rel_error(out, o_ref),
                })
    return rows


def feature_rows() -> list[Dict]:
    cfg = AttnConfig(n_heads=4, n_kv_heads=2, head_dim=32, window_size=32)
    rows = []
    for kind in ("random", "dispersed"):
        for seed in range(5):
            q, k, v = _inputs(kind, seed)
            q, k, v = q.double(), k.double(), v.double()
            ref = dense_attention(q, k, v)
            for features in (4, 8, 16, 32, 64, 128, 256):
                layer = build("performer", cfg.with_(extras={"features": features}))
                out, _ = layer.forward(q, k, v, chunk=q.shape[-2])
                rows.append({
                    "study": "feature_convergence",
                    "input": kind,
                    "seed": seed,
                    "variant": "performer",
                    "features": features,
                    "rel_l2_error": rel_error(out, ref),
                })
    return rows


def quant_rows() -> list[Dict]:
    cfg = AttnConfig(n_heads=4, n_kv_heads=2, head_dim=32)
    rows = []
    for kind in ("random", "dispersed"):
        for seed in range(3):
            q, k, v = _inputs(kind, seed)
            q, k, v = q.double(), k.double(), v.double()
            ref = dense_attention(q, k, v)
            for bits in (16, 8, 4):
                spec = QuantSpec(bits=bits, group_size=16)
                kq = roundtrip(k, spec).to(k.dtype)
                vq = roundtrip(v, spec).to(v.dtype)
                out = dense_attention(q, kq, vq)
                rows.append({
                    "study": "precision",
                    "input": kind,
                    "seed": seed,
                    "variant": "dense_kv",
                    "bits": bits,
                    "rel_l2_error": rel_error(out, ref),
                })
    return rows


def _summary(rows: list[Dict]) -> list[Dict]:
    out = []
    for study, input_kind, variant in sorted({(r["study"], r["input"], r["variant"]) for r in rows}):
        subset = [r for r in rows if (r["study"], r["input"], r["variant"]) == (study, input_kind, variant)]
        if study == "feature_convergence":
            for features in sorted({r["features"] for r in subset}):
                vals = [r["rel_l2_error"] for r in subset if r["features"] == features]
                out.append({"study": study, "input": input_kind, "variant": variant,
                            "setting": f"features={features}", "mean_rel_error": statistics.mean(vals),
                            "std_rel_error": statistics.pstdev(vals), "n": len(vals)})
        elif study == "precision":
            for bits in sorted({r["bits"] for r in subset}):
                vals = [r["rel_l2_error"] for r in subset if r["bits"] == bits]
                out.append({"study": study, "input": input_kind, "variant": variant,
                            "setting": f"fp{bits}", "mean_rel_error": statistics.mean(vals),
                            "std_rel_error": statistics.pstdev(vals), "n": len(vals)})
        else:
            out.append({"study": study, "input": input_kind, "variant": variant,
                        "setting": "all", "mean_rel_error": statistics.mean(r["output_rel_l2"] for r in subset),
                        "std_rel_error": statistics.pstdev(r["output_rel_l2"] for r in subset),
                        "mean_bound_ratio": statistics.mean(r["bound_ratio"] for r in subset),
                        "bound_satisfied_rate": statistics.mean(r["satisfied"] for r in subset),
                        "n": len(subset)})
    return out


def main() -> int:
    rows = selection_bound_rows() + feature_rows() + quant_rows()
    summary = _summary(rows)
    path = save_report(
        "theory_checks",
        summary,
        title="Checks of sparse-support bounds, feature convergence and KV precision",
        columns=["study", "input", "variant", "setting", "mean_rel_error",
                 "std_rel_error", "mean_bound_ratio", "bound_satisfied_rate", "n"],
        notes=(
            "Synthetic CPU validation across multiple seeds. Selection-bound rows "
            "test ||O_ref-O_selected|| <= 2·dropped_mass·Vmax. Feature rows test the "
            "expected variance reduction of positive random features; a non-monotone "
            "single seed is reported rather than hidden. Precision rows apply a "
            "group-wise KV codec before the exact dense attention operator."
        ),
    )
    print(f"wrote {path}")
    for r in summary:
        print(f"{r['study']:<20} {r['input']:<10} {r['variant']:<12} {r['setting']:<16} "
              f"err={r['mean_rel_error']:.4f}" + (
                  f" bound_ratio={r.get('mean_bound_ratio', 0):.3f}" if r["study"] == "selection_bound" else ""
              ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
