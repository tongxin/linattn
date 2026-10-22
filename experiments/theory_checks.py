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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from approximation_certificates import quantized_kv_certificate, sparse_support_certificate  # noqa: E402
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
            for name in ("quest", "moba", "minference", "xattention"):
                local = cfg.with_(extras={"pattern": "block_sparse"}) if name == "minference" else cfg
                out, support = _mask_support(build(name, local), q, k, v)
                if support is None:
                    continue
                cert = sparse_support_certificate(p_ref, v_rep, support)
                err = float(cert["error_norm"].sum())
                bound = float(cert["bound"].sum())
                dropped = float(cert["omitted_mass"].mean())
                exact_identity_error = float(
                    (cert["error"] - cert["exact_omitted"]).abs().max()
                )
                implementation_parity = float((out - cert["out_selected"]).abs().max())
                rows.append({
                    "study": "selection_bound",
                    "input": kind,
                    "seed": seed,
                    "variant": name,
                    "dropped_mass": dropped,
                    "mass_recall": 1.0 - dropped,
                    "value_norm_max": float(v_rep.norm(dim=-1).max()),
                    "output_l2_error": err,
                    "bound_2_delta_vmax": bound,
                    "bound_ratio": err / max(bound, 1e-12),
                    "satisfied": float(err <= bound + 1e-10),
                    "output_rel_l2": rel_error(out, o_ref),
                    "exact_decomposition_error": exact_identity_error,
                    "implementation_parity": implementation_parity,
                    "error_per_dropped_mass": err / max(dropped, 1e-12),
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
            k_rep = repeat_kv(k, cfg.group_size)
            v_rep = repeat_kv(v, cfg.group_size)
            for bits in (16, 8, 4):
                spec = QuantSpec(bits=bits, group_size=16)
                kq = roundtrip(k, spec).to(k.dtype)
                vq = roundtrip(v, spec).to(v.dtype)
                kq_rep = repeat_kv(kq, cfg.group_size)
                vq_rep = repeat_kv(vq, cfg.group_size)
                out = dense_attention(q, kq, vq)
                cert = quantized_kv_certificate(q, k_rep, v_rep, kq_rep, vq_rep)
                total = cert["total_error"].norm(dim=-1)
                bound = cert["total_bound"]
                rows.append({
                    "study": "precision",
                    "input": kind,
                    "seed": seed,
                    "variant": "dense_kv",
                    "bits": bits,
                    "rel_l2_error": rel_error(out, ref),
                    "attention_tv": float(cert["attention_tv"].mean()),
                    "attention_tv_bound": float(cert["attention_tv_bound"].mean()),
                    "key_error_l2": float(cert["key_error"].norm(dim=-1).mean()),
                    "value_error_l2": float(cert["value_error"].norm(dim=-1).mean()),
                    "certificate_ratio": float((total / bound.clamp_min(1e-12)).mean()),
                    "certificate_satisfied": float((total <= bound + 1e-8).float().mean()),
                    "exact_decomposition_error": float(
                        (cert["total_error"] - cert["key_error"] - cert["value_error"]).abs().max()
                    ),
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
                detail = [r for r in subset if r["bits"] == bits]
                out.append({"study": study, "input": input_kind, "variant": variant,
                            "setting": f"fp{bits}", "mean_rel_error": statistics.mean(vals),
                            "std_rel_error": statistics.pstdev(vals), "n": len(vals),
                            "mean_attention_tv": statistics.mean(r["attention_tv"] for r in detail),
                            "max_attention_tv_bound": max(r["attention_tv_bound"] for r in detail),
                            "max_certificate_ratio": max(r["certificate_ratio"] for r in detail),
                            "certificate_satisfied_rate": statistics.mean(
                                r["certificate_satisfied"] for r in detail
                            ),
                            "max_exact_decomposition_error": max(
                                r["exact_decomposition_error"] for r in detail
                            )})
        else:
            out.append({"study": study, "input": input_kind, "variant": variant,
                        "setting": "all", "mean_rel_error": statistics.mean(r["output_rel_l2"] for r in subset),
                        "std_rel_error": statistics.pstdev(r["output_rel_l2"] for r in subset),
                        "mean_bound_ratio": statistics.mean(r["bound_ratio"] for r in subset),
                        "bound_satisfied_rate": statistics.mean(r["satisfied"] for r in subset),
                        "n": len(subset),
                        "max_exact_decomposition_error": max(
                            r["exact_decomposition_error"] for r in subset
                        ),
                        "max_implementation_parity_error": max(
                            r["implementation_parity"] for r in subset
                        ),
                        "mean_error_per_dropped_mass": statistics.mean(
                            r["error_per_dropped_mass"] for r in subset
                        )})
    return out


def main() -> int:
    rows = selection_bound_rows() + feature_rows() + quant_rows()
    summary = _summary(rows)
    path = save_report(
        "theory_checks",
        summary,
        title="Checks of attention error identities, bounds, feature convergence and KV precision",
        columns=["study", "input", "variant", "setting", "mean_rel_error",
                 "std_rel_error", "mean_bound_ratio", "bound_satisfied_rate",
                 "mean_attention_tv", "max_attention_tv_bound", "max_certificate_ratio",
                 "certificate_satisfied_rate", "max_exact_decomposition_error",
                 "max_implementation_parity_error", "mean_error_per_dropped_mass", "n"],
        notes=(
            "Synthetic CPU validation across multiple seeds. Selection rows check the "
            "exact identity O_ref-O_selected=(1-a)(mu_omitted-mu_selected), selector "
            "implementation parity, and the conservative bound ||error||<=2(1-a)Vmax. "
            "Precision rows split total error into key-induced and value-induced terms "
            "and check TV(softmax(z),softmax(z+delta))<=tanh(max|delta|). Feature rows "
            "test the expected variance reduction of positive random features. The "
            "bounds are local to one attention operation, not model-level guarantees."
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
