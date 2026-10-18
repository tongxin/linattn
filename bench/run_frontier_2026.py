"""Focused quality/cost comparison for the 2025--2026 frontier variants.

This is deliberately a small, common-budget experiment rather than another
leaderboard.  It measures the properties the new papers claim to change:
selection error at a fixed read budget, actual entries read, state footprint, and
wall-clock latency on the same structured long-context input.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linattn.core import AttnConfig  # noqa: E402
from linattn.harness import compare  # noqa: E402
from linattn.registry import build  # noqa: E402
from linattn.reporting import save_report  # noqa: E402


NAMES = [
    "dense",
    "saga",
    "pisa",
    "jssa",
    "masa",
    "oval",
    "sfa",
    "random_eviction",
    "h2o",
    "snapkv",
    "tova",
    "dropkv",
    "restkv",
    "beaconkv",
]


def main() -> int:
    seq, budget = 512, 128
    cfg = AttnConfig(
        n_heads=8,
        n_kv_heads=4,
        head_dim=64,
        block_size=16,
        window_size=128,
        cache_budget=budget,
        topk_blocks=8,
        topk_tokens=budget,
        obs_window=8,
    )
    variants = []
    for name in NAMES:
        local = cfg.with_(extras={"feature_keep": 16}) if name == "sfa" else cfg
        variants.append(build(name, local))
    records = compare(variants, seq=seq, batch=1,
                      heads=8, kv_heads=4, dim=64, structured=True, chunk=128)
    for rec in records:
        rec["reads_per_query"] = rec.get("kv_reads", 0.0)
        rec["state_kib"] = rec.get("state_bytes", 0.0) / 1024
        rec["error"] = rec["extra"].get("error", "")
        rec["mass_recall"] = rec["extra"].get("mass_recall")
    path = save_report(
        "frontier_2026",
        records,
        title="2025–2026 frontier variants at T=512 and a 128-entry budget",
        columns=["variant", "family", "rel_err", "mass_recall", "reads_per_query",
                 "state_kib", "latency_ms", "error"],
        notes=(
            "One CPU run on the repository's structured input, not a reproduction "
            "of the authors' GPU benchmarks. `masa` is an exact-output-error oracle "
            "and is intentionally expensive. `oval` builds page bases in Python on "
            "every call; its latency is not representative of a fused kernel. The "
            "eviction variants receive the same protected prompt/window policy as "
            "the random baseline. Read this table as a sanity check on mechanism "
            "and cost accounting, not as evidence that one paper beats another."
        ),
    )
    print(f"wrote {path}")
    for rec in records:
        recall = "n/a" if rec["mass_recall"] is None else f"{rec['mass_recall']:.3f}"
        print(f"{rec['variant']:<10} err={rec['rel_err']:.4f} recall={recall:<6} "
              f"reads={rec['reads_per_query']:.1f} state={rec['state_kib']:.1f}KiB "
              f"{rec['latency_ms']:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
