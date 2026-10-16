"""Run every variant through the same quality/cost measurement and write a report.

Usage::

    python bench/run_suite.py --seq 2048 --budget 256 --topk-blocks 4

The output is ``reports/suite_<seq>_<budget>.{json,md}``: relative error against
dense attention, attention-mass recall of the selection, KV entries read per
query, persistent state bytes and wall-clock latency, for every implemented
variant at one common budget.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linattn.core import AttnConfig  # noqa: E402
from linattn.core import dense_attention  # noqa: E402
from linattn.harness import compare, structured_inputs  # noqa: E402
from linattn.metrics import attention_mass, mass_recall  # noqa: E402
from linattn.registry import VARIANTS, build  # noqa: E402
from linattn.reporting import save_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=4, dest="kv_heads")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64, dest="block_size")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--topk-blocks", type=int, default=4, dest="topk_blocks")
    parser.add_argument("--topk-tokens", type=int, default=512, dest="topk_tokens")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--no-structured", action="store_true")
    args = parser.parse_args()

    cfg = AttnConfig(
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        head_dim=args.dim,
        block_size=args.block_size,
        window_size=args.window,
        cache_budget=args.budget,
        topk_blocks=args.topk_blocks,
        topk_tokens=args.topk_tokens,
    )
    names = args.variants or list(VARIANTS)
    variants = [build(name, cfg) for name in names]
    records = compare(
        variants,
        seq=args.seq,
        batch=args.batch,
        heads=args.heads,
        kv_heads=args.kv_heads,
        dim=args.dim,
        structured=not args.no_structured,
        chunk=args.chunk,
    )
    for rec in records:
        rec["reads_per_query"] = rec.get("kv_reads", 0.0)
        rec["state_kib"] = rec.get("state_bytes", 0.0) / 1024
        rec["error"] = rec["extra"].get("error", "")
        rec["mass_recall"] = rec["extra"].get("mass_recall")
    path = save_report(
        f"suite_{args.seq}_{args.budget}",
        records,
        title=f"Variant comparison at T={args.seq}, budget={args.budget}",
        columns=["variant", "family", "rel_err", "mass_recall", "reads_per_query", "state_kib", "latency_ms", "error"],
        notes=(
            "Inputs have planted retrieval structure (spiked key norms) so that "
            "selection quality is measured against a non-uniform target. "
            "`rel_err` is relative L2 against dense attention on the same inputs, "
            "`mass_recall` is the fraction of attention mass the selection recovers "
            "(block level, full-sequence read), and "
            "`reads_per_query` is the mean number of KV entries actually read. "
            "Read the pairs, not the columns: a high recall at a small read budget is "
            "what a selector is for. Note also that on this input (planted retrieval "
            "spikes on top of a recency gradient) mass recall is easy even for a "
            "randomly initialised learned selector -- that is the same effect the "
            "eviction literature reports, and it is why `rel_err` and recall must be "
            "read together. `dense` and `mla` are structural baselines rather than "
            "selections, so their recall is not applicable and MLA's `rel_err` simply "
            "measures the difference between its random up-projection and the identity."
        ),
    )
    print(f"wrote {path}")
    for rec in records:
        recall = "n/a" if rec["mass_recall"] is None else f"{rec['mass_recall']:.3f}"
        print(f"{rec['variant']:<18} rel_err={rec['rel_err']:.4f}  recall={recall:<6} reads={rec['reads_per_query']:.1f}"
              f"  state={rec['state_kib']:.1f} KiB  {rec['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
