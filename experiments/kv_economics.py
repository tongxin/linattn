"""KV-layer economics: where the time goes, and what a cache hit is worth.

Two experiments over one cost model:

1. **Format sweep** -- dense fp16 / fp8, MLA, CSA2 and a recurrent state across
   context lengths, with the hit rate as an explicit parameter. The output names the
   dominant term in each cell: that is the instrument for "what should I optimise".
2. **Trace simulation** -- agentic sessions with tool turns and idle gaps, run
   through the prefix cache under LRU / TTL-LRU / FIFO / random plus a Belady bound at
   several cache capacities, then fed back into the cost model. A policy change or a
   capacity change is a *cost* change, not just a miss statistic.

Run::

    python experiments/kv_economics.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linattn.kv import (  # noqa: E402
    PRESETS,
    CacheConfig,
    Hardware,
    ModelShape,
    Workload,
    agentic_trace,
    breakdown,
    oracle_hits,
    simulate,
    trace_summary,
)
from linattn.reporting import save_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", nargs="*", type=int, default=[8192, 32768, 131072, 1048576])
    parser.add_argument("--formats", nargs="*",
                        default=["dense_fp16", "dense_fp8", "mla", "csa2", "gdn_state"])
    parser.add_argument("--hit-rates", nargs="*", type=float, default=[0.0, 0.5, 0.9])
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--gap-mean", type=float, default=60.0)
    parser.add_argument("--capacities", nargs="*", type=int, default=[256, 1024, 4096],
                        help="prefix-cache capacity in 16-token blocks")
    parser.add_argument("--ttl", type=float, default=300.0,
                        help="idle TTL used by the ttl_lru policy, seconds")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    shape = ModelShape()
    hw = Hardware()
    rows = []

    for name in args.formats:
        fmt = PRESETS[name]
        kind = "recurrent" if fmt.is_state_based else "dense"
        for ctx in args.contexts:
            for hr in args.hit_rates:
                work = Workload(prompt_tokens=ctx, output_tokens=4096, hit_rate=hr,
                                compute_kind=kind)
                b = breakdown(fmt, shape, hw, work)
                rows.append({
                    "experiment": "format",
                    "format": name,
                    "capacity_blocks": 0,
                    "context": ctx,
                    "hit_rate": hr,
                    "kv_gib": b.kv_bytes / 2**30,
                    "attn_ms": b.attn_ms,
                    "mlp_ms": b.mlp_ms,
                    "recompute_ms": b.recompute_ms,
                    "storage_ms": b.kv_write_ms + b.promote_ms,
                    "lookup_ms": b.lookup_ms,
                    "total_ms": b.total_ms,
                    "dominant": b.dominant,
                    "evictions": 0,
                    "expirations": 0,
                    "recompute_tokens": 0,
                })

    trace = agentic_trace(n_sessions=args.sessions, turns_per_session=args.turns,
                          gap_mean_s=args.gap_mean, seed=args.seed)
    summary = trace_summary(trace)

    # ---- 1b. algorithm leverage at a fixed format --------------------------------
    # Same KV bytes, different *attention algorithms*: this isolates compute from
    # storage, which is the distinction the model exists to make.
    for ctx in args.contexts:
        for kind, frac in (("dense", 1.0), ("sparse", 0.1), ("sparse", 0.02)):
            work = Workload(prompt_tokens=ctx, output_tokens=4096, hit_rate=0.9,
                            kv_read_fraction=frac, compute_kind=kind)
            b = breakdown(PRESETS["mla"], shape, hw, work)
            rows.append({
                "experiment": "algorithm",
                "format": f"{kind}/{frac:g}",
                "capacity_blocks": 0,
                "context": ctx,
                "hit_rate": 0.9,
                "kv_gib": b.kv_bytes / 2**30,
                "attn_ms": b.attn_ms,
                "mlp_ms": b.mlp_ms,
                "recompute_ms": b.recompute_ms,
                "storage_ms": b.kv_write_ms + b.promote_ms,
                "lookup_ms": b.lookup_ms,
                "total_ms": b.total_ms,
                "dominant": b.dominant,
                "evictions": 0,
                "expirations": 0,
                "recompute_tokens": 0,
            })
        work = Workload(prompt_tokens=ctx, output_tokens=4096, hit_rate=0.9, compute_kind="recurrent")
        b = breakdown(PRESETS["gdn_state"], shape, hw, work)
        rows.append({
            "experiment": "algorithm", "format": "recurrent", "capacity_blocks": 0,
            "context": ctx, "hit_rate": 0.9, "kv_gib": b.kv_bytes / 2**30,
            "attn_ms": b.attn_ms, "mlp_ms": b.mlp_ms, "recompute_ms": b.recompute_ms,
            "storage_ms": b.kv_write_ms + b.promote_ms, "lookup_ms": b.lookup_ms,
            "total_ms": b.total_ms, "dominant": b.dominant, "evictions": 0,
            "expirations": 0, "recompute_tokens": 0,
        })

    for policy in ("lru", "ttl_lru", "fifo", "random"):
        for cap in args.capacities:
            cfg = CacheConfig(capacity_blocks=cap, policy=policy,
                              ttl_s=args.ttl if policy == "ttl_lru" else None)
            res = simulate(trace, cfg)
            work = Workload(prompt_tokens=int(summary["mean_blocks"] * 16),
                            output_tokens=512, hit_rate=res.hit_rate)
            b = breakdown(PRESETS["mla"], shape, hw, work)
            rows.append({
                "experiment": "policy",
                "format": policy,
                "capacity_blocks": cap,
                "context": work.prompt_tokens,
                "hit_rate": res.hit_rate,
                "kv_gib": b.kv_bytes / 2**30,
                "attn_ms": b.attn_ms,
                    "mlp_ms": b.mlp_ms,
                "recompute_ms": b.recompute_ms,
                "storage_ms": b.kv_write_ms + b.promote_ms,
                "lookup_ms": b.lookup_ms,
                "total_ms": b.total_ms,
                "dominant": b.dominant,
                "evictions": res.evictions,
                "expirations": res.expirations,
                "recompute_tokens": res.recompute_tokens,
            })

    for cap in args.capacities:
        hit, total = oracle_hits(trace, cap)
        rows.append({
            "experiment": "policy",
            "format": "belady_oracle",
            "capacity_blocks": cap,
            "context": int(summary["mean_blocks"] * 16),
            "hit_rate": hit / total if total else 0.0,
            "kv_gib": 0.0,
            "attention_ms": 0.0,
            "recompute_ms": 0.0,
            "storage_ms": 0.0,
            "lookup_ms": 0.0,
            "total_ms": 0.0,
            "dominant": "bound",
            "evictions": 0,
            "expirations": 0,
            "recompute_tokens": int((total - hit) * 16),
        })

    notes = (
        "Roofline cost model: attention = max(FLOPs, KV reads / HBM bandwidth), plus "
        "KV writes, tier promotion, recompute-on-miss and prefix-cache lookups. 61 "
        "layers, 32 KV heads x 128, fp16 tensor cores, H100-class bandwidths. The "
        "state format is charged linear-time FLOPs (it is a different algorithm, not "
        "just a smaller cache). `dominant` names the largest component -- that is the "
        f"point of the table. Trace: {summary['requests']} requests, "
        f"{summary['sessions']} sessions over {summary['span_s']:.0f} s, mean "
        f"{summary['mean_blocks']:.0f} blocks/request "
        f"({summary['mean_blocks'] * 16:.0f} tokens); capacities "
        f"{args.capacities} blocks x 16 tokens, ttl_lru TTL {args.ttl:.0f} s. "
        "Belady is a block-level upper bound, optimistic for prefix reuse."
    )
    path = save_report(
        "kv_economics",
        rows,
        title="KV layer economics: format, context and the value of a cache hit",
        columns=["experiment", "format", "capacity_blocks", "context", "hit_rate", "kv_gib",
                 "attn_ms", "mlp_ms", "recompute_ms", "storage_ms", "total_ms", "dominant"],
        notes=notes,
    )
    print(f"wrote {path}")

    print(f"\n{'format':<12}{'ctx':>9}{'hit':>6}{'KV GiB':>9}{'attn ms':>10}{'recomp ms':>11}{'total ms':>10}  dominant")
    for r in rows:
        if r["experiment"] == "format":
            print(f"{r['format']:<12}{r['context']:>9}{r['hit_rate']:>6.2f}{r['kv_gib']:>9.2f}"
                  f"{r['attn_ms']:>10.1f}{r['recompute_ms']:>11.1f}{r['total_ms']:>10.1f}  {r['dominant']}")

    print(f"\n{'policy':<16}{'cap':>6}{'hit rate':>9}{'evict':>7}{'expire':>8}{'recompute tok':>15}")
    for r in rows:
        if r["experiment"] == "policy":
            print(f"{r['format']:<16}{r['capacity_blocks']:>6}{r['hit_rate']:>9.3f}"
                  f"{r['evictions']:>7}{r['expirations']:>8}{r['recompute_tokens']:>15}")

    print(f"\n{'algorithm':<14}{'ctx':>9}{'attn ms':>10}{'mlp ms':>9}{'total ms':>11}  dominant")
    for r in rows:
        if r["experiment"] == "algorithm":
            print(f"{r['format']:<14}{r['context']:>9}{r['attn_ms']:>10.1f}"
                  f"{r['mlp_ms']:>9.1f}{r['total_ms']:>11.1f}  {r['dominant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
