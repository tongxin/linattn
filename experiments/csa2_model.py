"""CSA2 end to end: what sharing and low precision buy, and what they cost.

Three questions, answered with the components in ``linattn.kv``:

1. **Footprint** -- the reported 890 B/token versus what a composition of (MLA-sized
   latent, FP4, 4-layer sharing) actually yields, and what sharing group would be
   needed to reach the report.
2. **Sharing cost** -- on a stack where each layer's KV differs from its group
   representative by ``sigma``, how much output divergence does reuse introduce?
   (Sharing is an architectural choice made in pretraining; this shows the sensitivity
   rather than assuming it works.)
3. **Cost** -- the KV-layer cost breakdown at 128k/1M for the composed CSA2 spec, so
   the storage win can be read against the attention term it does *not* change.

Run::

    python experiments/csa2_model.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from linattn.core import AttnConfig  # noqa: E402
from linattn.kv import Hardware, ModelShape, Workload, breakdown  # noqa: E402
from linattn.kv.csa2 import CSA2StackConfig, compare_share  # noqa: E402
from linattn.kv.format import PRESETS, SPECS  # noqa: E402
from linattn.reporting import save_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--compress", type=int, default=32)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--topk-groups", type=int, default=8, dest="topk_groups")
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--sigmas", nargs="*", type=float, default=[0.0, 0.05, 0.2, 1.0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    attn = AttnConfig(n_heads=4, n_kv_heads=2, head_dim=64, window_size=args.window, topk_blocks=4)
    rows = []

    # 1. footprint: reported vs composed vs "what it takes"
    for name in ("csa2_reported", "csa2", "csa2_reach890"):
        fmt = PRESETS[name]
        per_token = fmt.bytes_for(1_000_000, 61) / 1_000_000
        rows.append({
            "experiment": "footprint",
            "config": name,
            "detail": fmt.note or "",
            "bytes_per_token": per_token,
            "kib_per_token": per_token / 1024,
            "sharing_group": SPECS.get(name).cross_layer_group if name in SPECS else 0,
            "bits": SPECS.get(name).quant.bits if name in SPECS else 4,
            "divergence": 0.0,
        })

    # 2. sharing cost vs layer correlation
    torch.manual_seed(args.seed)
    base = torch.randn(1, attn.n_kv_heads, args.seq, attn.head_dim)
    for sigma in args.sigmas:
        g = torch.Generator().manual_seed(args.seed + 7)
        q = torch.stack([torch.randn(1, attn.n_heads, args.seq, attn.head_dim, generator=g)
                         for _ in range(args.layers)])
        k = torch.stack([base + sigma * torch.randn(1, attn.n_kv_heads, args.seq, attn.head_dim, generator=g)
                         for _ in range(args.layers)])
        v = torch.stack([base + sigma * torch.randn(1, attn.n_kv_heads, args.seq, attn.head_dim, generator=g)
                         for _ in range(args.layers)])
        cfg = CSA2StackConfig(n_layers=args.layers, layer_group=args.group, compress_ratio=args.compress,
                              bits=args.bits, topk_groups=args.topk_groups, window=args.window,
                              seed=args.seed)
        res = compare_share(cfg, attn, q, k, v)
        rows.append({
            "experiment": "sharing",
            "config": f"sigma={sigma:g}",
            "detail": f"{args.group}-layer groups, fp{args.bits}, pool x{args.compress}",
            "bytes_per_token": res["bytes_per_token_shared"],
            "kib_per_token": res["bytes_per_token_shared"] / 1024,
            "sharing_group": args.group,
            "bits": args.bits,
            "divergence": res["sharing_delta"],
        })

    # 3. cost: does the storage win change the dominant term?
    shape, hw = ModelShape(), Hardware()
    for fmt_name in ("dense_fp16", "mla", "csa2"):
        for ctx in (32768, 131072, 1048576):
            b = breakdown(PRESETS[fmt_name], shape, hw, Workload(prompt_tokens=ctx, output_tokens=4096, hit_rate=0.9))
            rows.append({
                "experiment": "cost",
                "config": f"{fmt_name} @ {ctx}",
                "detail": b.dominant,
                "bytes_per_token": b.kv_bytes / (ctx + 4096),
                "kib_per_token": b.kv_bytes / (ctx + 4096) / 1024,
                "sharing_group": 0,
                "bits": 0,
                "divergence": b.attn_ms,
            })

    path = save_report(
        "csa2_model",
        rows,
        title="CSA2 end to end: footprint, sharing cost and the attention term it leaves alone",
        columns=["experiment", "config", "bytes_per_token", "kib_per_token", "sharing_group", "bits",
                 "divergence", "detail"],
        notes=(
            "Footprint: reported 890 B/token (V4.1-Flash) vs a composition of a "
            "576-value MLA latent at FP4 with 4-layer sharing, and the sharing group "
            "needed to approach the report. Sharing: output divergence between a stack "
            "with cross-layer reuse and the same stack without it, as the per-layer KV "
            "correlation (sigma) varies -- sigma=0 means layers agree exactly. It is an "
            "architectural property, not something reuse can create post hoc. Cost: "
            "`divergence` column holds attention ms for the cost rows; the point is "
            "that the storage win does not move the attention term."
        ),
    )
    print(f"wrote {path}\n")
    for r in rows:
        if r["experiment"] == "footprint":
            print(f"  {r['config']:<16} {r['bytes_per_token']:>8.1f} B/token  group={r['sharing_group']:>2}  {r['detail']}")
    for r in rows:
        if r["experiment"] == "sharing":
            print(f"  {r['config']:<16} divergence={r['divergence']:.4f}  bytes/token={r['bytes_per_token']:.1f}")
    for r in rows:
        if r["experiment"] == "cost":
            print(f"  {r['config']:<26} {r['bytes_per_token']:>9.0f} B/token  attn={r['divergence']:>9.1f} ms  dominant={r['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
