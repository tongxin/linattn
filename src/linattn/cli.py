"""Small CLI: ``python -m linattn.cli <command>``.

Commands are deliberately few and map onto the questions the repo answers:

``backends``   what can this machine actually run?
``variants``   the variant catalogue, with family and citation
``cost``       analytic cost model for a sequence length
``select``     run one variant and report error vs. dense + attention-mass recall
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import torch

from linattn.core import dense_attention
from linattn.harness import run_variant, structured_inputs, make_inputs
from linattn.kernels import describe_backends
from linattn.metrics import attention_mass, efficiency_table, mass_recall, text_table
from linattn.registry import MIXERS, VARIANTS, build, build_any, build_mixer, describe, describe_mixer


def cmd_backends(args) -> int:
    report = describe_backends()
    for key, value in report.items():
        print(f"{key:>26}: {value}")
    return 0


def cmd_variants(args) -> int:
    print("attention variants  (q, k, v -> attention over a KV cache)")
    for name in VARIANTS:
        meta = describe(name)
        trained = "needs-training" if meta["needs_training"] else "training-free"
        print(f"{name:<18} {meta['family']:<10} {trained:<15} {meta['citation'][0] if meta['citation'] else ''}")
    print()
    print("state mixers  (x -> y with a fixed-size recurrent state; see docs/variants/state_mixers.md)")
    for kind in MIXERS:
        meta = describe_mixer(kind)
        trained = "needs-training" if meta["needs_training"] else "training-free"
        print(f"{kind:<18} {'mixer':<10} {trained:<15} {meta['citation'][0]}")
    print()
    print("hybrid stack  HybridLM interleaves attention layers with the mixers above "
          "(see experiments/hybrid_recall.py)")
    return 0


def cmd_mixers(args) -> int:
    """State/flop accounting per mixer kind at a given shape."""
    from linattn.variants.recurrent import LinearAttentionConfig

    cfg = LinearAttentionConfig(d_model=args.d_model, n_heads=args.heads, d_state=args.d_state)
    print(f"{'kind':<8} {'decay':<34} {'state floats':>13} {'flops/token':>12}")
    for kind in MIXERS:
        layer = build_mixer(kind, d_model=args.d_model, n_heads=args.heads, d_state=args.d_state)
        meta = describe_mixer(kind)
        floats = layer.cfg.n_heads * layer.cfg.d_state * layer.cfg.d_value
        print(f"{kind:<8} {meta['decay']:<34} {floats:>13} {layer.decode_flops_per_token():>12}")
    print()
    print("state size and per-token cost are identical across kinds; only the update")
    print("rule (and therefore quality) differs. See reports/state_capacity.md for what")
    print("a state of a given size can actually remember.")
    print()
    for kind in MIXERS:
        meta = describe_mixer(kind)
        print(f"{kind}: {meta['summary']}")
        print(f"    recurrence: {meta['recurrence']}")
        print(f"    gates: {meta['gates']}")
    return 0


def cmd_cost(args) -> int:
    names = args.variants or list(VARIANTS)
    variants = [build_any(name, cfg_kwargs(args)) for name in names]
    print(efficiency_table(variants, args.seq, batch=args.batch, markdown=False))
    print()
    print("state mixers  (fixed-size state; x -> y, no growing KV cache)")
    mixer_rows = []
    for kind in MIXERS:
        layer = build_mixer(kind, d_model=args.d_model, n_heads=args.mixer_heads, d_state=args.d_state)
        floats = layer.cfg.n_heads * layer.cfg.d_state * layer.cfg.d_value
        state_mib = layer.state_bytes(args.batch) / 2**20
        flops_m = layer.decode_flops_per_token(args.batch) / 1e6
        mixer_rows.append([kind, f"{floats:,}", f"{state_mib:.2f}", f"{flops_m:.1f}"])
    print(text_table(
        ["kind", "state floats", "state (MiB, fp32)", "decode FLOPs/token (M)"],
        mixer_rows,
        align="lrrr",
    ))
    return 0


def cfg_kwargs(args):
    from linattn.core import AttnConfig

    return AttnConfig(
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        head_dim=args.dim,
        block_size=args.block_size,
        window_size=args.window,
        cache_budget=args.budget,
        topk_blocks=args.topk_blocks,
        topk_tokens=args.topk_tokens,
    )


def cmd_select(args) -> int:
    cfg = cfg_kwargs(args)
    if args.structured:
        q, k, v = structured_inputs(1, cfg.n_heads, args.seq, cfg.head_dim, kv_heads=cfg.n_kv_heads, seed=args.seed)
    else:
        q, k, v = make_inputs(1, cfg.n_heads, args.seq, cfg.head_dim, kv_heads=cfg.n_kv_heads, seed=args.seed)
    variant = build(args.variant, cfg)
    dummy = torch.zeros_like(v)
    rec = run_variant(variant, q, k, v, dtype=torch.float64)
    mask, _ = _selection_mask(variant, q, k, v)
    if mask is not None:
        block = max(1, cfg.block_size)
        mass = attention_mass(q.double(), k.double(), v.double(), block=block)
        n_qb = mask.shape[-2] // block
        n_kb = mask.shape[-1] // block
        reduced = mask[..., : n_qb * block, : n_kb * block]
        reduced = reduced.view(*mask.shape[:-2], n_qb, block, n_kb, block).any(-3).any(-1)
        rec.mass_recall = float(mass_recall(mass[..., :n_qb, :n_kb], reduced).mean())
        rec.extra["mass_recall"] = rec.mass_recall
    print(json.dumps({"variant": rec.variant, "family": rec.family, **rec.as_dict()}, indent=2, default=str))
    return 0


def _selection_mask(variant, q, k, v):
    """Best-effort extraction of a variant's read mask (for recall reporting)."""
    from linattn.variants.selection import BlockSelectionAttention

    if not isinstance(variant, BlockSelectionAttention):
        return None, None
    state = variant.prepare_state(q.shape[0], q.device, q.dtype)
    state = variant.push_kv(state, k, v)
    with torch.no_grad():
        mask = variant.select(q, state, 0)
    return mask, state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="linattn", description="sparse / compressed-state attention toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shape_args(p):
        p.add_argument("--seq", type=int, default=1024)
        p.add_argument("--batch", type=int, default=1)
        p.add_argument("--heads", type=int, default=8)
        p.add_argument("--kv-heads", type=int, default=8, dest="kv_heads")
        p.add_argument("--dim", type=int, default=64)
        p.add_argument("--block-size", type=int, default=64, dest="block_size")
        p.add_argument("--window", type=int, default=256)
        p.add_argument("--budget", type=int, default=512)
        p.add_argument("--topk-blocks", type=int, default=4, dest="topk_blocks")
        p.add_argument("--topk-tokens", type=int, default=256, dest="topk_tokens")

    sub.add_parser("backends", help="report available backends").set_defaults(func=cmd_backends)
    sub.add_parser("variants", help="list attention variants and state mixers").set_defaults(func=cmd_variants)

    mixers = sub.add_parser("mixers", help="state-mixer accounting and recurrence forms")
    mixers.add_argument("--d-model", type=int, default=2048, dest="d_model")
    mixers.add_argument("--heads", type=int, default=16)
    mixers.add_argument("--d-state", type=int, default=128, dest="d_state")
    mixers.set_defaults(func=cmd_mixers)

    cost = sub.add_parser("cost", help="analytic cost model")
    add_shape_args(cost)
    cost.add_argument("--d-model", type=int, default=2048, dest="d_model")
    cost.add_argument("--mixer-heads", type=int, default=16, dest="mixer_heads")
    cost.add_argument("--d-state", type=int, default=128, dest="d_state")
    cost.add_argument("--variants", nargs="*", default=None)
    cost.set_defaults(func=cmd_cost)

    sel = sub.add_parser("select", help="run one variant against dense attention")
    add_shape_args(sel)
    sel.add_argument("--variant", required=True)
    sel.add_argument("--structured", action="store_true", help="plant retrieval structure in the inputs")
    sel.add_argument("--seed", type=int, default=0)
    sel.set_defaults(func=cmd_select)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
