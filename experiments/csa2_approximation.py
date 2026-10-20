"""CSA2 versus dense softmax: separate pooling, selection and quantisation errors.

The analytic cost table says CSA2 is cheap. It does not say it is close to dense
softmax. This experiment changes one mechanism at a time on the same inputs:

* pooling only: ``compress_ratio > 1``, no quantisation, all groups selected;
* selection only: ``compress_ratio = 1``, no quantisation, vary top-k groups;
* quantisation only: ``compress_ratio = 1``, all groups selected, vary bits;
* combined: the default-like CSA2 inference configuration.

The inputs are random and planted-retrieval tensors, not a pretrained model. The
result is a controlled error decomposition and a sanity check, not a replacement
for RULER/LongBench or a coding-agent evaluation.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from linattn.core import AttnConfig, block_reduce, dense_attention  # noqa: E402
from linattn.harness import make_inputs, structured_inputs  # noqa: E402
from linattn.metrics import rel_error  # noqa: E402
from linattn.registry import build  # noqa: E402
from linattn.reporting import save_report  # noqa: E402


def _cfg(args: argparse.Namespace, window: int) -> AttnConfig:
    return AttnConfig(
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        head_dim=args.dim,
        block_size=16,
        window_size=window,
        topk_blocks=8,
        sink_size=0,
    )


def _selected_mass(
    layer,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    compress: int,
    window: int,
    topk_groups: int,
) -> float:
    """Fraction of dense attention mass covered by CSA2's selected support."""
    _, weights = dense_attention(q, k, v, causal=True, return_weights=True)
    kc, _ = layer.compressed_kv(k, v)
    t_q, t_k = q.shape[-2], k.shape[-2]
    n_groups = kc.shape[-2]
    qpos = torch.arange(t_q, device=q.device)
    group_end = (torch.arange(n_groups, device=q.device) + 1) * compress - 1
    w = max(1, min(window, t_k))
    eligible = (group_end.unsqueeze(0) <= qpos.unsqueeze(1)) & (
        group_end.unsqueeze(0) < (qpos.unsqueeze(1) - w + 1)
    )
    scores = layer.indexer_scores(q, kc).masked_fill(~eligible.unsqueeze(0).unsqueeze(0), float("-inf"))
    keep = max(1, min(topk_groups, n_groups))
    sel = torch.topk(scores, keep, dim=-1).indices
    group_mask = torch.zeros(q.shape[0], q.shape[1], t_q, n_groups,
                             dtype=torch.bool, device=q.device)
    group_mask.scatter_(-1, sel, True)
    group_mask &= eligible.unsqueeze(0).unsqueeze(0)
    token_mask = group_mask.repeat_interleave(compress, dim=-1)[..., :t_k]
    kj = torch.arange(t_k, device=q.device)
    window_mask = (kj.unsqueeze(0) <= qpos.unsqueeze(1)) & (
        (qpos.unsqueeze(1) - kj.unsqueeze(0)) < w
    )
    support = token_mask | window_mask
    return float((weights * support).sum() / weights.sum().clamp_min(1e-12))


def _run(
    *,
    study: str,
    input_kind: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    reference: torch.Tensor,
    dense_flops: float,
    args: argparse.Namespace,
    compress: int,
    bits: int,
    window: int,
    topk_groups: int,
) -> Dict:
    cfg = _cfg(args, window)
    layer = build("csa2", cfg.with_(extras={
        "compress_ratio": compress,
        "bits": bits,
        "quant_group": args.quant_group,
        "window": window,
        "topk_groups": topk_groups,
    }))
    out, _, stats = layer.forward(q, k, v, chunk=args.seq, collect_stats=True)
    flat_ref = reference.reshape(reference.shape[0], -1)
    flat_out = out.reshape(out.shape[0], -1)
    cosine = float(torch.nn.functional.cosine_similarity(flat_out, flat_ref, dim=-1).mean())
    return {
        "study": study,
        "input": input_kind,
        "compress": compress,
        "bits": bits,
        "window": window,
        "topk_groups": topk_groups,
        "rel_err": rel_error(out, reference),
        "max_abs_err": float((out - reference).abs().max()),
        "cosine": cosine,
        "mass_capture": _selected_mass(layer, q, k, v, compress, window, topk_groups),
        "bytes_per_token": layer.memory_bytes(args.seq, args.batch) / max(1, args.seq * args.batch),
        "flops_g": layer.flops(args.seq, args.batch) / 1e9,
        "dense_flops_over_csa2": dense_flops / max(1e-12, layer.flops(args.seq, args.batch)),
        "reads_per_query": float(stats[-1]["reads_per_query"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2, dest="kv_heads")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--quant-group", type=int, default=32, dest="quant_group")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    args.seq = min(args.seq, 1024)

    rows: list[Dict] = []
    for input_kind in ("random", "structured"):
        q, k, v = (
            structured_inputs(args.batch, args.heads, args.seq, args.dim,
                              kv_heads=args.kv_heads, seed=args.seed)
            if input_kind == "structured"
            else make_inputs(args.batch, args.heads, args.seq, args.dim,
                             kv_heads=args.kv_heads, seed=args.seed)
        )
        q, k, v = q.double(), k.double(), v.double()
        reference = dense_attention(q, k, v)
        dense_flops = build("dense", _cfg(args, args.seq)).flops(args.seq, args.batch)

        # Exactness controls: either a full raw window, or token-level groups with
        # every group selected.
        rows.append(_run(study="exactness", input_kind=input_kind, q=q, k=k, v=v,
                         reference=reference, dense_flops=dense_flops, args=args,
                         compress=args.seq, bits=16, window=args.seq, topk_groups=1))
        rows.append(_run(study="exactness", input_kind=input_kind, q=q, k=k, v=v,
                         reference=reference, dense_flops=dense_flops, args=args,
                         compress=1, bits=16, window=1, topk_groups=args.seq + 2))

        # Pool only the prefix: no top-k loss and no quantisation.
        for m in (1, 2, 4, 8, 16, 32, 64):
            rows.append(_run(study="pooling", input_kind=input_kind, q=q, k=k, v=v,
                             reference=reference, dense_flops=dense_flops, args=args,
                             compress=m, bits=16, window=1,
                             topk_groups=math.ceil(args.seq / m) + 2))

        # Token-level groups isolate top-k selection from pooling.
        for w in (16, 64, 128):
            for topk in (4, 8, 16, 32):
                rows.append(_run(study="selection", input_kind=input_kind, q=q, k=k, v=v,
                                 reference=reference, dense_flops=dense_flops, args=args,
                                 compress=1, bits=16, window=w, topk_groups=topk))

        # Quantise every previous token, keeping the current token raw.
        for bits in (16, 8, 4):
            rows.append(_run(study="quantization", input_kind=input_kind, q=q, k=k, v=v,
                             reference=reference, dense_flops=dense_flops, args=args,
                             compress=1, bits=bits, window=1, topk_groups=args.seq + 2))

        # Combined, including the default-like configuration.
        for m in (8, 16, 32, 64):
            for bits in (16, 4):
                rows.append(_run(study="combined", input_kind=input_kind, q=q, k=k, v=v,
                                 reference=reference, dense_flops=dense_flops, args=args,
                                 compress=m, bits=bits, window=128, topk_groups=8))

    path = save_report(
        "csa2_approximation",
        rows,
        title="CSA2 versus dense softmax: pooling, selection and precision error",
        columns=["study", "input", "compress", "bits", "window", "topk_groups",
                 "rel_err", "max_abs_err", "cosine", "mass_capture",
                 "bytes_per_token", "flops_g", "dense_flops_over_csa2", "reads_per_query"],
        notes=(
            "This is a controlled synthetic decomposition, not a pretrained-model "
            "benchmark. `mass_capture` uses the exact dense attention weights and "
            "counts the selected raw window plus selected compressed groups. "
            "`pooling` uses all groups and no quantisation; `selection` uses "
            "compress=1 so it has no pooling; `quantization` also uses compress=1 "
            "and selects every group. `combined` uses the default-like top-8 groups "
            "and a 128-token raw window. The random CSA2 indexer is untrained, so "
            "selection rows should be read as plumbing/controls, not as the "
            "authors' trained model."
        ),
    )
    print(f"wrote {path}")
    for r in rows:
        if r["study"] in ("exactness", "quantization", "combined"):
            print(f"{r['input']:<10} {r['study']:<11} m={r['compress']:<3} "
                  f"fp{r['bits']:<2} W={r['window']:<4} k={r['topk_groups']:<3} "
                  f"err={r['rel_err']:.4f} capture={r['mass_capture']:.3f} "
                  f"B/t={r['bytes_per_token']:.1f} "
                  f"dense/CSA2={r['dense_flops_over_csa2']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
