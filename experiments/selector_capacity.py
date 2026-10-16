"""Can a cheap learned selector reproduce dense attention's ranking?

The learned-selector family (DSA's lightning indexer, SeerAttention's block gate,
NSA's branch gates) is only viable if the parameterisation can rank keys/blocks
the way softmax attention does.  This experiment measures that directly, on
inputs with planted retrieval structure:

* fit the DSA indexer (low-dim, ReLU, sum over heads) and the SeerAttention gate
  (pooled q/k -> projection -> sigmoid) against the dense attention distribution;
* report the cross-entropy before and after fitting, plus the attention-mass
  recall of the selection that the fitted selector produces;
* fit NSA's branch gates against the dense output and report the output error
  before and after.

Run::

    python experiments/selector_capacity.py

Results land in ``reports/selector_capacity.{json,md}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linattn.core import AttnConfig, dense_attention  # noqa: E402
from linattn.harness import structured_inputs  # noqa: E402
from linattn.metrics import attention_mass, mass_recall, rel_error  # noqa: E402
from linattn.reporting import save_report  # noqa: E402
from linattn.variants.native_sparse import NSAAttention  # noqa: E402
from linattn.variants.selection import DSAAttention, SeerAttentionR  # noqa: E402


def block_recall(mask: torch.Tensor, mass: torch.Tensor, block: int) -> float:
    n_qb, n_kb = mass.shape[-2:]
    reduced = mask[..., : n_qb * block, : n_kb * block]
    reduced = reduced.view(*mask.shape[:-2], n_qb, block, n_kb, block).any(-3).any(-1)
    return float(mass_recall(mass, reduced).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2, dest="kv_heads")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cfg = AttnConfig(
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        head_dim=args.dim,
        block_size=64,
        window_size=128,
        topk_tokens=128,
        topk_blocks=4,
        compress_ratio=16,
    )
    q, k, v = structured_inputs(1, cfg.n_heads, args.seq, cfg.head_dim, kv_heads=cfg.n_kv_heads, seed=args.seed)
    q, k, v = q.double(), k.double(), v.double()
    mass = attention_mass(q, k, v, block=cfg.block_size)
    reference = dense_attention(q, k, v)
    records = []

    # --- DSA lightning indexer -------------------------------------------------
    dsa = DSAAttention(cfg.with_(extras={"indexer_heads": 8, "indexer_dim": 16, "compress_ratio": 1}))
    state = dsa.prepare_state(1, q.device, q.dtype)
    state = dsa.push_kv(state, k, v)
    with torch.no_grad():
        before_mask = dsa.select(q, state, 0)
        recall_before = block_recall(before_mask, mass, cfg.block_size)
        scores = dsa.indexer_scores(q, k)

        target = dense_attention(q, k, v, causal=True, return_weights=True)[1]
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-9)
        ce_before = float(-(torch.log_softmax(scores, -1) * target).nan_to_num().sum(-1).mean())
    ce_after = dsa.fit(q, k, v, steps=args.steps, lr=0.05)
    with torch.no_grad():
        after_mask = dsa.select(q, state, 0)
        recall_after = block_recall(after_mask, mass, cfg.block_size)
    records.append({
        "selector": "dsa_indexer",
        "cross_entropy_before": ce_before,
        "cross_entropy_after": ce_after,
        "mass_recall_before": recall_before,
        "mass_recall_after": recall_after,
        "budget_fraction": float(after_mask.float().mean()),
    })
    print(f"dsa_indexer         CE {ce_before:.3f} -> {ce_after:.3f}   recall {recall_before:.3f} -> {recall_after:.3f}")

    # --- SeerAttention-R gate --------------------------------------------------
    seer = SeerAttentionR(cfg.with_(extras={"pool_dim": 16, "seer_topn": cfg.topk_blocks}))
    state = seer.prepare_state(1, q.device, q.dtype)
    state = seer.push_kv(state, k, v)
    with torch.no_grad():
        before_mask = seer.select(q, state, 0)
        recall_before = block_recall(before_mask, mass, cfg.block_size)
        gates = seer.block_gate(q, k)
        target_blocks = mass.mean(1, keepdim=True).expand_as(gates)
        ce_before = float(-(torch.log_softmax(gates, -1) * target_blocks).nan_to_num().sum(-1).mean())
    ce_after = seer.fit_gate(q, k, v, steps=args.steps, lr=0.05)
    with torch.no_grad():
        after_mask = seer.select(q, state, 0)
        recall_after = block_recall(after_mask, mass, cfg.block_size)
    records.append({
        "selector": "seerattention_gate",
        "cross_entropy_before": ce_before,
        "cross_entropy_after": ce_after,
        "mass_recall_before": recall_before,
        "mass_recall_after": recall_after,
        "budget_fraction": float(after_mask.float().mean()),
    })
    print(f"seerattention_gate  CE {ce_before:.3f} -> {ce_after:.3f}   recall {recall_before:.3f} -> {recall_after:.3f}")

    # --- NSA branch gates ------------------------------------------------------
    nsa = NSAAttention(cfg)
    err_before = rel_error(nsa.forward(q, k, v, chunk=args.seq)[0], reference)
    loss = nsa.fit_gates(q, k, v, steps=args.steps, lr=0.05)
    err_after = rel_error(nsa.forward(q, k, v, chunk=args.seq)[0], reference)
    records.append({
        "selector": "nsa_branch_gates",
        "cross_entropy_before": float("nan"),
        "cross_entropy_after": float("nan"),
        "mass_recall_before": 1.0 - err_before,
        "mass_recall_after": 1.0 - err_after,
        "budget_fraction": float("nan"),
    })
    print(f"nsa_branch_gates    output rel_err {err_before:.4f} -> {err_after:.4f} (fit loss {loss:.4f})")

    path = save_report(
        "selector_capacity",
        records,
        title="Learned-selector capacity: before and after fitting",
        columns=["selector", "cross_entropy_before", "cross_entropy_after", "mass_recall_before",
                 "mass_recall_after", "budget_fraction"],
        notes=(
            "Cross-entropy is between the selector's induced distribution over keys/blocks "
            "and the dense attention mass, so `before` is a randomly initialised selector "
            "and `after` is one fitted on the same inputs (`steps` gradient steps). "
            "The gap between the two is the price of the training stage that DSA, "
            "SeerAttention-R and NSA require. For NSA the metric is 1 - relative output "
            "error against dense attention, because its gates act on outputs rather than "
            "on a distribution."
        ),
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
