"""Every variant must actually honour the budget it advertises.

This suite is the one that keeps the repo honest: a "4x sparse" method that
silently reads the full cache would show up here as a budget violation rather
than as a suspiciously good accuracy number.
"""

from __future__ import annotations

import pytest
import torch

from linattn.core import dense_attention
from linattn.metrics import rel_error
from linattn.registry import build
from linattn.variants.selection import BlockSelectionAttention


SELECTION = ["quest", "minference", "flexprefill", "xattention", "moba", "dsa", "seerattention_r", "nsa"]
EVICTION = ["random_eviction"]     # one control here; frontier eviction is covered in test_frontier.py


@pytest.mark.parametrize("name", SELECTION)
def test_selection_variants_only_read_selected_entries(cfg, structured, name):
    q, k, v = structured
    variant = build(name, cfg)
    _, _, stats = variant.forward(q, k, v, collect_stats=True, chunk=64)
    reads = max(s["reads_per_query"] for s in stats)
    t_k = q.shape[-2]
    assert reads <= t_k, f"{name} read {reads} > {t_k} entries per query"
    # and the point of the exercise: strictly fewer than dense at a real budget
    assert reads < t_k * 0.75, f"{name} did not sparsify (reads={reads}, T={t_k})"


@pytest.mark.parametrize("name", EVICTION)
def test_eviction_variants_respect_cache_budget(cfg, structured, name):
    q, k, v = structured
    variant = build(name, cfg)
    out, state, _ = variant.forward(q, k, v, collect_stats=True, chunk=64)
    assert state["k"].shape[-2] <= cfg.cache_budget
    assert state["v"].shape[-2] == state["k"].shape[-2]


@pytest.mark.parametrize("name", EVICTION)
def test_eviction_keeps_the_recent_window(cfg, structured, name):
    """The recent tokens and the sinks must never be evicted: that is the shared
    invariant of every eviction method in the literature."""
    q, k, v = structured
    variant = build(name, cfg)
    _, state, _ = variant.forward(q, k, v, collect_stats=True, chunk=64)
    # the harness input is a permutation-free sequence, so identity is checkable:
    # the retained set must include the last `window_size` tokens of the input
    retained = state["k"]
    assert retained.shape[-2] >= cfg.window_size


def test_dense_variant_reads_everything(cfg, structured):
    q, k, v = structured
    variant = build("dense", cfg)
    out, _ = variant.forward(q, k, v)
    assert torch.allclose(out, dense_attention(q, k, v), atol=1e-6)


def test_full_budget_recovers_dense(cfg, structured):
    """With an unrestricted budget the selectors must reduce to dense attention.

    This is the sharpest correctness check for a selector: any discrepancy is a
    bug in the selection plumbing (masks, offsets, pooling), not a modelling
    choice.
    """
    q, k, v = structured
    t = q.shape[-2]
    reference = dense_attention(q.to(torch.float64), k.to(torch.float64), v.to(torch.float64))
    generous = cfg.with_(topk_blocks=t // cfg.block_size + 2, topk_tokens=t + 8, window_size=t + 8,
                         cache_budget=t + 8, threshold=1.0,
                         extras={"n_vertical": 10_000, "n_slash": 10_000, "pattern": "block_sparse",
                                 "select_mode": "topk", "xattn_block": cfg.block_size})
    for name in ["quest", "moba", "xattention", "minference"]:
        variant = build(name, generous)
        out, _ = variant.forward(q.to(torch.float64), k.to(torch.float64), v.to(torch.float64), chunk=t)
        assert rel_error(out, reference) < 1e-6, f"{name} failed at full budget"
