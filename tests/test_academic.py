"""Academic variants: invariant checks, degenerate exactness, and cost scaling.

Approximate methods cannot be validated by output parity. What *can* be checked:

* degenerate configurations that reduce to exact attention (identity projection,
  one hash bucket, full window, all blocks, one head) -- if those fail, the
  implementation is wrong;
* budgets and causality (measured reads never exceed the advertised budget);
* asymptotic behaviour (constant reads for low-rank/feature methods, linear growth
  for local ones, quadratic for dense).
"""

from __future__ import annotations

import pytest
import torch

from linattn.core import AttnConfig, dense_attention
from linattn.metrics import rel_error
from linattn.registry import build


def _cfg(**kw) -> AttnConfig:
    base = dict(n_heads=4, n_kv_heads=2, head_dim=32, block_size=16, window_size=64,
                topk_blocks=4, cache_budget=128, sink_size=2)
    base.update(kw)
    return AttnConfig(**base)


def _inputs(cfg, t=256, batch=1, seed=0, fill=None):
    torch.manual_seed(seed)
    q = torch.randn(batch, cfg.n_heads, t, cfg.head_dim, dtype=torch.float64)
    k = torch.randn(batch, cfg.n_kv_heads, t, cfg.head_dim, dtype=torch.float64)
    v = torch.randn(batch, cfg.n_kv_heads, t, cfg.head_dim, dtype=torch.float64)
    return q, k, v


# --------------------------------------------------------------------------------------
# degenerate configurations must equal exact attention
# --------------------------------------------------------------------------------------
def test_linformer_identity_projection_is_exact():
    cfg = _cfg()
    q, k, v = _inputs(cfg)
    attn = build("linformer", cfg.with_(extras={"linformer_k": 256, "identity": True}))
    out, _ = attn.forward(q, k, v, chunk=256)
    assert rel_error(out, dense_attention(q, k, v, causal=False)) < 1e-10


def test_bigbird_with_full_window_is_exact():
    cfg = _cfg()
    q, k, v = _inputs(cfg)
    attn = build("bigbird", cfg.with_(extras={"window": 256, "n_random": 0, "n_global": 0}))
    out, _ = attn.forward(q, k, v, chunk=256)
    assert rel_error(out, dense_attention(q, k, v)) < 1e-10


def test_reformer_with_one_bucket_is_exact():
    cfg = _cfg()
    q, k, v = _inputs(cfg)
    attn = build("reformer_lsh", cfg.with_(extras={"buckets": 1}))
    out, _ = attn.forward(q, k, v, chunk=256)
    assert rel_error(out, dense_attention(q, k, v)) < 1e-10


def test_free_solo_with_full_window_and_no_cells_is_exact():
    cfg = _cfg()
    q, k, v = _inputs(cfg)
    attn = build("free_solo", cfg.with_(extras={"window": 256, "topk_cells": 0}))
    out, _ = attn.forward(q, k, v, chunk=256)
    assert rel_error(out, dense_attention(q, k, v)) < 1e-10


def test_namoh_with_one_head_is_exact():
    cfg = _cfg(n_heads=1, n_kv_heads=1)
    q, k, v = _inputs(cfg)
    attn = build("namoh", cfg.with_(extras={"window": 4}))
    out, _ = attn.forward(q, k, v, chunk=256)
    assert rel_error(out, dense_attention(q, k, v)) < 1e-10


def test_mass_oracle_with_all_blocks_is_exact():
    cfg = _cfg()
    q, k, v = _inputs(cfg)
    attn = build("mass_oracle", cfg.with_(extras={"topk_blocks": 256 // 16, "sinks": 0}))
    out, _ = attn.forward(q, k, v, chunk=256)
    assert rel_error(out, dense_attention(q, k, v)) < 1e-10


def test_performer_improves_with_more_features():
    cfg = _cfg(n_heads=2, n_kv_heads=2, head_dim=16)
    q, k, v = _inputs(cfg, t=64)
    ref = dense_attention(q, k, v)
    small = build("performer", cfg.with_(extras={"features": 16}))
    large = build("performer", cfg.with_(extras={"features": 1024}))
    err_small = rel_error(small.forward(q, k, v)[0], ref)
    err_large = rel_error(large.forward(q, k, v)[0], ref)
    assert err_large < err_small
    assert err_large < 0.6


# --------------------------------------------------------------------------------------
# budgets and cost scaling
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["linformer", "performer", "mass_oracle", "free_solo", "namoh", "hla_mixing"])
def test_reads_stay_within_the_advertised_budget(name):
    cfg = _cfg()
    q, k, v = _inputs(cfg)
    attn = build(name, cfg)
    _, _, stats = attn.forward(q, k, v, chunk=256, collect_stats=True)
    reads = stats[-1]["reads_per_query"]
    assert reads <= attn.read_budget(256) + 1
    if name != "hla_mixing":        # state-based: reads include the state
        assert reads <= 256


def test_low_rank_and_feature_methods_do_not_grow_with_context():
    cfg = _cfg()
    for name, extra in (("linformer", {"linformer_k": 32}), ("performer", {"features": 32})):
        attn = build(name, cfg.with_(extras=extra))
        assert attn.read_budget(2048) == attn.read_budget(16384)


def test_long_context_costs_grow_linearly_for_local_variants():
    cfg = _cfg(window_size=128)
    local = build("sliding_window", cfg)
    attn = build("free_solo", cfg.with_(extras={"window": 128, "topk_cells": 2}))
    assert local.read_budget(8192) < 8192 / 8
    assert attn.read_budget(8192) < 8192 / 2


def test_hla_mixes_two_branches_with_a_data_dependent_gate():
    cfg = _cfg()
    q, k, v = _inputs(cfg)
    attn = build("hla_mixing", cfg)
    out, _, stats = attn.forward(q, k, v, chunk=256, collect_stats=True)
    assert torch.isfinite(out).all()
    gate = stats[-1]["gate"]
    assert 0.0 < gate < 1.0


# --------------------------------------------------------------------------------------
# the academic eviction lineage: budget honesty and protected windows
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["h2o", "snapkv", "tova", "random_eviction"])
def test_scored_eviction_keeps_the_budget(name):
    cfg = _cfg(cache_budget=64, window_size=16, obs_window=8)
    q, k, v = _inputs(cfg)
    attn = build(name, cfg)
    _, state = attn.forward(q, k, v, chunk=64)
    assert state["k"].shape[-2] <= cfg.cache_budget
    assert state["k"].shape[-2] == state["v"].shape[-2]


@pytest.mark.parametrize("name", ["h2o", "snapkv", "tova"])
def test_scored_eviction_differs_from_each_other(name):
    """H2O (accumulated), TOVA (last query) and SnapKV (observation window) must not
    collapse to the same selection -- otherwise the score is not doing anything."""
    cfg = _cfg(cache_budget=48, window_size=8, obs_window=8)
    q, k, v = _inputs(cfg, t=192)
    out = {}
    for other in ("h2o", "snapkv", "tova"):
        attn = build(other, cfg)
        with torch.no_grad():
            out[other] = attn.forward(q, k, v, chunk=48)[0]
    assert not torch.allclose(out["h2o"], out["tova"], atol=1e-6)
    assert not torch.allclose(out["snapkv"], out["tova"], atol=1e-6)
