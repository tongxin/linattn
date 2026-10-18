"""Validation for the 2025--2026 frontier variants.

The selection methods are checked where a reference implementation can be
unambiguous: at an unrestricted budget they must reduce to dense attention.  Their
approximations are then tested for budget honesty rather than output parity.
"""

from __future__ import annotations

import pytest
import torch

from linattn.core import AttnConfig, dense_attention
from linattn.metrics import rel_error
from linattn.registry import build, build_mixer


def _cfg(**kw) -> AttnConfig:
    base = dict(n_heads=4, n_kv_heads=2, head_dim=16, block_size=16, window_size=32,
                topk_blocks=2, topk_tokens=32, cache_budget=48, obs_window=8, sink_size=2)
    base.update(kw)
    return AttnConfig(**base)


def _inputs(cfg: AttnConfig, t: int = 128, batch: int = 1):
    torch.manual_seed(17)
    q = torch.randn(batch, cfg.n_heads, t, cfg.head_dim, dtype=torch.float64)
    k = torch.randn(batch, cfg.n_kv_heads, t, cfg.head_dim, dtype=torch.float64)
    v = torch.randn(batch, cfg.n_kv_heads, t, cfg.head_dim, dtype=torch.float64)
    return q, k, v


@pytest.mark.parametrize("name", ["saga", "pisa", "jssa", "masa", "oval"])
def test_frontier_selection_reduces_to_dense_at_full_budget(name):
    cfg = _cfg(window_size=128, topk_blocks=8, topk_tokens=128, sink_size=0)
    q, k, v = _inputs(cfg)
    out, _ = build(name, cfg).forward(q, k, v, chunk=128)
    assert rel_error(out, dense_attention(q, k, v)) < 1e-10


def test_feature_sparsity_is_dense_when_every_feature_is_kept():
    cfg = _cfg(extras={"feature_keep": 16})
    q, k, v = _inputs(cfg)
    out, _ = build("sfa", cfg).forward(q, k, v, chunk=128)
    assert rel_error(out, dense_attention(q, k, v)) < 1e-10


@pytest.mark.parametrize("name", ["saga", "pisa", "jssa", "masa", "oval"])
def test_frontier_selection_honours_its_read_budget(name):
    cfg = _cfg()
    q, k, v = _inputs(cfg, t=128)
    attn = build(name, cfg)
    _, _, stats = attn.forward(q, k, v, chunk=64, collect_stats=True)
    reads = max(s["reads_per_query"] for s in stats)
    assert reads <= attn.read_budget(128) + 1
    assert reads < 128 * 0.75


@pytest.mark.parametrize("name", ["dropkv", "restkv", "beaconkv"])
def test_output_aware_eviction_keeps_the_cache_budget(name):
    cfg = _cfg(cache_budget=32, window_size=8, obs_window=8)
    q, k, v = _inputs(cfg, t=160)
    _, state = build(name, cfg).forward(q, k, v, chunk=32)
    assert state["k"].shape[-2] <= cfg.cache_budget
    assert state["k"].shape[-2] == state["v"].shape[-2]


def test_dropkv_is_not_attention_mass_eviction():
    cfg = _cfg(cache_budget=32, window_size=8, obs_window=8)
    q, k, v = _inputs(cfg, t=160)
    with torch.no_grad():
        drop = build("dropkv", cfg).forward(q, k, v, chunk=32)[0]
        h2o = build("h2o", cfg).forward(q, k, v, chunk=32)[0]
    assert not torch.allclose(drop, h2o, atol=1e-6)


@pytest.mark.parametrize("kind", ["qdelta", "kdn", "swila"])
def test_frontier_mixers_have_fixed_state_and_finite_outputs(kind):
    layer = build_mixer(kind, d_model=32, n_heads=2, d_state=8, chunk=8).double()
    x1 = torch.randn(2, 12, 32, dtype=torch.float64)
    x2 = torch.randn(2, 48, 32, dtype=torch.float64)
    with torch.no_grad():
        y1, s1 = layer.forward_recurrent(x1)
        y2, s2 = layer.forward_recurrent(x2)
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
    assert s1.shape == s2.shape and s1.dim() == 4


def test_kdn_tracks_positive_uncertainty():
    layer = build_mixer("kdn", d_model=32, n_heads=2, d_state=8).double()
    x = torch.randn(2, 16, 32, dtype=torch.float64)
    with torch.no_grad():
        _, state = layer.forward_recurrent(x)
    assert (state[..., -1] > 0).all()


def test_swila_state_grows_only_with_experts_not_context():
    layer = build_mixer("swila", d_model=32, n_heads=2, d_state=8, n_experts=4)
    assert layer.state_bytes(1) == 4 * 2 * 4 * 8 * layer.cfg.d_value
