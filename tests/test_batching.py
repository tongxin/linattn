"""Batch-request support: the same call at B>1 must equal B=1 per item.

The variants are written with explicit batch dimensions, but until this file existed
the suite was almost entirely ``batch=1``.  Two real bugs were hiding there: MInference's
vertical-slash mask reshaped ``(B,H,k)`` to ``(B,1,H*k)`` (a crash at B>1, silently
wrong at B=1), and nothing checked that incremental decode or the state mixers kept
items independent.  These tests are the guard.

Scope note: this is *equal-length* batching.  There is no ``lengths``/padding support,
so ragged or continuously-batched requests need one call per length (or an explicit
mask) -- see ``docs/reproducing.md#batch-and-request-semantics``.
"""

from __future__ import annotations

import pytest
import torch

from linattn.core import AttnConfig
from linattn.harness import compare, structured_inputs
from linattn.metrics import rel_error
from linattn.registry import VARIANTS, build, build_mixer
from linattn.variants.recurrent import STATE_KINDS

DETERMINISTIC = [name for name in VARIANTS if name != "random_eviction"]


@pytest.fixture
def cfg() -> AttnConfig:
    return AttnConfig(n_heads=4, n_kv_heads=2, head_dim=64, block_size=16, window_size=64,
                      cache_budget=128, topk_blocks=4, topk_tokens=128, obs_window=16, sink_size=2)


@pytest.mark.parametrize("name", DETERMINISTIC)
def test_variant_is_batch_consistent(name, cfg):
    """One batched call == one call per item, for the same variant instance."""
    batch, seq = 3, 192
    q, k, v = structured_inputs(batch, cfg.n_heads, seq, cfg.head_dim, kv_heads=cfg.n_kv_heads, seed=0)
    variant = build(name, cfg)
    with torch.no_grad():
        batched, _ = variant.forward(q, k, v, chunk=64)
        for i in range(batch):
            single, _ = variant.forward(q[i : i + 1], k[i : i + 1], v[i : i + 1], chunk=64)
            assert rel_error(batched[i : i + 1], single) < 1e-9, f"{name}: item {i} depends on batch composition"


@pytest.mark.parametrize("name", ["dense", "sliding_window", "quest", "moba", "nsa"])
def test_incremental_decode_is_batch_consistent(name, cfg):
    """The cache path (one token per call) must also keep items independent."""
    batch, seq = 3, 96
    q, k, v = structured_inputs(batch, cfg.n_heads, seq, cfg.head_dim, kv_heads=cfg.n_kv_heads, seed=1)
    variant = build(name, cfg)
    with torch.no_grad():
        batched, _ = variant.forward(q, k, v, chunk=1)
        for i in range(batch):
            single, _ = variant.forward(q[i : i + 1], k[i : i + 1], v[i : i + 1], chunk=1)
            assert rel_error(batched[i : i + 1], single) < 1e-9


def test_random_eviction_is_deterministic_for_a_fixed_seed(cfg):
    """The random baseline cannot be batch-invariant (one RNG stream per call), but it
    must be reproducible for a fixed seed and shape -- otherwise its numbers are noise.
    """
    q, k, v = structured_inputs(3, cfg.n_heads, 192, cfg.head_dim, kv_heads=cfg.n_kv_heads, seed=3)
    first = build("random_eviction", cfg).forward(q, k, v, chunk=64)[0]
    second = build("random_eviction", cfg).forward(q, k, v, chunk=64)[0]
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


@pytest.mark.parametrize("kind", list(STATE_KINDS))
def test_state_mixers_are_batch_consistent(kind):
    """Mixers own their projections, so check them in float64 for exactness."""
    layer = build_mixer(kind, d_model=64, n_heads=4, d_state=16, chunk=16).double()
    x = torch.randn(3, 48, 64, dtype=torch.float64)
    with torch.no_grad():
        batched, state_b = layer.forward_chunked(x)
        for i in range(3):
            single, state_s = layer.forward_chunked(x[i : i + 1])
            assert torch.allclose(batched[i : i + 1], single, atol=1e-10)
            assert torch.allclose(state_b[i : i + 1], state_s, atol=1e-10)


def test_harness_supports_batching(cfg):
    """The benchmark harness must accept batch > 1 and scale state with it."""
    records = compare([build("dense", cfg), build("quest", cfg)], seq=256, batch=2,
                      heads=cfg.n_heads, kv_heads=cfg.n_kv_heads, dim=cfg.head_dim, chunk=64)
    assert all(r["rel_err"] == r["rel_err"] for r in records)          # no NaN
    by_name = {r["variant"]: r for r in records}
    # dense state is T * H_kv * D * 2 tensors * 2 bytes, times batch
    expected = 256 * cfg.n_kv_heads * cfg.head_dim * 2 * 2 * 2
    assert by_name["dense"]["state_bytes"] == pytest.approx(expected, rel=1e-6)


def test_eviction_keeps_per_item_budgets(cfg):
    """A batched decode loop must not let one item's cache evict another's."""
    batch, seq = 2, 96
    q, k, v = structured_inputs(batch, cfg.n_heads, seq, cfg.head_dim, kv_heads=cfg.n_kv_heads, seed=4)
    variant = build("random_eviction", cfg)
    with torch.no_grad():
        _, state = variant.forward(q, k, v, chunk=1)
    assert state["k"].shape[0] == batch
    assert state["k"].shape[-2] <= cfg.cache_budget
    # the two items have different content, so their retained key sets must differ
    assert not torch.equal(state["k"][0], state["k"][1])
