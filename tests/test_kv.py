"""KV layer: cache semantics, policy ordering, and cost-model sanity."""

from __future__ import annotations

import pytest

from linattn import cli
from linattn.kv import (
    PRESETS,
    CacheConfig,
    Hardware,
    ModelShape,
    Request,
    Workload,
    agentic_trace,
    breakdown,
    oracle_hits,
    simulate,
)


def _trace(hot: int = 8, churn: int = 40, n: int = 40):
    """Every request reuses a hot prefix, then appends one unique block."""
    hot_blocks = tuple(range(hot))
    return [Request(at=float(i), blocks=hot_blocks + (1000 + i,)) for i in range(n)]


def test_lookup_returns_leading_match_only():
    cache_cfg = CacheConfig(capacity_blocks=16)
    from linattn.kv import BlockCache

    cache = BlockCache(cache_cfg)
    cache.insert([1, 2, 3])
    assert cache.lookup([1, 2]) == 2            # prefix hit
    assert cache.lookup([1, 2, 3, 4]) == 3      # stops at the first miss
    assert cache.lookup([9]) == 0


def test_capacity_is_respected():
    cfg = CacheConfig(capacity_blocks=4)
    res = simulate(_trace(churn=40, n=40), cfg)
    assert res.peak_occupancy_tokens <= 4 * cfg.block_size
    assert res.evictions > 0


def test_lru_beats_random_when_capacity_binds():
    trace = _trace(hot=8, n=60)
    cfg = CacheConfig(capacity_blocks=12)
    lru = simulate(trace, CacheConfig(**{**cfg.__dict__, "policy": "lru"}))
    rnd = simulate(trace, CacheConfig(**{**cfg.__dict__, "policy": "random"}))
    assert lru.hit_rate > rnd.hit_rate


def test_ttl_expiry_forces_recompute():
    """A gap longer than the TTL must cost the prefix: the mechanism behind agentic
    recompute, and why the 17.5%-after-TTL measurement matters."""
    blocks = tuple(range(10))
    trace = [Request(at=0.0, blocks=blocks), Request(at=1000.0, blocks=blocks)]
    no_ttl = simulate(trace, CacheConfig(capacity_blocks=64, policy="lru"))
    with_ttl = simulate(trace, CacheConfig(capacity_blocks=64, policy="ttl_lru", ttl_s=300.0))
    assert no_ttl.hit_rate == 0.5
    assert with_ttl.hit_rate == 0.0
    assert with_ttl.expirations == 10


def test_oracle_is_an_upper_bound():
    trace = _trace(hot=8, n=50)
    hit, total = oracle_hits(trace, capacity_blocks=8)
    best = simulate(trace, CacheConfig(capacity_blocks=8, policy="lru"))
    assert hit / total >= best.hit_rate - 1e-9


def test_cost_falls_with_hit_rate_and_recompute_takes_over_at_zero():
    shape, hw = ModelShape(), Hardware()
    fmt = PRESETS["dense_fp16"]
    lo = breakdown(fmt, shape, hw, Workload(prompt_tokens=131072, hit_rate=0.0))
    hi = breakdown(fmt, shape, hw, Workload(prompt_tokens=131072, hit_rate=0.95))
    assert hi.total_ms < lo.total_ms
    assert lo.dominant == "recompute"       # with no reuse, recompute is the wall
    assert hi.recompute_ms < lo.recompute_ms


def test_compression_changes_bytes_not_flops():
    """The lesson the model is built to teach: a smaller KV format leaves attention
    FLOPs untouched, so it cannot fix a compute-bound context."""
    shape, hw = ModelShape(), Hardware()
    dense = breakdown(PRESETS["dense_fp16"], shape, hw, Workload(prompt_tokens=131072, hit_rate=0.9))
    mla = breakdown(PRESETS["mla"], shape, hw, Workload(prompt_tokens=131072, hit_rate=0.9))
    assert mla.kv_bytes < dense.kv_bytes / 3
    assert mla.attn_ms == pytest.approx(dense.attn_ms, rel=1e-9)


def test_sparse_attention_cuts_attention_time_and_storage_is_unchanged():
    shape, hw = ModelShape(), Hardware()
    dense = breakdown(PRESETS["mla"], shape, hw, Workload(prompt_tokens=131072, hit_rate=0.9))
    sparse = breakdown(PRESETS["mla"], shape, hw,
                       Workload(prompt_tokens=131072, hit_rate=0.9, compute_kind="sparse",
                                kv_read_fraction=0.1))
    assert sparse.attn_ms < dense.attn_ms / 5
    assert sparse.kv_bytes == pytest.approx(dense.kv_bytes)


def test_state_format_is_constant_in_context_length():
    shape = ModelShape()
    state = PRESETS["gdn_state"]
    short = state.bytes_for(1024, shape.n_layers)
    long = state.bytes_for(1_048_576, shape.n_layers)
    assert short == long
    assert PRESETS["mla"].bytes_for(1_048_576, shape.n_layers) > PRESETS["mla"].bytes_for(1024, shape.n_layers)


def test_format_scales_with_batch_and_layers():
    fmt = PRESETS["mla"]
    base = fmt.bytes_for(4096, n_layers=61, batch=1)
    assert fmt.bytes_for(4096, n_layers=61, batch=3) == pytest.approx(3 * base)
    assert fmt.bytes_for(4096, n_layers=122, batch=1) == pytest.approx(2 * base)


def test_cli_kv_cost(capsys):
    assert cli.main(["kv-cost", "--prompt", "131072", "--output", "4096",
                     "--hit-rate", "0.8", "--format", "mla"]) == 0
    out = capsys.readouterr().out
    assert "dominant" in out and "recompute" in out


def test_agentic_trace_shape():
    trace = agentic_trace(n_sessions=4, turns_per_session=3, seed=1)
    assert 0 < len(trace) <= 12
    assert all(len(r.blocks) >= 32 for r in trace)
    assert [r.at for r in trace] == sorted(r.at for r in trace)
