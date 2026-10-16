"""Shared fixtures: small shapes so the whole suite runs on a laptop CPU."""

from __future__ import annotations

import pytest
import torch

from linattn.core import AttnConfig
from linattn.harness import make_inputs, structured_inputs


@pytest.fixture
def cfg() -> AttnConfig:
    return AttnConfig(
        n_heads=4,
        n_kv_heads=2,      # GQA on purpose: the group-union effect shows up here
        head_dim=32,
        block_size=16,
        window_size=32,
        cache_budget=64,
        topk_blocks=3,
        topk_tokens=48,
        compress_ratio=8,
        obs_window=16,
        sink_size=2,
        stride=4,
    )


@pytest.fixture
def inputs(cfg):
    q, k, v = make_inputs(1, cfg.n_heads, 96, cfg.head_dim, kv_heads=cfg.n_kv_heads, seed=0)
    return q, k, v


@pytest.fixture
def structured(cfg):
    q, k, v = structured_inputs(1, cfg.n_heads, 192, cfg.head_dim, kv_heads=cfg.n_kv_heads, seed=1)
    return q, k, v


@pytest.fixture(params=["float64", "float32"])
def real_dtype(request):
    """Variants are validated in float64 (algorithmic) and float32 (numerical)."""
    return torch.float64 if request.param == "float64" else torch.float32
