"""The oracle itself has to be right before anything else is measured."""

from __future__ import annotations

import math

import pytest
import torch

from linattn.core import AttnConfig, DenseAttention, causal_mask, dense_attention, repeat_kv


def test_dense_matches_manual_softmax():
    torch.manual_seed(0)
    q = torch.randn(2, 2, 5, 8, dtype=torch.float64)
    k = torch.randn(2, 2, 5, 8, dtype=torch.float64)
    v = torch.randn(2, 2, 5, 8, dtype=torch.float64)
    out = dense_attention(q, k, v)
    scores = q @ k.transpose(-1, -2) / math.sqrt(8)
    allowed = torch.ones(5, 5, dtype=torch.bool).tril()
    weights = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
    assert torch.allclose(out, weights @ v, atol=1e-12)


def test_causal_mask_semantics():
    mask = causal_mask(3, 6, offset=3)
    # query 0 (absolute position 3) may read keys 0..3
    assert mask[0].tolist() == [True, True, True, True, False, False]
    assert mask[2].tolist() == [True, True, True, True, True, True]


def test_gqa_expansion_matches_repeat():
    x = torch.arange(8, dtype=torch.float64).reshape(1, 2, 1, 4)
    expanded = repeat_kv(x, 3)
    assert expanded.shape == (1, 6, 1, 4)
    assert torch.allclose(expanded[:, 0], expanded[:, 1]) and torch.allclose(expanded[:, 1], expanded[:, 2])


def test_dense_variant_matches_function():
    cfg = AttnConfig(n_heads=2, n_kv_heads=1, head_dim=8)
    variant = DenseAttention(cfg)
    torch.manual_seed(0)
    q = torch.randn(1, 2, 7, 8, dtype=torch.float64)
    k = torch.randn(1, 1, 7, 8, dtype=torch.float64)
    v = torch.randn(1, 1, 7, 8, dtype=torch.float64)
    out, _ = variant.forward(q, k, v)
    assert torch.allclose(out, dense_attention(q, k, v), atol=1e-12)


def test_dense_chunked_matches_single_shot():
    cfg = AttnConfig(n_heads=2, n_kv_heads=2, head_dim=8)
    variant = DenseAttention(cfg)
    torch.manual_seed(0)
    q = torch.randn(1, 2, 32, 8, dtype=torch.float64)
    k = torch.randn(1, 2, 32, 8, dtype=torch.float64)
    v = torch.randn(1, 2, 32, 8, dtype=torch.float64)
    full, _ = variant.forward(q, k, v, chunk=32)
    chunked, _ = variant.forward(q, k, v, chunk=7)
    assert torch.allclose(full, chunked, atol=1e-12)


def test_incremental_decode_matches_prefill():
    """The cache contract: feeding tokens one at a time equals a single prefill."""
    cfg = AttnConfig(n_heads=2, n_kv_heads=2, head_dim=8)
    variant = DenseAttention(cfg)
    torch.manual_seed(0)
    q = torch.randn(1, 2, 16, 8, dtype=torch.float64)
    k = torch.randn(1, 2, 16, 8, dtype=torch.float64)
    v = torch.randn(1, 2, 16, 8, dtype=torch.float64)
    prefill, _ = variant.forward(q, k, v, chunk=16)
    state = None
    pieces = []
    for i in range(16):
        out, state = variant.forward(q[:, :, i : i + 1], k[:, :, i : i + 1], v[:, :, i : i + 1], state=state)
        pieces.append(out)
    decode = torch.cat(pieces, dim=-2)
    assert torch.allclose(prefill, decode, atol=1e-12)
