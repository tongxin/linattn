"""NSA, MLA and the hybrid stack."""

from __future__ import annotations

import pytest
import torch

from linattn.core import dense_attention
from linattn.metrics import rel_error
from linattn.registry import build
from linattn.variants.hybrid import HybridConfig, HybridLM, make_mqar_batch
from linattn.variants.mla import MLAttention
from linattn.variants.native_sparse import NSAAttention


def test_mla_materialisation_is_a_projection(cfg):
    variant = MLAttention(cfg.with_(n_kv_heads=1))       # the latent is a single KV head
    latent = torch.randn(1, 1, 32, variant.latent_dim, dtype=torch.float64)
    k, v = variant.materialize(latent)
    assert k.shape == (1, cfg.n_heads, 32, cfg.head_dim)
    assert v.shape == k.shape


def test_mla_output_matches_dense_over_materialised_kv(cfg):
    torch.manual_seed(0)
    variant = MLAttention(cfg.with_(n_kv_heads=1))
    q = torch.randn(1, cfg.n_heads, 32, cfg.head_dim, dtype=torch.float64)
    latent = torch.randn(1, 1, 32, variant.latent_dim, dtype=torch.float64)
    out, _ = variant.forward(q, latent, latent)
    k, v = variant.materialize(latent)
    assert torch.allclose(out, dense_attention(q, k, v), atol=1e-10)


def test_mla_cache_is_smaller_than_dense(cfg):
    variant = MLAttention(cfg)
    assert variant.memory_bytes(4096) < variant.dense_equivalent_bytes(4096)
    assert variant.compression_ratio() > 1.0


def test_nsa_branches_are_all_used(cfg, structured):
    q, k, v = structured
    variant = NSAAttention(cfg)
    out, _ = variant.forward(q, k, v, chunk=q.shape[-2])
    # with uniform gates the output must lie inside the convex hull of the branches
    state = variant.prepare_state(q.shape[0], q.device, q.dtype)
    state = variant.push_kv(state, k, v)
    branches = [
        variant.branch_compression(q, k, v, 0),
        variant.branch_selection(q, k, v, 0),
        variant.branch_window(q, k, v, 0),
    ]
    stacked = torch.stack(branches)
    assert (out <= stacked.amax(0) + 1e-6).all() and (out >= stacked.amin(0) - 1e-6).all()


def test_nsa_gate_fitting_improves_fidelity(cfg, structured):
    q, k, v = structured
    target = dense_attention(q.to(torch.float64), k.to(torch.float64), v.to(torch.float64))
    variant = NSAAttention(cfg)
    before, _ = variant.forward(q, k, v, chunk=q.shape[-2])
    err_before = rel_error(before, target)
    variant.fit_gates(q.double(), k.double(), v.double(), steps=80, lr=0.05)
    after, _ = variant.forward(q, k, v, chunk=q.shape[-2])
    err_after = rel_error(after, target)
    assert err_after <= err_before + 1e-6, "learned gates should not make the output worse"


def test_hybrid_layer_pattern():
    cfg = HybridConfig(n_layers=6, attn_period=3)
    model = HybridLM(cfg)
    assert model.kinds == ["attn", "linear", "linear", "attn", "linear", "linear"]


def test_hybrid_state_scaling_law():
    """Attention state grows with sequence length; linear state does not."""
    hybrid = HybridLM(HybridConfig(n_layers=4, attn_period=4))     # 1 attention, 3 linear
    linear_only = HybridLM(HybridConfig(n_layers=4, attn_period=0))
    assert hybrid.state_bytes(8192) > hybrid.state_bytes(1024)
    assert linear_only.state_bytes(8192) == linear_only.state_bytes(1024)
    # and the hybrid's growth is entirely the attention layer's
    growth = hybrid.state_bytes(8192) - hybrid.state_bytes(1024)
    assert growth > 0 and linear_only.state_bytes(1024) > 0


def test_mqar_batch_is_learnable_in_shape():
    seq, target = make_mqar_batch(2, 64, 32, n_pairs=3, seed=0)
    assert seq.shape == target.shape == (2, 64)
    assert (target != -100).sum() == 6  # 3 pairs x 2 batch, one supervised slot each
