"""Kernel parity: the C kernel, the Triton fallbacks, and the flex-mask wrapper.

The Triton kernels themselves require CUDA (marked ``gpu``); what runs everywhere
is the *parity contract* -- that every accelerated entry point agrees with the
torch reference that the semantics tests already pin down.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from linattn.kernels import describe_backends
from linattn.kernels.c_attention import build_library, paged_decode, paged_decode_torch
from linattn.kernels.flex_mask import check_block_mask, mask_to_block_mask
from linattn.kernels.triton.block_sparse import block_sparse_attention, block_sparse_attention_torch
from linattn.kernels.triton.delta_chunk import delta_chunk, delta_chunk_torch
from linattn.kernels.triton.fused_indexer import indexer_topk, indexer_topk_torch
from linattn.kernels.triton.quest_pages import page_bounds_torch, page_select_torch
from linattn.variants.recurrent import LinearAttentionConfig, LinearAttentionLayer


def test_backend_report_is_well_formed():
    report = describe_backends()
    for key in ("platform", "torch", "cuda", "triton_installed", "flex_attention_importable"):
        assert key in report


@pytest.mark.skipif(build_library() is None, reason="no C compiler available")
def test_c_kernel_matches_torch_reference():
    torch.manual_seed(0)
    t, dim, page = 4096, 64, 8
    k = torch.randn(t, dim)
    v = torch.randn(t, dim)
    q = torch.randn(dim)
    pages = torch.arange(0, 128)
    ref = paged_decode_torch(q, k, v, pages, page_size=page)
    for kernel in ("naive", "tiled"):
        out, stats = paged_decode(q, k, v, pages, page_size=page, kernel=kernel)
        assert torch.allclose(out, ref, atol=2e-4), f"{kernel} kernel disagrees with torch"
        assert stats["kv_entries"] == 128 * page
        assert stats["bytes"] > 0


@pytest.mark.skipif(build_library() is None, reason="no C compiler available")
def test_c_kernel_tiling_reads_fewer_entries_when_budget_is_small():
    """The IO accounting must reflect the selection, not the sequence length."""
    t, dim, page = 8192, 64, 16
    k = torch.randn(t, dim)
    v = torch.randn(t, dim)
    q = torch.randn(dim)
    pages = torch.arange(0, 8)  # 8 pages of 16 tokens = 128 entries
    _, stats = paged_decode(q, k, v, pages, page_size=page, kernel="tiled")
    assert stats["kv_entries"] == 128
    assert stats["kv_entries"] < stats["dense_entries"] / 8


def test_flex_mask_reduction_is_a_superset():
    torch.manual_seed(0)
    mask = torch.zeros(1, 2, 64, 64, dtype=torch.bool)
    mask[..., :32, :16] = True                     # one block-ish rectangle
    mask[..., 40, 10] = True                       # a single stray token
    reduced = mask_to_block_mask(mask, 16, 16)
    ok, coverage = check_block_mask(mask, reduced, 16, 16)
    assert ok, "block reduction dropped a selected entry"
    assert coverage >= float(mask.float().mean())


def test_block_sparse_fallback_matches_dense_when_all_blocks_selected():
    torch.manual_seed(0)
    b, h, t, d, bs = 1, 2, 32, 16, 8
    q = torch.randn(b, h, t, d, dtype=torch.float64)
    k = torch.randn(b, h, t, d, dtype=torch.float64)
    v = torch.randn(b, h, t, d, dtype=torch.float64)
    ids = torch.arange(t // bs).view(1, 1, 1, -1).expand(b, h, t, -1)
    out = block_sparse_attention(q, k, v, ids, block_size=bs, backend="torch")
    from linattn.core import dense_attention

    # the kernel itself is mask-free: with every block selected it must reproduce
    # *unmasked* attention (causality is the selector's job, not the kernel's)
    assert torch.allclose(out, dense_attention(q, k, v, causal=False), atol=1e-10)


def test_indexer_fallback_matches_reference():
    torch.manual_seed(0)
    b, tq, tk, hi, d, kk = 1, 8, 32, 3, 8, 5
    q_idx = torch.randn(b, tq, hi, d)
    k_idx = torch.randn(b, tk, hi, d)
    w = torch.rand(b, tq, hi)
    values, indices = indexer_topk(q_idx, k_idx, w, kk, backend="torch")
    ref_values, ref_indices = indexer_topk_torch(q_idx, k_idx, w, kk)
    assert torch.allclose(values, ref_values)
    assert torch.equal(indices, ref_indices)


def test_quest_page_bounds_match_bruteforce():
    torch.manual_seed(0)
    k = torch.randn(1, 2, 32, 8, dtype=torch.float64)
    mins, maxs = page_bounds_torch(k, page_size=8)
    for page in range(4):
        chunk = k[..., page * 8 : (page + 1) * 8, :]
        assert torch.allclose(mins[..., page, :], chunk.amin(dim=-2), atol=1e-12)
        assert torch.allclose(maxs[..., page, :], chunk.amax(dim=-2), atol=1e-12)
    ids, scores = page_select_torch(torch.randn(1, 2, 32, 8, dtype=torch.float64), mins, maxs, 8, 2)
    assert ids.shape == (1, 2, 32, 2)
    assert scores.shape == (1, 2, 32, 4)


def test_delta_chunk_fallback_matches_layer():
    torch.manual_seed(0)
    cfg = LinearAttentionConfig(d_model=32, n_heads=2, d_state=8, chunk=16, kind="gdn", decay_clamp=2.0)
    layer = LinearAttentionLayer(cfg).double()
    x = torch.randn(1, 16, 32, dtype=torch.float64)
    with torch.no_grad():
        y_ref, s_ref = layer.forward_recurrent(x)
    q, k, v, erase, write, g = layer._project(x)
    state = torch.zeros(1, cfg.n_heads, cfg.d_state, cfg.d_value, dtype=torch.float64)
    out, new_state = delta_chunk(q, k, v, erase, g, state, use_erase=True, backend="torch")
    projected = layer.o_proj(out.transpose(1, 2).reshape(1, 16, -1))
    assert torch.allclose(projected, y_ref, atol=1e-8)
    assert torch.allclose(new_state, s_ref, atol=1e-8)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_block_sparse_matches_fallback():  # pragma: no cover - GPU only
    torch.manual_seed(0)
    b, h, t, d, bs = 1, 4, 128, 64, 32
    q = torch.randn(b, h, t, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, t, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, t, d, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, t // bs, (b, h, t, 2), device="cuda")
    ids[..., 1] = torch.arange(t, device="cuda") // bs
    fast = block_sparse_attention(q, k, v, ids, block_size=bs, backend="triton")
    slow = block_sparse_attention_torch(q, k, v, ids, block_size=bs)
    assert torch.allclose(fast.float(), slow.float(), atol=2e-2, rtol=2e-2)
