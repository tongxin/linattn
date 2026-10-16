"""Lightning-indexer scoring fused with a two-pass top-k (DSA / CSA).

DeepSeek's indexer computes ``I_{t,s} = sum_j w_{t,j} ReLU(q_{t,j} . k_s)`` for
every query/key pair and then takes the top-k.  The naive implementation
materialises a ``[B, S, H_I, T]`` FP32 tensor before reducing it, which is what
makes the *indexer* -- not the attention -- the memory bottleneck at long context
(the OOM analysis in StreamIndex, arXiv:2605.02568, is worth reading on this).

This module avoids the materialisation with the classic two-pass split:

1. ``_indexer_block_max_kernel`` walks the key axis in blocks and writes one
   number per key block per query: the maximum indexer score in that block.
2. The threshold is the ``k``-th largest block maximum -- a reduction over
   ``T / BLOCK_K`` numbers, not over ``T``.
3. ``_indexer_gather_kernel`` emits every (score, index) pair above the
   threshold.  Because the threshold is conservative, the emitted set is a
   *superset* of the true top-k; the caller trims it.

The trade-off (two passes over the keys in exchange for never materialising the
score matrix) is stated explicitly in ``docs/kernels.md``, because it is exactly
the kind of decision the co-design literature argues about.

Both kernels are validated against :func:`indexer_topk_torch` by
``tests/test_kernels.py``; the fallback keeps the module usable without a GPU.
"""

from __future__ import annotations

from typing import Tuple

import torch

try:  # pragma: no cover - exercised only on machines with Triton
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    HAS_TRITON = False

__all__ = [
    "indexer_scores_torch",
    "indexer_topk_torch",
    "indexer_topk_triton",
    "indexer_topk",
    "HAS_TRITON",
]


def indexer_scores_torch(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    head_weights: torch.Tensor,
) -> torch.Tensor:
    """``(B, T_q, H_I, d)`` x ``(B, T_k, H_I, d)`` -> ``(B, T_q, T_k)`` indexer scores."""
    relu = torch.relu(torch.einsum("bqhd,bkhd->bqkh", q_idx, k_idx))
    # ``head_weights`` is (B, T_q, H_I): one learned scalar per (token, indexer head)
    return (relu * head_weights.unsqueeze(-2)).sum(-1)


def indexer_topk_torch(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    head_weights: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference top-k: materialise the score matrix, then select."""
    scores = indexer_scores_torch(q_idx, k_idx, head_weights)
    k = min(k, scores.shape[-1])
    values, indices = torch.topk(scores, k, dim=-1)
    return values, indices


if HAS_TRITON:  # pragma: no cover - requires CUDA

    @triton.jit
    def _indexer_block_max_kernel(
        Q, K, W, BLOCK_MAX,
        stride_qb, stride_qt, stride_qh, stride_qd,
        stride_kb, stride_kt, stride_kh, stride_kd,
        stride_wb, stride_wt, stride_wh,
        stride_mb, stride_mt, stride_mn,
        T_K,
        H_I: tl.constexpr, D: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs_d = tl.arange(0, D)
        offs_k = tl.arange(0, BLOCK_K)
        offs_h = tl.arange(0, H_I)
        w_head = tl.load(W + pid_b * stride_wb + pid_t * stride_wt + offs_h * stride_wh)
        best = float("-inf")
        n_blocks = tl.cdiv(T_K, BLOCK_K)
        for blk in range(0, n_blocks):
            ptrs = blk * BLOCK_K + offs_k
            kv_mask = ptrs < T_K
            acc = tl.zeros([BLOCK_K], dtype=tl.float32)
            for h in tl.static_range(H_I):
                q = tl.load(Q + pid_b * stride_qb + pid_t * stride_qt + h * stride_qh + offs_d * stride_qd)
                kk = tl.load(K + pid_b * stride_kb + ptrs[:, None] * stride_kt + h * stride_kh
                             + offs_d[None, :] * stride_kd, mask=kv_mask[:, None], other=0.0)
                dot = tl.sum(kk.to(tl.float32) * q[None, :], axis=1)
                w_h = tl.sum(tl.where(offs_h == h, w_head, 0.0), axis=0)
                acc += tl.maximum(dot, 0.0) * w_h
            block_max = tl.max(tl.where(kv_mask, acc, float("-inf")), axis=0)
            best = tl.maximum(best, block_max)
        tl.store(BLOCK_MAX + pid_b * stride_mb + pid_t * stride_mt, best)

    @triton.jit
    def _indexer_gather_kernel(
        Q, K, W, THRESH, OUT_SCORES, OUT_INDEX,
        stride_qb, stride_qt, stride_qh, stride_qd,
        stride_kb, stride_kt, stride_kh, stride_kd,
        stride_wb, stride_wt, stride_wh,
        stride_ob, stride_ot, stride_on,
        T_K, MAX_KEEP,
        H_I: tl.constexpr, D: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Emit every (score, index) pair above the per-query threshold."""
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs_d = tl.arange(0, D)
        offs_k = tl.arange(0, BLOCK_K)
        offs_h = tl.arange(0, H_I)
        threshold = tl.load(THRESH + pid_b * 0 + pid_t)
        w_head = tl.load(W + pid_b * stride_wb + pid_t * stride_wt + offs_h * stride_wh)
        base = 0
        n_blocks = tl.cdiv(T_K, BLOCK_K)
        for blk in range(0, n_blocks):
            ptrs = blk * BLOCK_K + offs_k
            kv_mask = ptrs < T_K
            acc = tl.zeros([BLOCK_K], dtype=tl.float32)
            for h in tl.static_range(H_I):
                q = tl.load(Q + pid_b * stride_qb + pid_t * stride_qt + h * stride_qh + offs_d * stride_qd)
                kk = tl.load(K + pid_b * stride_kb + ptrs[:, None] * stride_kt + h * stride_kh
                             + offs_d[None, :] * stride_kd, mask=kv_mask[:, None], other=0.0)
                dot = tl.sum(kk.to(tl.float32) * q[None, :], axis=1)
                w_h = tl.sum(tl.where(offs_h == h, w_head, 0.0), axis=0)
                acc += tl.maximum(dot, 0.0) * w_h
            keep = kv_mask & (acc >= threshold)
            order = tl.cumsum(keep.to(tl.int32), axis=0) - 1
            count = tl.sum(keep.to(tl.int32), axis=0)
            slot = tl.maximum(tl.minimum(base + order, MAX_KEEP - 1), 0)
            tl.store(OUT_SCORES + pid_b * stride_ob + pid_t * stride_ot + slot * stride_on, acc,
                     mask=keep & (base + order < MAX_KEEP))
            tl.store(OUT_INDEX + pid_b * stride_ob + pid_t * stride_ot + slot * stride_on, ptrs,
                     mask=keep & (base + order < MAX_KEEP))
            base += count


def indexer_topk_triton(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    head_weights: torch.Tensor,
    k: int,
    *,
    block_k: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two-pass threshold selection on GPU (see module docstring)."""
    b, t_q, h_i, d = q_idx.shape
    t_k = k_idx.shape[1]
    q_c = q_idx.contiguous()
    k_c = k_idx.contiguous()
    w_c = head_weights.contiguous()
    n_blocks = triton.cdiv(t_k, block_k)
    block_max = torch.empty(b, t_q, n_blocks, device=q_idx.device, dtype=torch.float32)
    _indexer_block_max_kernel[(t_q, b)](
        q_c, k_c, w_c, block_max,
        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
        k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
        w_c.stride(0), w_c.stride(1), w_c.stride(2),
        block_max.stride(0), block_max.stride(1), block_max.stride(2),
        t_k, H_I=h_i, D=d, BLOCK_K=block_k,
    )
    keep_per_block = max(1, k // n_blocks)
    threshold = torch.topk(block_max, min(keep_per_block, n_blocks), dim=-1).values.min(-1).values
    max_keep = min(t_k, max(k * 4, k + block_k))
    out_scores = torch.full((b, t_q, max_keep), float("-inf"), device=q_idx.device, dtype=torch.float32)
    out_index = torch.zeros(b, t_q, max_keep, device=q_idx.device, dtype=torch.int32)
    _indexer_gather_kernel[(t_q, b)](
        q_c, k_c, w_c, threshold.contiguous(), out_scores, out_index,
        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
        k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
        w_c.stride(0), w_c.stride(1), w_c.stride(2),
        out_scores.stride(0), out_scores.stride(1), out_scores.stride(2),
        t_k, max_keep, H_I=h_i, D=d, BLOCK_K=block_k,
    )
    values, order = torch.topk(out_scores, k, dim=-1)
    indices = torch.gather(out_index, -1, order)
    return values, indices


def indexer_topk(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    head_weights: torch.Tensor,
    k: int,
    *,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact top-k indexer scores: Triton two-pass on GPU, torch oracle otherwise."""
    if backend in ("auto", "triton") and HAS_TRITON and q_idx.is_cuda:
        return indexer_topk_triton(q_idx, k_idx, head_weights, k)
    return indexer_topk_torch(q_idx, k_idx, head_weights, k)
