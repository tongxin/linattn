"""Quest's per-page metadata maintenance and query-aware page scoring.

Quest keeps, for every page and every KV head, the channel-wise minimum and
maximum of the keys in that page.  Two properties make this a nice kernel target:

* the metadata is *tiny* -- ``2 * D`` numbers per page per head, i.e. ``2*D/P``
  values per token, which is why Quest's selection overhead stays negligible;
* it is *incremental* -- appending a token only touches one page, so the update is
  O(D) rather than O(T*D).

The scope kernel then computes, for each query, the upper bound
``sum_d max(q_d * m_d, q_d * M_d)`` for every page and keeps the top-k.  The page
count is ``T / P`` (16x smaller than T for the paper's P=16), so the selection
tensor is 16x smaller than a token-level score matrix -- the reason Quest-style
selection is cheaper than token-level selectors at equal recall.
"""

from __future__ import annotations

from typing import Tuple

import torch

try:  # pragma: no cover
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    HAS_TRITON = False

__all__ = ["page_bounds_torch", "page_scores_torch", "page_select_torch", "page_update_triton", "page_select_triton",
           "HAS_TRITON"]


def page_bounds_torch(k: torch.Tensor, page_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Channel-wise min/max per page: ``(B,H_kv,T,D) -> (B,H_kv,n_pages,D)``."""
    b, h, t, d = k.shape
    n_pages = (t + page_size - 1) // page_size
    pad = n_pages * page_size - t
    # pad with +/- inf so that the padded rows cannot win either reduction
    k_min = torch.nn.functional.pad(k, (0, 0, 0, pad), value=float("inf"))
    k_max = torch.nn.functional.pad(k, (0, 0, 0, pad), value=float("-inf"))
    mins = k_min.view(b, h, n_pages, page_size, d).amin(dim=-2)
    maxim = k_max.view(b, h, n_pages, page_size, d).amax(dim=-2)
    return mins, maxim


def page_scores_torch(q: torch.Tensor, mins: torch.Tensor, maxim: torch.Tensor) -> torch.Tensor:
    """Upper-bound page scores ``(B,H,T_q,n_pages)``."""
    qe = q.unsqueeze(-2)
    return torch.maximum(qe * mins.unsqueeze(2), qe * maxim.unsqueeze(2)).sum(-1)


def page_select_torch(
    q: torch.Tensor,
    mins: torch.Tensor,
    maxim: torch.Tensor,
    page_size: int,
    topk: int,
    *,
    offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k page ids per query plus the page scores."""
    scores = page_scores_torch(q, mins, maxim)
    t_q = q.shape[-2]
    n_pages = scores.shape[-1]
    page_start = torch.arange(n_pages, device=q.device) * page_size
    q_pos = torch.arange(t_q, device=q.device) + offset
    visible = page_start.unsqueeze(0) <= q_pos.unsqueeze(1)
    scores = scores.masked_fill(~visible, float("-inf"))
    own = (q_pos // page_size).clamp(max=n_pages - 1)
    scores[..., own] = float("inf")
    k = max(1, min(topk, n_pages))
    ids = torch.topk(scores, k, dim=-1).indices
    return ids, scores


if HAS_TRITON:  # pragma: no cover - requires CUDA

    @triton.jit
    def _page_update_kernel(
        K, PAGE_MIN, PAGE_MAX, T, PAGE_SIZE: tl.constexpr, D: tl.constexpr, N_PAGES: tl.constexpr,
    ):
        """Maintain min/max for one page of one (batch, kv head)."""
        page = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        rows = page * PAGE_SIZE + tl.arange(0, PAGE_SIZE)
        offs_d = tl.arange(0, D)
        mask = rows[:, None] < T
        k = tl.load(K + (pid_b * N_PAGES * 0) + (pid_b * T + rows)[:, None] * D + offs_d[None, :],
                    mask=mask, other=0.0)
        mins = tl.min(tl.where(mask, k, float("inf")), axis=0)
        maxs = tl.max(tl.where(mask, k, float("-inf")), axis=0)
        out = pid_b * N_PAGES * D + pid_h * 0 + page * D + offs_d
        tl.store(PAGE_MIN + out, mins)
        tl.store(PAGE_MAX + out, maxs)

    @triton.jit
    def _page_score_kernel(
        Q, PAGE_MIN, PAGE_MAX, SCORES,
        stride_qb, stride_qh, stride_qt, stride_qd,
        N_PAGES, D: tl.constexpr, BLOCK_P: tl.constexpr,
    ):
        """Upper-bound score for every page, for one (batch, head, query).

        Selection (top-k) is deliberately left to the caller: a fused top-k inside
        this kernel is the remaining engineering step, and the two-pass approach in
        ``fused_indexer.py`` is the template for it.  Writing all page scores costs
        ``T/P`` floats per query, i.e. 16x less than a token-level score matrix at
        the paper's page size, which is the whole reason Quest's overhead is small.
        """
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        offs_d = tl.arange(0, D)
        q = tl.load(Q + pid_b * stride_qb + pid_h * stride_qh + pid_t * stride_qt + offs_d * stride_qd)
        for blk in range(0, tl.cdiv(N_PAGES, BLOCK_P)):
            pages = blk * BLOCK_P + tl.arange(0, BLOCK_P)
            pmask = pages < N_PAGES
            base = pid_b * N_PAGES * D + pages[:, None] * D + offs_d[None, :]
            mins = tl.load(PAGE_MIN + base, mask=pmask[:, None], other=0.0)
            maxs = tl.load(PAGE_MAX + base, mask=pmask[:, None], other=0.0)
            lo = mins * q[None, :]
            hi = maxs * q[None, :]
            s = tl.sum(tl.maximum(lo, hi), axis=1)
            tl.store(SCORES + ((pid_b * tl.num_programs(1) + pid_h) * tl.num_programs(0) + pid_t) * N_PAGES
                     + pages, s, mask=pmask)


def page_update_triton(k: torch.Tensor, page_size: int, n_pages: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU path for the metadata update (falls back to torch without Triton)."""
    if not (HAS_TRITON and k.is_cuda):
        return page_bounds_torch(k, page_size)
    b, h, t, d = k.shape
    mins = torch.empty(b, h, n_pages, d, device=k.device, dtype=k.dtype)
    maxs = torch.empty_like(mins)
    _page_update_kernel[(n_pages, h, b)](k, mins, maxs, t, PAGE_SIZE=page_size, D=d, N_PAGES=n_pages)
    return mins, maxs


def page_select_triton(q, mins, maxim, page_size: int, topk: int, *, offset: int = 0):
    """GPU path for page scoring, with the top-k left to torch (see kernel docstring)."""
    if not (HAS_TRITON and q.is_cuda):
        return page_select_torch(q, mins, maxim, page_size, topk, offset=offset)
    b, h, t_q, d = q.shape
    n_pages = mins.shape[-2]
    scores = torch.empty(b, h, t_q, n_pages, device=q.device, dtype=torch.float32)
    _page_score_kernel[(t_q, h, b)](q, mins, maxim, scores, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                                    n_pages, D=d, BLOCK_P=64)
    q_pos = torch.arange(t_q, device=q.device) + offset
    page_start = torch.arange(n_pages, device=q.device) * page_size
    visible = page_start.unsqueeze(0) <= q_pos.unsqueeze(1)
    scores = scores.masked_fill(~visible, float("-inf"))
    own = (q_pos // page_size).clamp(max=n_pages - 1)
    scores[..., own] = float("inf")
    ids = torch.topk(scores, max(1, min(topk, n_pages)), dim=-1).indices
    return ids, scores
