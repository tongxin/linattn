"""Chunkwise delta-rule scan for SSD / Gated DeltaNet / KDA.

The kernel implements one chunk of the algorithm derived in
``linattn.variants.recurrent``:

1. decay-normalise the state inside the chunk (``d = k/G_i`` for the erase-left and
   write-left factor, ``e = k*G_i`` for the erase-right factor, ``q~ = q*G_i``);
2. form ``M_i,j = beta_i (e_i . d_j)`` for ``j < i`` with ``tl.dot`` and a strictly
   lower mask;
3. solve ``(I + M) Y = diag(beta) (V - E S_0)`` by forward substitution over the
   chunk rows (the erase interactions are sequential within a chunk and parallel
   across chunks, which is why this is a chunked *scan*);
4. emit the state ``S = diag(exp(gc_last)) (S_0 + D^T Y)`` and the outputs
   ``O = Q~ S_0 + tril(Q~ D^T) Y`` with two more ``tl.dot``s.

``HAS_ERASE: tl.constexpr`` selects SSD (False) versus the delta-rule models
(True), so a single kernel covers all three variants in the taxonomy.

Validation: ``tests/test_kernels.py`` checks the kernel against
``LinearAttentionLayer.forward_recurrent`` on a GPU, and the torch fallback
against it on any machine.

Implementation status: the substitution below is written as a sequence of row
updates inside the kernel, which is correct but not the fastest form; the
production version (as in the FLA library and in the Kimi Linear release) uses a
blocked forward substitution where the ``16 x 16`` diagonal blocks of ``(I + M)``
are inverted with ``tl.dot`` and the off-diagonal blocks are folded in with
matmuls.  The blocked form is listed as the next step in ``docs/kernels.md``
rather than being written untested here.
"""

from __future__ import annotations

from typing import Tuple

import torch

from linattn.variants.recurrent import LinearAttentionConfig, LinearAttentionLayer

try:  # pragma: no cover
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    HAS_TRITON = False

__all__ = ["delta_chunk_torch", "delta_chunk_triton", "delta_chunk", "HAS_TRITON"]


def delta_chunk_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    state: torch.Tensor,
    *,
    use_erase: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch implementation of exactly the same chunk math (the parity oracle).

    Shapes: ``q,k``: ``(B,H,C,D_K)``; ``v``: ``(B,H,C,D_V)``; ``beta``: ``(B,H,C)``;
    ``log_decay``: ``(B,H,C,D_K)``; ``state``: ``(B,H,D_K,D_V)``.
    """
    b, h, c, dk = k.shape
    dv = v.shape[-1]
    if log_decay.dim() == 3:                     # per-head decay: broadcast per channel
        log_decay = log_decay.unsqueeze(-1).expand(b, h, c, dk)
    gc = log_decay.cumsum(dim=2)
    G = gc.exp()
    d, e, qt = k / G, k * G, q * G
    tril = torch.ones(c, c, dtype=torch.bool, device=k.device).tril(-1)
    tril0 = torch.ones(c, c, dtype=torch.bool, device=k.device).tril(0)
    read = torch.einsum("bhcd,bhde->bhce", e, state)
    if use_erase:
        m = torch.einsum("bhcd,bhjd->bhcj", e, d) * tril * beta.unsqueeze(-1)
        rhs = beta.unsqueeze(-1) * (v - read)
    else:
        m = torch.zeros(b, h, c, c, dtype=k.dtype, device=k.device)
        rhs = beta.unsqueeze(-1) * v
    y = torch.linalg.solve_triangular(torch.eye(c, dtype=k.dtype, device=k.device) + m, rhs, upper=False)
    new_state = torch.exp(gc[:, :, -1, :]).unsqueeze(-1) * (
        state + torch.einsum("bhcd,bhce->bhde", d, y)
    )
    qd = torch.einsum("bhcd,bhjd->bhcj", qt, d) * tril0
    out = torch.einsum("bhcd,bhde->bhce", qt, state) + torch.einsum("bhcj,bhje->bhce", qd, y)
    return out, new_state


if HAS_TRITON:  # pragma: no cover - requires CUDA

    @triton.jit
    def _delta_chunk_kernel(
        Q, K, V, BETA, G, STATE, OUT, NEW_STATE,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_bb, stride_bh, stride_bt,
        stride_gb, stride_gh, stride_gt, stride_gd,
        stride_sb, stride_sh, stride_sd, stride_se,
        HAS_ERASE: tl.constexpr, BLOCK_C: tl.constexpr, D_K: tl.constexpr, D_V: tl.constexpr,
    ):
        pid_c = tl.program_id(0)      # chunk index
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        rows = tl.arange(0, BLOCK_C)
        dk = tl.arange(0, D_K)
        dv = tl.arange(0, D_V)
        row_mask = rows < BLOCK_C
        base_t = pid_c * BLOCK_C

        k = tl.load(K + pid_b * stride_kb + pid_h * stride_kh + (base_t + rows)[:, None] * stride_kt
                    + dk[None, :] * stride_kd, mask=row_mask[:, None], other=0.0)
        q = tl.load(Q + pid_b * stride_qb + pid_h * stride_qh + (base_t + rows)[:, None] * stride_qt
                    + dk[None, :] * stride_qd, mask=row_mask[:, None], other=0.0)
        v = tl.load(V + pid_b * stride_vb + pid_h * stride_vh + (base_t + rows)[:, None] * stride_vt
                    + dv[None, :] * stride_vd, mask=row_mask[:, None], other=0.0)
        beta = tl.load(BETA + pid_b * stride_bb + pid_h * stride_bh + (base_t + rows) * stride_bt,
                       mask=row_mask, other=0.0)
        g = tl.load(G + pid_b * stride_gb + pid_h * stride_gh + (base_t + rows)[:, None] * stride_gt
                    + dk[None, :] * stride_gd, mask=row_mask[:, None], other=0.0)
        gc = tl.cumsum(g, axis=0)
        G_i = tl.exp(gc)
        d = k / G_i
        e = k * G_i
        qt = q * G_i

        state = tl.load(STATE + pid_b * stride_sb + pid_h * stride_sh + dk[:, None] * stride_sd
                        + dv[None, :] * stride_se)

        lower = rows[:, None] > rows[None, :]
        # M[i, j] = beta_i * (e_i . d_j) for j < i
        m = tl.dot(e, tl.trans(d)) * tl.where(lower, beta[:, None], 0.0)
        read = tl.dot(e, state)
        if HAS_ERASE:
            rhs = beta[:, None] * (v - read)
        else:
            m = tl.zeros([BLOCK_C, BLOCK_C], dtype=tl.float32)
            rhs = beta[:, None] * v

        # forward substitution: y_i = rhs_i - sum_{j<i} M[i, j] y_j
        y = tl.zeros([BLOCK_C, D_V], dtype=tl.float32)
        for i in tl.static_range(BLOCK_C):
            m_row = tl.sum(tl.where(rows[:, None] == i, m, 0.0), axis=0)      # (BLOCK_C,)
            corr = tl.sum(m_row[:, None] * y, axis=0)                          # (D_V,)
            y_i = tl.sum(tl.where(rows[:, None] == i, rhs, 0.0), axis=0) - corr
            y = tl.where(rows[:, None] == i, y_i[None, :], y)

        decay_last = tl.exp(gc[BLOCK_C - 1, :])                                # (D_K,)
        new_state = decay_last[:, None] * (state + tl.dot(tl.trans(d), y))
        causal = rows[:, None] >= rows[None, :]
        qd = tl.dot(qt, tl.trans(d)) * tl.where(causal, 1.0, 0.0)
        out = tl.dot(qt, state) + tl.dot(qd, y)

        tl.store(OUT + pid_b * stride_qb + pid_h * stride_qh + (base_t + rows)[:, None] * stride_qt
                 + dk[None, :] * stride_qd, out.to(OUT.dtype.element_ty), mask=row_mask[:, None])
        tl.store(NEW_STATE + pid_b * stride_sb + pid_h * stride_sh + dk[:, None] * stride_sd
                 + dv[None, :] * stride_se, new_state.to(NEW_STATE.dtype.element_ty))


def delta_chunk_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    state: torch.Tensor,
    *,
    use_erase: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the Triton chunk kernel over all chunks (host-side chunk loop)."""
    b, h, c_total, dk = k.shape
    block_c = 32
    dv = v.shape[-1]
    if dk > 128 or dv > 128:
        raise ValueError("kernel supports head dims up to 128")
    if c_total % block_c:
        pad = block_c - (c_total % block_c)
        q = torch.nn.functional.pad(q, (0, 0, 0, pad))
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        beta = torch.nn.functional.pad(beta, (0, pad))
        log_decay = torch.nn.functional.pad(log_decay, (0, 0, 0, pad))
    out = torch.empty_like(q)
    new_state = torch.empty_like(state)
    n_chunks = k.shape[2] // block_c
    _delta_chunk_kernel[(n_chunks, h, b)](
        q, k, v, beta, log_decay, state, out, new_state,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        log_decay.stride(0), log_decay.stride(1), log_decay.stride(2), log_decay.stride(3),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        HAS_ERASE=use_erase, BLOCK_C=block_c, D_K=dk, D_V=dv,
    )
    return out[..., :c_total, :], new_state


def delta_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    state: torch.Tensor,
    *,
    use_erase: bool = True,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to Triton on GPU, torch otherwise."""
    if backend in ("auto", "triton") and HAS_TRITON and q.is_cuda:
        return delta_chunk_triton(q, k, v, beta, log_decay, state, use_erase=use_erase)
    return delta_chunk_torch(q, k, v, beta, log_decay, state, use_erase=use_erase)
