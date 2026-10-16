"""Block-sparse attention over a selected set of KV blocks.

This is the execution kernel shared by every selection variant in
``linattn.variants.selection``: the difference between Quest, MInference,
XAttention, MoBA and DSA is *which* blocks are in ``block_ids``, not how the
attention over them is computed.  Making that separation explicit is the main
architectural lesson of the 2024-2026 literature: a good selector plus a generic
block-sparse kernel beats a bespoke kernel per method.

Layout (decode path, which is what matters for long-horizon agentic workloads):

* one program per ``(batch, kv_head, query_token)``
* loop over the selected key blocks, ``BLOCK_K`` tokens at a time
* online softmax (running max/sum) exactly as in FlashAttention

The torch fallback ``block_sparse_attention_torch`` implements the same
semantics with a gather, so the two are directly comparable:
``tests/test_kernels.py`` asserts equality when a GPU is present.
"""

from __future__ import annotations

from typing import Optional

import torch

try:  # pragma: no cover - exercised only on machines with Triton
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    HAS_TRITON = False

__all__ = ["block_sparse_attention_torch", "block_sparse_attention", "HAS_TRITON"]


def block_sparse_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """Reference implementation: gather the selected blocks, then attention.

    ``q``: ``(B, H, T_q, D)``; ``k``/``v``: ``(B, H_kv, T_k, D)``;
    ``block_ids``: ``(B, H_kv, n_blocks)`` with the indices of the key blocks to
    read, or ``(B, H_kv, T_q, n_blocks)`` for per-query selection (MoBA/DSA).
    """
    if block_size is None:
        raise ValueError("block_size is required")
    b, h_q, t_q, d = q.shape
    h_kv, t_k = k.shape[1], k.shape[-2]
    group = h_q // h_kv
    if block_ids.dim() == 3:
        block_ids = block_ids.unsqueeze(-2).expand(b, h_kv, t_q, block_ids.shape[-1])
    n_blocks = block_ids.shape[-1]
    offsets = torch.arange(block_size, device=k.device)
    raw = block_ids.unsqueeze(-1) * block_size + offsets.view(1, 1, 1, 1, -1)      # (B,Hkv,Tq,N,bs)
    valid = (raw < t_k).reshape(b, h_kv, t_q, n_blocks * block_size)
    flat = raw.reshape(b, h_kv, -1).clamp(max=t_k - 1)                             # (B,Hkv,M)
    gather_idx = flat.unsqueeze(-1).expand(b, h_kv, flat.shape[-1], d)
    k_sel = torch.gather(k, -2, gather_idx).reshape(b, h_kv, t_q, n_blocks * block_size, d)
    v_sel = torch.gather(v, -2, gather_idx).reshape(b, h_kv, t_q, n_blocks * block_size, d)
    k_sel = k_sel.unsqueeze(2).expand(b, h_kv, group, t_q, -1, d).reshape(b, h_q, t_q, -1, d)
    v_sel = v_sel.unsqueeze(2).expand(b, h_kv, group, t_q, -1, d).reshape(b, h_q, t_q, -1, d)
    valid = valid.unsqueeze(2).expand(b, h_kv, group, t_q, -1).reshape(b, h_q, t_q, -1)
    scores = torch.einsum("bhtd,bhtnd->bhtn", q, k_sel) / (d**0.5)
    scores = scores.masked_fill(~valid, float("-inf"))
    out = torch.einsum("bhtn,bhtnd->bhtd", torch.softmax(scores, dim=-1), v_sel)
    return out.to(q.dtype)


if HAS_TRITON:  # pragma: no cover - requires CUDA

    @triton.jit
    def _block_sparse_fwd_kernel(
        Q, K, V, BLOCKS, OUT,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_sb, stride_sh, stride_st, stride_sn,
        stride_ob, stride_oh, stride_ot, stride_od,
        T_K, T_Q,
        scale,
        GROUP: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
        N_BLOCKS: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_h = tl.program_id(1)          # query head
        pid_b = tl.program_id(2)
        pid_kv = pid_h // GROUP           # the KV head this query head reads

        offs_d = tl.arange(0, BLOCK_D)
        q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + pid_q * stride_qt + offs_d * stride_qd
        q = tl.load(q_ptrs)

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        offs_k = tl.arange(0, BLOCK_K)
        for slot in tl.static_range(N_BLOCKS):
            blk = tl.load(BLOCKS + pid_b * stride_sb + pid_kv * stride_sh + pid_q * stride_st + slot * stride_sn)
            ptrs = blk.to(tl.int32) * BLOCK_K + offs_k
            kv_mask = ptrs < T_K
            k_ptrs = K + pid_b * stride_kb + pid_kv * stride_kh + ptrs[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k_blk = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
            s = tl.sum(k_blk.to(tl.float32) * q[None, :], axis=1) * scale
            s = tl.where(kv_mask, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            v_ptrs = V + pid_b * stride_vb + pid_kv * stride_vh + ptrs[:, None] * stride_vt + offs_d[None, :] * stride_vd
            v_blk = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)
            acc = acc * alpha + tl.sum(p[:, None] * v_blk.to(tl.float32), axis=0)
            m_i = m_new

        acc = acc / tl.maximum(l_i, 1e-12)
        out_ptrs = OUT + pid_b * stride_ob + pid_h * stride_oh + pid_q * stride_ot + offs_d * stride_od
        tl.store(out_ptrs, acc.to(OUT.dtype.element_ty))


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    block_size: int,
    backend: str = "auto",
) -> torch.Tensor:
    """Dispatch to the Triton kernel when possible, otherwise the torch reference.

    ``block_ids`` must be ``(B, H_kv, T_q, n_blocks)`` for the kernel path; the
    torch path also accepts the broadcast ``(B, H_kv, n_blocks)`` form.
    """
    if backend in ("auto", "triton") and HAS_TRITON and q.is_cuda and block_ids.dim() == 4:
        b, h_q, t_q, d = q.shape
        h_kv = k.shape[1]
        n_blocks = block_ids.shape[-1]
        out = torch.empty_like(q)
        # one program per (query token, query head, batch); the KV head is derived
        # inside the kernel so that all heads of a group share the same block list
        _block_sparse_fwd_kernel[(t_q, h_q, b)](  # pragma: no cover - GPU only
            q, k, v, block_ids.contiguous(), out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            block_ids.stride(0), block_ids.stride(1), block_ids.stride(2), block_ids.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            k.shape[-2], t_q, 1.0 / (d**0.5),
            GROUP=h_q // h_kv, BLOCK_D=d, BLOCK_K=block_size, N_BLOCKS=n_blocks,
        )
        return out
    return block_sparse_attention_torch(q, k, v, block_ids, block_size=block_size)
