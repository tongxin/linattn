"""Paged decode attention as a hand-written Metal kernel (MLX custom kernel).

Decode is the case that matters for long-horizon agents: one query, a long cache,
and a selector that hands the kernel a list of pages to read.  The kernel below is
the MLX/Metal equivalent of ``kernels/c/decode_attn.c`` and of the Triton
``block_sparse`` decode path.

Layout
------

* one **SIMD group** (32 lanes) per ``(batch, query head)``;
* lanes split the head dimension (``CH = D/32`` channels each), so every key byte is
  read once and the dot product is a ``simd_sum`` across the group;
* online softmax (running max / running sum) exactly as in FlashAttention, with the
  state kept in registers;
* ``ids`` is the per-query page list produced by Quest / DSA / MoBA / XAttention --
  which is why one kernel serves every selection variant.

Why a custom kernel at all, given MLX ships a fused SDPA?  Two reasons, both
measured in ``reports/mlx_*.md``:

1. SDPA cannot express a *per-query* key list (it is rank-4 and assumes a shared
   key axis), so block-sparse decode otherwise degenerates into a gather that
   materialises the selected blocks;
2. on this Metal device MLX's generic fp32 matmul carries ~8e-4 relative error
   (vs ~1e-6 on MLX's CPU stream), while this kernel's explicit fp32 FMAs agree
   with torch to ~4e-7 -- the numerics of a kernel are a design decision, not a
   property of the device.

Constraints: ``D`` must be a multiple of 32 and ``<= 128``; decode only
(``T_q == 1``).  ``metal_decode_attention`` falls back to the ops path otherwise.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from linattn.kernels.mlx import mx
from linattn.kernels.mlx import ops

__all__ = ["metal_available_for", "metal_decode_attention", "decode_attention_torch", "KERNEL_SOURCE"]

_KERNELS = {}

KERNEL_SOURCE = """
    uint lane = thread_position_in_grid.x;
    uint h = thread_position_in_grid.y;
    uint b = thread_position_in_grid.z;
    constexpr int CH = DIM / 32;
    const int kvh = int(h) / GROUP;

    float m = -INFINITY;
    float l = 0.0f;
    float acc[CH];
    for (int c = 0; c < CH; ++c) { acc[c] = 0.0f; }

    for (uint p = 0; p < uint(NP); ++p) {
        int page = ids[(b * HEADS + h) * NP + p];
        for (int j = 0; j < PS; ++j) {
            int tok = page * PS + j;
            if (tok < 0 || tok >= TK) { continue; }
            float part = 0.0f;
            for (int c = 0; c < CH; ++c) {
                int d = int(lane) * CH + c;
                part += q[(b * HEADS + h) * DIM + d]
                      * k[((b * HKV + kvh) * TK + tok) * DIM + d];
            }
            float dot = simd_sum(part) * scale;
            float m_new = fmax(m, dot);
            float alpha = exp(m - m_new);
            float pw = exp(dot - m_new);
            l = l * alpha + pw;
            for (int c = 0; c < CH; ++c) {
                int d = int(lane) * CH + c;
                acc[c] = acc[c] * alpha
                       + pw * v[((b * HKV + kvh) * TK + tok) * DIM + d];
            }
            m = m_new;
        }
    }

    float denom = (l > 0.0f) ? l : 1.0f;
    for (int c = 0; c < CH; ++c) {
        int d = int(lane) * CH + c;
        out[(b * HEADS + h) * DIM + d] = acc[c] / denom;
    }
"""


def metal_available_for(dim: int) -> bool:
    """The kernel needs ``D % 32 == 0`` and ``D <= 128``, plus a Metal device."""
    from linattn.kernels.mlx import metal_available

    return metal_available() and dim % 32 == 0 and 0 < dim <= 128


def _get_kernel(dim: int, heads: int, hkv: int, npages: int, page: int, tk: int, group: int):
    key = (dim, heads, hkv, npages, page, tk, group)
    if key in _KERNELS:
        return _KERNELS[key]
    kernel = mx.fast.metal_kernel(
        name=f"paged_decode_{dim}_{heads}_{hkv}_{npages}_{page}_{tk}_{group}",
        input_names=["q", "k", "v", "ids", "scale"],
        output_names=["out"],
        source=KERNEL_SOURCE,
    )
    _KERNELS[key] = kernel
    return kernel


def metal_decode_attention(
    q,
    k,
    v,
    page_ids,
    *,
    page_size: int,
    n_kv_heads: int,
) -> Tuple[object, bool]:
    """Decode attention over a page list.  Returns ``(out, used_metal)``.

    ``q``: ``(B, H, 1, D)``; ``k``/``v``: ``(B, H_kv, T_k, D)``;
    ``page_ids``: ``(B, H, n_pages)`` int32.
    """
    b, h, t_q, d = q.shape
    if t_q != 1 or not metal_available_for(d):
        k_exp = ops.expand_kv_heads(k, h)
        v_exp = ops.expand_kv_heads(v, h)
        ids = page_ids[:, :, None, :]
        return ops.block_sparse_attention(q, k_exp, v_exp, ids, block_size=page_size), False

    tk = k.shape[2]
    npages = page_ids.shape[-1]
    group = h // n_kv_heads
    kernel = _get_kernel(d, h, n_kv_heads, npages, page_size, tk, group)
    (out,) = kernel(
        inputs=[q, k, v, page_ids.astype(mx.int32), 1.0 / math.sqrt(d)],
        grid=(32, h, b),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, h, 1, d)],
        output_dtypes=[q.dtype],
        template=[("DIM", int(d)), ("HEADS", int(h)), ("HKV", int(n_kv_heads)),
                  ("NP", int(npages)), ("PS", int(page_size)), ("TK", int(tk)),
                  ("GROUP", int(group))],
    )
    return out, True


def decode_attention_torch(q, k, v, page_ids, *, page_size: int):
    """Reference: gather the selected pages, then plain softmax attention."""
    b, h, _, d = q.shape
    k_exp, v_exp = k, v
    if k.shape[1] != h:
        rep = h // k.shape[1]
        k_exp = k.repeat_interleave(rep, dim=1)
        v_exp = v.repeat_interleave(rep, dim=1)
    entry = (page_ids[..., None] * page_size + torch_arange(page_size)).reshape(b, h, -1)
    entry = entry.clamp(max=k.shape[2] - 1)
    k_sel = k_exp.gather(2, entry[..., None].expand(*entry.shape, d))
    v_sel = v_exp.gather(2, entry[..., None].expand(*entry.shape, d))
    return torch_scaled_dot_product_attention(q, k_sel, v_sel)


def torch_arange(n: int):
    import torch

    return torch.arange(n)


def torch_scaled_dot_product_attention(q, k, v):
    import torch

    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def _selftest() -> int:  # pragma: no cover - manual, GPU only
    import torch

    rng = np.random.default_rng(0)
    b, h, hkv, tk, d, page = 1, 4, 2, 512, 64, 16
    q = rng.standard_normal((b, h, 1, d)).astype(np.float32)
    k = rng.standard_normal((b, hkv, tk, d)).astype(np.float32)
    v = rng.standard_normal((b, hkv, tk, d)).astype(np.float32)
    ids = rng.integers(0, tk // page, (b, h, 8)).astype(np.int32)
    out, used = metal_decode_attention(ops.to_mlx(q), ops.to_mlx(k), ops.to_mlx(v),
                                      ops.to_mlx(ids), page_size=page, n_kv_heads=hkv)
    ref = decode_attention_torch(*(torch.from_numpy(a) for a in (q, k, v)),
                                 torch.from_numpy(ids.astype(np.int64)), page_size=page)
    got = np.array(out)
    rel = float(np.abs(got - ref.numpy()).max() / np.abs(ref.numpy()).max())
    print(f"metal kernel used: {used} | rel_err vs torch: {rel:.2e}")
    return 0 if rel < 1e-4 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_selftest())
