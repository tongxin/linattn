"""MLX-native primitives for the attention variants.

Each function mirrors a torch reference so the two can be compared directly:

===========================  =========================================================
``dense_attention``          ``linattn.core.dense_attention``
``sliding_window_attention`` ``linattn.variants.local.SlidingWindowAttention``
``block_sparse_attention``   ``linattn.kernels.triton.block_sparse.*_torch``
``indexer_topk``             ``linattn.kernels.triton.fused_indexer.indexer_topk_torch``
``chunked_delta_rule``       ``linattn.kernels.triton.delta_chunk.delta_chunk_torch``
===========================  =========================================================

Parity notes:

* MLX's fused SDPA takes ``mask="causal"`` and otherwise a boolean (True = keep) or
  additive mask broadcast against ``[B, N, T_q, T_kv]``.
* MLX has no float64, and its GPU matmul path carries ~1e-3 relative error on this
  device (vs ~1e-6 on MLX's CPU backend -- measured in ``bench/bench_mlx.py``), so
  parity is checked against *float32* torch references with a device-aware
  tolerance (``tests/test_mlx.py``).
* ``mx.take`` is the gather primitive; GQA heads are expanded explicitly because the
  block indices are per query, which fused GQA cannot express.

``python -m linattn.kernels.mlx.ops`` runs a self-test against naive references.
"""

from __future__ import annotations

import math

import numpy as np

from linattn.kernels.mlx import mx

__all__ = [
    "to_mlx",
    "to_torch",
    "causal_mask",
    "dense_attention",
    "sliding_window_attention",
    "expand_kv_heads",
    "block_sparse_attention",
    "indexer_scores",
    "indexer_topk",
    "chunked_delta_rule",
]


def to_mlx(x, dtype=None):
    """torch/numpy -> ``mx.array`` (float32 unless a dtype is given)."""
    if mx is None:
        raise RuntimeError("mlx is not available")
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    arr = mx.array(np.ascontiguousarray(x))
    return arr if dtype is None else arr.astype(dtype)


def to_torch(arr, like=None):
    """``mx.array`` -> torch tensor (via numpy; no graph crosses the boundary)."""
    import torch

    out = torch.from_numpy(np.array(arr))
    return out if like is None else out.to(dtype=like.dtype)


def causal_mask(t_q: int, t_k: int, offset: int = 0):
    """Boolean mask ``(T_q, T_k)``, ``True`` where attention is allowed."""
    qi = mx.arange(t_q) + offset
    kj = mx.arange(t_k)
    return kj[None, :] <= qi[:, None]


def dense_attention(q, k, v, *, causal: bool = True, mask=None, offset: int = 0):
    """Fused SDPA over the full cache.

    MLX's ``mask="causal"`` is **bottom-right aligned**: with ``T_q < T_k`` query
    ``i`` may read keys up to ``i + (T_k - T_q)``.  PyTorch's ``is_causal=True`` (and
    this repository's ``causal_mask``) is top-left aligned, i.e. ``j <= i``.  The
    two agree only when ``T_q == T_k``, so an explicit mask is used whenever the
    shape differs -- which is exactly the incremental-decode case.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    if mask is not None:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    if not causal:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    if q.shape[-2] == k.shape[-2] and offset == 0:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=causal_mask(q.shape[-2], k.shape[-2], offset)
    )
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(q.shape[-1]), mask=mask)


def sliding_window_attention(q, k, v, *, window: int, sinks: int = 0, offset: int = 0):
    """Causal window + leading sink tokens, as an explicit boolean mask."""
    t_q, t_k = q.shape[-2], k.shape[-2]
    qi = mx.arange(t_q) + offset
    kj = mx.arange(t_k)
    allow = (kj[None, :] <= qi[:, None]) & (
        ((qi[:, None] - kj[None, :]) < window) | (kj[None, :] < sinks)
    )
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(q.shape[-1]), mask=allow
    )


def expand_kv_heads(k, n_heads: int):
    """``(B, H_kv, T, D) -> (B, H, T, D)``, repeating each KV head ``group`` times."""
    h_kv = k.shape[1]
    if h_kv == n_heads:
        return k
    if n_heads % h_kv:
        raise ValueError("n_heads must be a multiple of n_kv_heads")
    return mx.repeat(k, n_heads // h_kv, axis=1)


def block_sparse_attention(q, k, v, block_ids, *, block_size: int, offset: int | None = None):
    """Attention over a per-query list of key blocks.

    ``q``: ``(B,H,T_q,D)``; ``k``/``v``: ``(B,H,T_k,D)`` with heads already
    expanded; ``block_ids``: ``(B,H,T_q,n_blocks)``.  One flat ``mx.take`` gathers
    every selected entry, then SDPA runs over the compacted set.

    ``offset`` enables the causal intersection: a selected entry is masked unless
    ``token <= query_position`` (and inside the cache).  Selectors may rank a
    partially-future page highly -- the torch path intersects with the causal mask,
    so the MLX path must too, or the parity test silently compares two different
    computations.
    """
    b, h, t_q, _ = q.shape
    t_k = k.shape[2]
    n_blocks = block_ids.shape[-1]
    offsets = mx.arange(block_size)
    raw = (block_ids[..., None] * block_size + offsets).reshape(b, h, t_q, n_blocks * block_size)
    entry = mx.minimum(raw, t_k - 1).astype(mx.int32)
    base = (mx.arange(b)[:, None, None, None] * h + mx.arange(h)[None, :, None, None]) * t_k
    flat = (base + entry).reshape(-1)
    width = n_blocks * block_size
    k_sel = mx.take(k.reshape(b * h * t_k, -1), flat, axis=0).reshape(b, h, t_q, width, -1)
    v_sel = mx.take(v.reshape(b * h * t_k, -1), flat, axis=0).reshape(b, h, t_q, width, -1)
    # SDPA is rank-4, so each query becomes its own "batch" of one query with its own
    # selected key set: (B*H*T_q, 1, 1, D) x (B*H*T_q, 1, width, D).
    q_flat = q.reshape(b * h * t_q, 1, 1, q.shape[-1])
    k_flat = k_sel.reshape(b * h * t_q, 1, width, k_sel.shape[-1])
    v_flat = v_sel.reshape(b * h * t_q, 1, width, v_sel.shape[-1])
    if offset is None:
        out = mx.fast.scaled_dot_product_attention(
            q_flat, k_flat, v_flat, scale=1.0 / math.sqrt(q.shape[-1])
        )
    else:
        q_pos = mx.arange(t_q) + offset
        keep = (raw <= q_pos[None, None, :, None]) & (raw < t_k)
        # keep: (B,H,T_q,width) -> (B*H*T_q, 1, 1, width)
        keep = keep.reshape(b * h * t_q, 1, 1, width)
        out = mx.fast.scaled_dot_product_attention(
            q_flat, k_flat, v_flat, scale=1.0 / math.sqrt(q.shape[-1]), mask=keep
        )
    return out.reshape(b, h, t_q, q.shape[-1])


def indexer_scores(q_idx, k_idx, head_weights):
    """DSA lightning-indexer scores ``(B, T_q, T_k)``.

    ``I_{t,s} = sum_h w_{t,h} ReLU(q_{t,h} . k_{s,h})`` with ``q_idx``/``k_idx``
    ``(B, T, H_I, D)`` and ``head_weights`` ``(B, T_q, H_I)``.
    """
    dots = mx.einsum("bqhd,bkhd->bqkh", q_idx, k_idx)
    return (mx.maximum(dots, 0.0) * head_weights[:, :, None, :]).sum(axis=-1)


def indexer_topk(q_idx, k_idx, head_weights, k: int):
    """Top-k indexer scores: returns ``(values, indices)`` along the key axis.

    Two MLX details: ``mx.topk`` returns values only (indices come from
    ``argsort``) and in **ascending** order, so the values are flipped to match
    ``torch.topk``.  Tie order is not guaranteed to match torch; compare values (or
    the resulting attention) rather than raw indices.
    """
    scores = indexer_scores(q_idx, k_idx, head_weights)
    k = min(k, scores.shape[-1])
    values = mx.flip(mx.topk(scores, k, axis=-1), axis=-1)
    indices = mx.argsort(-scores, axis=-1)[..., :k]
    return values, indices


def _forward_substitution(m, rhs):
    """Solve ``(I + m) y = rhs`` with ``m`` strictly lower triangular, on GPU.

    MLX 0.32 exposes ``linalg.solve_triangular`` / ``inv`` / ``solve`` for the CPU
    only ("not yet supported on the GPU"), so the chunked delta rule -- which needs
    one small triangular solve per chunk -- would otherwise force a device round
    trip on every layer.  Forward substitution is a loop of ``C`` vectorised steps,
    which stays on the GPU and is exact (it is the same elimination the linalg call
    performs).  ``C`` is 16-64 in practice, so the loop is short.
    """
    c = m.shape[-2]
    rows = []
    for i in range(c):
        if i == 0:
            y_i = rhs[:, :, 0, :]
        else:
            prev = mx.stack(rows, axis=2)                       # (B,H,i,D_V)
            coeff = m[:, :, i, :i]                              # (B,H,i)
            corr = (coeff[:, :, :, None] * prev).sum(axis=2)    # (B,H,D_V)
            y_i = rhs[:, :, i, :] - corr
        rows.append(y_i)
    return mx.stack(rows, axis=2)


def chunked_delta_rule(q, k, v, beta, log_decay, state, *, use_erase: bool = True):
    """Chunkwise delta rule (derivation in ``variants/recurrent.py``).

    Shapes: ``q,k`` ``(B,H,C,D_K)``; ``v`` ``(B,H,C,D_V)``; ``beta`` ``(B,H,C)``;
    ``log_decay`` ``(B,H,C,D_K)``; ``state`` ``(B,H,D_K,D_V)``.
    MLX has ``linalg.solve_triangular``, so the per-chunk solve maps over directly.

    **Invariant:** keys must be L2-normalised (the delta-rule stability condition).
    The chunk matrix is ``M[i,j] = beta_i (e_i . d_j)`` with ``e = k*G`` and
    ``d = k/G``, so with unnormalised keys ``M`` grows like ``||k||^2 / decay`` and
    the triangular solve is exponentially ill-conditioned -- we measured ``|y|``
    reaching 1e24 with ``||k|| ~ 8.5``, at which point two implementations disagree
    by 1e-2 even though both are correct.  Normalised keys keep ``M = O(beta)``.
    """
    b, h, c, dk = k.shape
    gc = mx.cumsum(log_decay, axis=2)
    G = mx.exp(gc)
    d = k / G
    e = k * G
    qt = q * G
    idx = mx.arange(c)
    lower = idx[:, None] > idx[None, :]
    lower0 = idx[:, None] >= idx[None, :]
    read = mx.einsum("bhcd,bhde->bhce", e, state)
    if use_erase:
        m = mx.einsum("bhcd,bhjd->bhcj", e, d) * mx.where(lower, beta[:, :, :, None], 0.0)
        rhs = beta[:, :, :, None] * (v - read)
    else:
        m = mx.zeros((b, h, c, c), dtype=k.dtype)
        rhs = beta[:, :, :, None] * v
    y = _forward_substitution(m, rhs)
    new_state = mx.exp(gc[:, :, -1, :])[:, :, :, None] * (
        state + mx.einsum("bhcd,bhce->bhde", d, y)
    )
    qd = mx.einsum("bhcd,bhjd->bhcj", qt, d) * mx.where(lower0, 1.0, 0.0)
    out = mx.einsum("bhcd,bhde->bhce", qt, state) + mx.einsum("bhcj,bhje->bhce", qd, y)
    return out, new_state


# ---------------------------------------------------------------------------------------
# self-test against naive references
# ---------------------------------------------------------------------------------------
def _selftest() -> int:  # pragma: no cover - manual check
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    failures = []

    def check(name, got, want, tol=6e-3):
        g = np.array(got, dtype=np.float32)
        w = np.array(want, dtype=np.float32)
        err = float(np.abs(g - w).max() / max(1e-6, np.abs(w).max()))
        ok = err < tol
        if not ok:
            failures.append(name)
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<28} rel_err={err:.2e}")

    b, h, t_q, t_k, d = 2, 4, 8, 12, 32
    q = rng.standard_normal((b, h, t_q, d)).astype(np.float32)
    k = rng.standard_normal((b, h, t_k, d)).astype(np.float32)
    v = rng.standard_normal((b, h, t_k, d)).astype(np.float32)
    qt, kt, vt = map(torch.from_numpy, (q, k, v))

    check("dense (causal)", dense_attention(to_mlx(q), to_mlx(k), to_mlx(v)),
          F.scaled_dot_product_attention(qt, kt, vt, is_causal=True))
    from linattn.core import AttnConfig, dense_attention as torch_dense
    from linattn.kernels.triton.block_sparse import block_sparse_attention_torch
    from linattn.kernels.triton.delta_chunk import delta_chunk_torch
    from linattn.kernels.triton.fused_indexer import indexer_topk_torch
    from linattn.variants.local import SlidingWindowAttention

    check("dense (causal, t_q == t_k)", dense_attention(to_mlx(q[:, :, :8]), to_mlx(k[:, :, :8]), to_mlx(v[:, :, :8])),
          torch_dense(qt[:, :, :8], kt[:, :, :8], vt[:, :, :8], causal=True))
    # the incremental-decode case: 4 new queries against 12 cached keys, offset 8
    check("dense (decode offset)", dense_attention(to_mlx(q[:, :, :4]), to_mlx(k), to_mlx(v), offset=8),
          torch_dense(qt[:, :, :4], kt, vt, causal=True))
    swing = SlidingWindowAttention(AttnConfig(n_heads=h, n_kv_heads=h, head_dim=d, window_size=4, sink_size=1))
    check("sliding window",
          sliding_window_attention(to_mlx(q), to_mlx(k), to_mlx(v), window=4, sinks=1),
          swing.forward(qt, kt, vt)[0])

    ids = torch.randint(0, t_k // 4, (b, h, t_q, 2))
    check("block sparse",
          block_sparse_attention(to_mlx(q), to_mlx(k), to_mlx(v), to_mlx(ids), block_size=4),
          block_sparse_attention_torch(qt, kt, vt, ids, block_size=4))

    q_idx = rng.standard_normal((1, 6, 3, 8)).astype(np.float32)
    k_idx = rng.standard_normal((1, 10, 3, 8)).astype(np.float32)
    w = rng.random((1, 6, 3)).astype(np.float32)
    vals, _ = indexer_topk(to_mlx(q_idx), to_mlx(k_idx), to_mlx(w), 4)
    want = indexer_topk_torch(torch.from_numpy(q_idx), torch.from_numpy(k_idx), torch.from_numpy(w), 4)[0]
    check("indexer top-k (values)", vals, want)

    b2, h2, c, dk, dv = 1, 2, 16, 8, 8
    qq = rng.standard_normal((b2, h2, c, dk)).astype(np.float32)
    # L2-normalised keys: the delta-rule invariant (see chunked_delta_rule docstring)
    kk = rng.standard_normal((b2, h2, c, dk))
    kk = (kk / np.linalg.norm(kk, axis=-1, keepdims=True)).astype(np.float32)
    vv = rng.standard_normal((b2, h2, c, dv)).astype(np.float32)
    beta = rng.random((b2, h2, c)).astype(np.float32)
    g = (-rng.random((b2, h2, c, dk)) * 0.2).astype(np.float32)
    st = np.zeros((b2, h2, dk, dv), dtype=np.float32)
    out, new_state = chunked_delta_rule(to_mlx(qq), to_mlx(kk), to_mlx(vv), to_mlx(beta), to_mlx(g), to_mlx(st))
    want_out, want_state = delta_chunk_torch(*(torch.from_numpy(a) for a in (qq, kk, vv, beta, g, st)),
                                             use_erase=True)
    check("delta chunk (out)", out, want_out, tol=1e-4)
    check("delta chunk (state)", new_state, want_state, tol=1e-4)

    print("selftest:", "PASS" if not failures else f"FAIL {failures}")
    return 0 if not failures else 1


def _ref_sliding(q, k, v, window, sinks):
    import torch

    t_q, t_k = q.shape[-2], k.shape[-2]
    qi, kj = torch.arange(t_q)[:, None], torch.arange(t_k)[None, :]
    allow = (kj <= qi) & (((qi - kj) < window) | (kj < sinks))
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    return torch.softmax(scores.masked_fill(~allow, float("-inf")), -1) @ v


def _ref_block_sparse(q, k, v, ids, block_size):
    import torch

    b, h, t_q, _ = q.shape
    t_k = k.shape[2]
    entry = (ids[..., None] * block_size + torch.arange(block_size)).reshape(b, h, t_q, -1).clamp(max=t_k - 1)
    gather = lambda x: torch.gather(x, 2, entry[..., None].expand(*entry.shape, x.shape[-1]))
    return torch.nn.functional.scaled_dot_product_attention(q, gather(k), gather(v))


def _ref_indexer(q_idx, k_idx, w):
    import torch

    relu = torch.relu(torch.einsum("bqhd,bkhd->bqkh", q_idx, k_idx))
    return (relu * w[:, :, None, :]).sum(-1)


def _ref_delta(q, k, v, beta, log_decay, state):
    import torch

    q, k, v = torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v)
    beta, log_decay, state = torch.from_numpy(beta), torch.from_numpy(log_decay), torch.from_numpy(state)
    b, h, c, dk = k.shape
    gc = torch.cumsum(log_decay, 2)
    G = torch.exp(gc)
    d, e, qt = k / G, k * G, q * G
    tril = torch.ones(c, c, dtype=torch.bool).tril(-1)
    tril0 = torch.ones(c, c, dtype=torch.bool).tril(0)
    m = torch.einsum("bhcd,bhjd->bhcj", e, d) * tril * beta[:, :, :, None]
    rhs = beta[:, :, :, None] * (v - torch.einsum("bhcd,bhde->bhce", e, state))
    y = torch.linalg.solve_triangular(torch.eye(c) + m, rhs, upper=False)
    new_state = torch.exp(gc[:, :, -1, :])[:, :, :, None] * (state + torch.einsum("bhcd,bhce->bhde", d, y))
    qd = torch.einsum("bhcd,bhjd->bhcj", qt, d) * tril0
    return (torch.einsum("bhcd,bhde->bhce", qt, state) + torch.einsum("bhcj,bhje->bhce", qd, y)).numpy(), new_state.numpy()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_selftest())
