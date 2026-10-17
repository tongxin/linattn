"""MLX-native selection + execution for the variants that survive the 2026 rethink.

Implemented: ``dense``, ``sliding_window``, ``quest`` (page bounds + top-k pages)
and ``dsa`` (indexer over *compressed pages*, i.e. the CSA form DeepSeek-V4 ships).
Together they cover the three mechanisms the survey argues are still load-bearing:
exact attention, structural locality, and query-aware page selection.

Each call takes ``(q, k, v)`` as ``mx.array`` (``q`` may be a decode step with
``T_q == 1``) and returns ``(out, stats)``, where ``stats`` reports KV entries read
per query -- the metric that decides decode speed on a memory-bound kernel.  Decode
with a page list routes through the custom Metal kernel
(``kernels/mlx/metal_decode.py``).
"""

from __future__ import annotations

from linattn.kernels.mlx import mx, ops
from linattn.kernels.mlx.metal_decode import metal_decode_attention

__all__ = ["mlx_forward", "SUPPORTED"]

SUPPORTED = ("dense", "sliding_window", "quest", "dsa")


def mlx_forward(
    name: str,
    q,
    k,
    v,
    *,
    window: int = 256,
    sinks: int = 0,
    page_size: int = 16,
    topk_pages: int = 8,
    topk_tokens: int = 512,
    indexer_dim: int = 16,
    n_indexer_heads: int = 4,
    compress_ratio: int = 1,
    seed: int = 0,
    use_metal: bool = True,
    indexer_proj=None,
):
    """Run one MLX variant.  Returns ``(out, stats)``."""
    if name not in SUPPORTED:
        raise KeyError(f"mlx backend supports {SUPPORTED}, not {name!r}")
    b, h, t_q, d = q.shape
    t_k = k.shape[2]
    n_kv = k.shape[1]
    stats = {"reads_per_query": float(t_k), "selected_pages": 0.0, "metal": 0.0}

    if name == "dense":
        return ops.dense_attention(q, k, v, causal=t_q != 1, offset=t_k - t_q), stats

    if name == "sliding_window":
        out = ops.sliding_window_attention(q, k, v, window=window, sinks=sinks, offset=t_k - t_q)
        # report the *actual* mask count, so the number is comparable with the torch
        # variant's accounting (early queries see fewer keys)
        q_pos = mx.arange(t_q) + (t_k - t_q)
        kj = mx.arange(t_k)
        allow = (kj[None, :] <= q_pos[:, None]) & (
            ((q_pos[:, None] - kj[None, :]) < window) | (kj[None, :] < sinks)
        )
        stats["reads_per_query"] = float(allow.sum(axis=-1).mean())
        return out, stats

    if name == "quest":
        page_ids = _quest_pages(q, k, page_size=page_size, topk_pages=topk_pages)
    else:
        page_ids = _csa_pages(
            q, k, page_size=page_size, topk_tokens=topk_tokens, indexer_dim=indexer_dim,
            n_indexer_heads=n_indexer_heads, compress_ratio=compress_ratio, seed=seed,
            proj=indexer_proj,
        )

    width = page_ids.shape[-1] * page_size
    raw = (page_ids[..., None] * page_size + mx.arange(page_size)).reshape(b, h, t_q, width)
    q_pos = mx.arange(t_q) + (t_k - t_q)
    effective = (raw <= q_pos[None, None, :, None]) & (raw < t_k)
    stats["reads_per_query"] = float(effective.sum(-1).mean())
    stats["selected_pages"] = float(page_ids.shape[-1])
    if t_q == 1 and use_metal:
        out, used = metal_decode_attention(q, k, v, page_ids, page_size=page_size, n_kv_heads=n_kv)
        stats["metal"] = 1.0 if used else 0.0
        return out, stats

    k_exp = ops.expand_kv_heads(k, h)
    v_exp = ops.expand_kv_heads(v, h)
    return ops.block_sparse_attention(
        q, k_exp, v_exp, page_ids, block_size=page_size, offset=t_k - t_q
    ), stats


def _pool_pages(k, page_size: int):
    """Mean-pool keys into pages: ``(B,Hkv,T,D) -> (B,Hkv,P,D)`` plus ``P``."""
    b, h, t_k, d = k.shape
    n_pages = (t_k + page_size - 1) // page_size
    pad = n_pages * page_size - t_k
    if pad:
        k = mx.pad(k, [(0, 0), (0, 0), (0, pad), (0, 0)])
    return k.reshape(b, h, n_pages, page_size, d).mean(axis=3), n_pages


def _quest_pages(q, k, *, page_size: int, topk_pages: int):
    """Quest: rank pages by ``sum_d max(q_d m_d, q_d M_d)`` over channel bounds."""
    b, h, t_q, d = q.shape
    t_k = k.shape[2]
    n_kv = k.shape[1]
    n_pages = (t_k + page_size - 1) // page_size
    pad = n_pages * page_size - t_k
    k_pad = mx.pad(k, [(0, 0), (0, 0), (0, pad), (0, 0)]) if pad else k
    pages = k_pad.reshape(b, n_kv, n_pages, page_size, d)
    lo, hi = pages.min(axis=3), pages.max(axis=3)
    group = h // n_kv
    if group > 1:
        lo, hi = mx.repeat(lo, group, axis=1), mx.repeat(hi, group, axis=1)
    qe = q[:, :, :, None, :]
    scores = mx.maximum(qe * lo[:, :, None], qe * hi[:, :, None]).sum(-1)   # (B,H,Tq,P)
    q_pos = mx.arange(t_q) + (t_k - t_q)
    visible = (mx.arange(n_pages) * page_size)[None, :] <= q_pos[:, None]
    scores = mx.where(visible[None, None], scores, -1e30)
    # the query's own page is always read, and it counts towards the budget (same
    # rule as the torch Quest implementation: set it to +inf, then take top-k)
    own = mx.minimum(q_pos // page_size, n_pages - 1)
    own_mask = mx.arange(n_pages)[None, None, None, :] == own[None, None, :, None]
    scores = mx.where(own_mask, 1e30, scores)
    keep = max(1, min(topk_pages, n_pages))
    ids = mx.argsort(-scores, axis=-1)[..., :keep].astype(mx.int32)
    return ids


def _csa_pages(q, k, *, page_size, topk_tokens, indexer_dim, n_indexer_heads, compress_ratio, seed, proj=None):
    """Compressed sparse attention: index *pooled pages*, then read the top pages.

    Page-level (CSA) rather than token-level (DSA) indexing: it yields a fixed-width
    page list, which is what the Metal kernel consumes, and it is the compressed form
    DeepSeek-V4 ships.  Token-level selection is covered by the torch DSA path.

    The indexer weights are random here: this validates the selection + execution
    pipeline on Metal, not model quality.  Fitting one needs calibration data and
    gradient steps (see ``variants/selection.py``).
    """
    b, h, t_q, d = q.shape
    t_k = k.shape[2]
    n_kv = k.shape[1]
    group = h // n_kv
    pool = max(page_size, compress_ratio)
    k_page, n_pages = _pool_pages(k, pool)
    if group > 1:
        k_page = mx.repeat(k_page, group, axis=1)

    if proj is None:
        proj = mx.random.normal((h, n_indexer_heads, d, indexer_dim), key=mx.random.key(seed)) * 0.02
    else:
        proj = proj if isinstance(proj, type(mx.array(0))) else mx.array(proj)
    w = mx.softmax(mx.ones((n_indexer_heads,)), axis=-1)
    qi = mx.einsum("bhtd,hedf->bhtef", q, proj)               # (B,H,Tq,E,F)
    ki = mx.einsum("bhpd,hedf->bhpef", k_page, proj)           # (B,H,P,E,F)
    relu = mx.maximum(mx.einsum("bhtef,bhpef->bhtpe", qi, ki), 0.0)
    scores = (relu * w[None, None, None, None, :]).sum(-1)    # (B,H,Tq,P)

    q_pos = mx.arange(t_q) + (t_k - t_q)
    visible = (mx.arange(n_pages) * pool)[None, :] <= q_pos[:, None]
    scores = mx.where(visible[None, None], scores, -1e30)
    budget_pages = max(1, topk_tokens // pool) if topk_tokens else n_pages
    keep = max(1, min(budget_pages, n_pages))
    ids = mx.argsort(-scores, axis=-1)[..., :keep].astype(mx.int32)
    own = mx.broadcast_to(mx.minimum(q_pos // pool, n_pages - 1).astype(mx.int32)[None, None, :, None],
                          (b, h, t_q, 1))
    return mx.concatenate([ids, own], axis=-1)
