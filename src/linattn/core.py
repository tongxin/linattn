"""Shared primitives for every attention variant.

Design notes
------------
All variants implement the same *streaming* contract::

    out, state = variant.forward(q, k, v, state=None)

``forward`` walks the sequence in query chunks (``chunk`` tokens at a time) and
delegates to :meth:`AttentionVariant.attend`, which is the only thing a variant
has to implement.  The chunked driver is what makes the cache semantics
identical across variants: a variant that keeps the whole KV cache, one that
evicts, one that only keeps a recurrent state and one that reads a learned
selection of blocks all return the same tensor shape and all support
``T_q == 1`` decoding without special-casing.

Everything is float32/float64 friendly and runs on CPU so that the reference
implementations are usable as oracles on any machine; the accelerated backends
in :mod:`linattn.kernels` are validated against these oracles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "AttnConfig",
    "AttentionVariant",
    "dense_attention",
    "DenseAttention",
    "causal_mask",
    "repeat_kv",
    "block_reduce",
    "block_expand",
    "counts_to_shapes",
]


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass
class AttnConfig:
    """Shape and policy knobs shared by all variants.

    Keeping this in one dataclass means a variant can be swapped for another in
    the harness with a single ``replace(cfg, ...)`` call, which is how the
    comparison tables in ``docs/`` are produced.
    """

    n_heads: int = 4
    n_kv_heads: int = 4
    head_dim: int = 64
    block_size: int = 64          # KV block/page granularity for sparse variants
    window_size: int = 256        # sliding-window / local branch size
    cache_budget: int = 512       # tokens retained by eviction variants
    topk_blocks: int = 4          # blocks selected by block-sparse variants
    topk_tokens: int = 256        # tokens selected by token-level variants (DSA)
    compress_ratio: int = 16      # pooling stride for compressed branches (NSA/CSA)
    obs_window: int = 64          # observation window for windowed selection
    sink_size: int = 4            # attention-sink tokens always kept
    stride: int = 8               # partial-stride scoring for XAttention
    threshold: float = 0.9        # block-selection threshold (mass fraction)
    dtype: torch.dtype = torch.float32
    seed: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def group_size(self) -> int:
        return max(1, self.n_heads // self.n_kv_heads)

    def with_(self, **kw: Any) -> "AttnConfig":
        return replace(self, **kw)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``(B, H_kv, T, D)`` to ``(B, H_kv * n_rep, T, D)`` (GQA)."""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def causal_mask(q_len: int, kv_len: int, offset: int = 0, device=None) -> torch.Tensor:
    """Boolean mask ``(q_len, kv_len)`` that is ``True`` where attention is allowed.

    ``offset`` is the absolute position of the first query: query ``i`` attends to
    keys ``<= offset + i``.
    """
    qi = torch.arange(q_len, device=device).unsqueeze(1) + offset
    kj = torch.arange(kv_len, device=device).unsqueeze(0)
    return kj <= qi


def block_reduce(x: torch.Tensor, block: int, dim: int = -2, mode: str = "mean") -> torch.Tensor:
    """Mean-pool ``x`` along ``dim`` in non-overlapping blocks of size ``block``.

    The tail is padded by repeating the last row, so the number of output blocks is
    always ``ceil(len / block)``.  Using one helper for every block-pooling site in
    the codebase keeps block indices consistent between the selection masks (which
    are expanded back to tokens with ``repeat_interleave``) and the pooled scores.
    """
    if block == 1:
        return x
    x = x.movedim(dim, -2)
    t = x.shape[-2]
    pad = (-t) % block
    if pad:
        x = torch.cat([x, x[..., -1:, :].expand(*x.shape[:-2], pad, x.shape[-1])], dim=-2)
    x = x.reshape(*x.shape[:-2], x.shape[-2] // block, block, x.shape[-1])
    x = x.mean(dim=-2) if mode == "mean" else x.sum(dim=-2)
    return x.movedim(-2, dim)


def block_expand(x: torch.Tensor, block: int, dim: int = -2) -> torch.Tensor:
    """Repeat each entry of ``x`` ``block`` times along ``dim`` (inverse of block_reduce)."""
    if block == 1:
        return x
    x = x.movedim(dim, -2)
    x = x.unsqueeze(-2).expand(*x.shape[:-1], block, x.shape[-1])
    x = x.reshape(*x.shape[:-3], x.shape[-3] * block, x.shape[-1])
    return x.movedim(-2, dim)


def counts_to_shapes(counts: torch.Tensor, dim: int, keepdims: bool = False) -> torch.Tensor:
    """Return 1 where ``counts`` is non-zero along ``dim`` (used by IO accounting)."""
    out = (counts > 0).to(torch.int64)
    return out if keepdims else out.squeeze(dim)


def _softmax_with_mask(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1)


# --------------------------------------------------------------------------------------
# reference attention
# --------------------------------------------------------------------------------------
def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
    return_weights: bool = False,
):
    """Plain softmax attention, the oracle every variant is validated against.

    ``q`` has shape ``(B, H, T_q, D)``, ``k``/``v`` have shape ``(B, H_kv, T_kv, D)``.
    """
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    k_rep = repeat_kv(k, q.shape[1] // k.shape[1])
    v_rep = repeat_kv(v, q.shape[1] // v.shape[1])
    scores = torch.matmul(q, k_rep.transpose(-1, -2)) * scale
    if mask is not None:
        allow = mask
    elif causal and q.shape[-2] != k.shape[-2]:
        # asymmetric shapes only happen in the tests; assume offset semantics handled by caller
        allow = causal_mask(q.shape[-2], k.shape[-2], offset=k.shape[-2] - q.shape[-2], device=q.device)
    elif causal:
        allow = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device).tril()
    else:
        allow = None
    if allow is not None and not allow.all():
        scores = scores.masked_fill(~allow, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v_rep)
    if return_weights:
        return out, weights
    return out


# --------------------------------------------------------------------------------------
# the variant interface
# --------------------------------------------------------------------------------------
class AttentionVariant:
    """Base class implementing the chunked streaming driver.

    Subclasses implement :meth:`attend`, :meth:`memory_bytes` and :meth:`flops`.
    Optionally they implement :meth:`prepare_state`, :meth:`push_kv` and
    :meth:`finalize` to control how the cache evolves as chunks arrive.
    """

    name: str = "base"
    family: str = "dense"          # dense | local | eviction | selection | native | recurrent
    trained: bool = False          # whether the variant normally needs training
    paper: tuple = ()              # (title, url) citations, surfaced in --help style output
    storage_precision: str = "fp16 KV"  # persistent cache precision used by memory_bytes

    def __init__(self, cfg: Optional[AttnConfig] = None, **kwargs: Any) -> None:
        self.cfg = cfg or AttnConfig()
        for key, value in kwargs.items():
            if not hasattr(self.cfg, key):
                raise TypeError(f"{type(self).__name__}: unknown config field {key!r}")
            setattr(self.cfg, key, value)
        if self.cfg.n_heads % self.cfg.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")

    # -- state management ---------------------------------------------------------
    def prepare_state(self, batch: int, device, dtype) -> Dict[str, Any]:
        """Initial cache/state before any token has been seen."""
        return {"k": torch.zeros(batch, self.cfg.n_kv_heads, 0, self.cfg.head_dim, device=device, dtype=dtype),
                "v": torch.zeros(batch, self.cfg.n_kv_heads, 0, self.cfg.head_dim, device=device, dtype=dtype),
                "pos": 0}

    def push_kv(self, state: Dict[str, Any], k: torch.Tensor, v: torch.Tensor) -> Dict[str, Any]:
        """Append a new chunk of keys/values to the cache (default: unbounded)."""
        state["k"] = torch.cat([state["k"], k], dim=-2)
        state["v"] = torch.cat([state["v"], v], dim=-2)
        return state

    def attend(self, q: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def finalize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called after each chunk; used by eviction variants to compact."""
        return state

    # -- driver -------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        state: Optional[Dict[str, Any]] = None,
        chunk: Optional[int] = None,
        causal: bool = True,
        collect_stats: bool = False,
    ):
        """Chunked causal attention with explicit cache semantics.

        Returns ``(out, state)``.  ``out`` has the same shape as ``q``.  Passing a
        state returned by a previous call performs incremental decoding, which is
        what the cache-equivalence tests exercise.

        Batching: every variant is written with an explicit batch dimension, so
        ``B > 1`` is supported and each item is independent (``tests/test_batching.py``
        checks that a batched call equals one call per item).  This is *equal-length*
        batching only -- there is no ``lengths``/padding argument, so ragged or
        continuously-batched requests need one call per sequence length (or an explicit
        mask); see ``docs/reproducing.md``.
        """
        if q.dim() != 4:
            raise ValueError("q must have shape (B, H, T, D)")
        b, h, t, d = q.shape
        if state is None:
            state = self.prepare_state(b, q.device, q.dtype)
        chunk = chunk or max(1, self.default_chunk())
        outs = []
        pos = state.get("pos", 0)
        stats = [] if collect_stats else None
        self._collect = collect_stats
        self._masks = []
        for start in range(0, t, chunk):
            stop = min(start + chunk, t)
            qc, kc, vc = q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop]
            state = self.push_kv(state, kc, vc)
            out = self.attend(qc, state, offset=pos + start)
            state = self.finalize(state)
            outs.append(out)
            if stats is not None:
                stats.append(self.stats())
        state["pos"] = pos + t
        out = torch.cat(outs, dim=-2)
        if collect_stats:
            return out, state, stats
        return out, state

    def default_chunk(self) -> int:
        return 256

    # -- introspection ------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        """Per-call diagnostics; variants override to report their real budget use."""
        return {}

    def memory_bytes(self, seq_len: int, batch: int = 1) -> int:
        """Analytic persistent memory (bytes) of the cache/state at ``seq_len``."""
        elems = batch * self.cfg.n_kv_heads * seq_len * self.cfg.head_dim * 2
        return elems * 2  # fp16 storage convention

    def flops(self, seq_len: int, batch: int = 1) -> int:
        """Analytic attention FLOPs for prefill of ``seq_len`` tokens (2*matmul convention)."""
        h, d = self.cfg.n_heads, self.cfg.head_dim
        return int(2 * 2 * batch * h * seq_len * seq_len * d)

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        """KV entries read per query, the quantity that drives decode memory traffic."""
        return float(seq_len)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        c = self.cfg
        return (f"{type(self).__name__}(heads={c.n_heads}, kv={c.n_kv_heads}, d={c.head_dim}, "
                f"block={c.block_size}, budget={c.cache_budget}, topk_blocks={c.topk_blocks})")


class DenseAttention(AttentionVariant):
    """Full softmax attention: the oracle, and the reference point for cost models."""

    name = "dense"
    family = "dense"
    paper = ("Vaswani et al., Attention Is All You Need (2017)", "https://arxiv.org/abs/1706.03762")

    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        t_q, t_k = q.shape[-2], k.shape[-2]
        # keys live at absolute positions 0..t_k-1, queries at offset..offset+t_q-1
        allow = causal_mask(t_q, t_k, offset=offset, device=q.device)
        return dense_attention(q, k, v, causal=False, mask=allow)

    def compute_bytes(self, seq_len: int, batch: int = 1) -> float:  # pragma: no cover - doc helper
        return float(4 * batch * self.cfg.n_heads * seq_len * seq_len * 2)

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(seq_len)


dense_attention_impl = dense_attention
