"""What is left of the eviction family: the control baseline.

Score-based KV eviction (H2O, SnapKV, PyramidKV, R-KV, Ada-KV) was retired from the
shipped catalogue after the 2026 result that made the family obsolete: on reasoning
workloads the *score* contributes almost nothing, and uniform random eviction --
provided the prompt and the recent window are kept -- matches the strongest tuned
selector at 32-43% higher throughput (Random Attention, arXiv:2609.03430).  The
argument is in ``docs/survey.md`` §4.2; those papers stay in ``docs/citations.bib``
because they are the evidence for it.

What survives is that *control*: :class:`RandomEvictionBaseline`.  Keeping it means
the claim stays reproducible and any future scored selector has something honest to
beat.  It is deliberately not labelled "eviction" any more -- it is a baseline, not
a method anyone should ship for quality.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from linattn.core import AttentionVariant, causal_mask, dense_attention, repeat_kv

__all__ = ["EvictionAttention", "RandomEvictionBaseline"]


class EvictionAttention(AttentionVariant):
    """Base class: keeps the full cache until it exceeds the budget, then compacts."""

    family = "eviction"

    def __init__(self, cfg=None, **kw):
        super().__init__(cfg, **kw)
        self._reads = 0.0
        self._evictions = 0.0

    def prepare_state(self, batch: int, device, dtype) -> Dict[str, Any]:
        state = super().prepare_state(batch, device, dtype)
        state["score"] = torch.zeros(batch, self.cfg.n_kv_heads, 0, device=device, dtype=torch.float32)
        state["mass"] = None
        state["obs"] = None
        return state

    def push_kv(self, state, k, v):
        state = super().push_kv(state, k, v)
        pad = torch.zeros(k.shape[0], k.shape[1], k.shape[-2], device=k.device, dtype=torch.float32)
        state["score"] = torch.cat([state["score"], pad], dim=-1)
        return state

    # -- attention over whatever is currently in the cache -------------------------
    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        t_q, t_k = q.shape[-2], k.shape[-2]
        allow = causal_mask(t_q, t_k, offset=offset, device=q.device)
        k_rep = repeat_kv(k, self.cfg.group_size)
        scores = torch.matmul(q, k_rep.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        scores = scores.masked_fill(~allow, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, repeat_kv(v, self.cfg.group_size))
        self._reads = float(t_k)
        # bookkeeping used by the subclasses' scores
        state["mass"] = weights.detach().sum(dim=-2)            # (B, H, T_k)
        obs = min(self.cfg.obs_window, t_q)
        state["obs"] = weights.detach()[..., -obs:, :]          # (B, H, obs, T_k)
        return out

    # -- score bookkeeping and compaction -----------------------------------------
    def _group_mass(self, state) -> torch.Tensor:
        mass = state["mass"]  # (B, H, T_k)
        b, h, t = mass.shape
        return mass.view(b, self.cfg.n_kv_heads, self.cfg.group_size, t).sum(2)

    def cache_budget(self) -> int:
        """Retained tokens after compaction (PyramidKV overrides this per layer)."""
        return int(self.cfg.cache_budget)

    def _compaction_keep(self, state, budget: int) -> Optional[torch.Tensor]:
        """Return the indices to keep; ``None`` means "no compaction needed"."""
        raise NotImplementedError

    def finalize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t_k = state["k"].shape[-2]
        budget = self.cache_budget()
        if t_k <= budget:
            return state
        keep = self._compaction_keep(state, budget)
        if keep is None:
            return state
        # keep is (B, H_kv, K_budget) -> gather per batch item
        b = state["k"].shape[0]
        idx = keep.unsqueeze(-1).expand(*keep.shape, self.cfg.head_dim)
        state["k"] = torch.gather(state["k"], -2, idx)
        state["v"] = torch.gather(state["v"], -2, idx)
        state["score"] = torch.gather(state["score"], -1, keep)
        self._evictions += float(t_k - budget)
        return state

    def stats(self) -> Dict[str, float]:
        return {"reads_per_query": self._reads, "evicted_tokens": self._evictions}

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(min(seq_len, self.cache_budget()))

    def flops(self, seq_len: int, batch: int = 1) -> int:
        h, d = self.cfg.n_heads, self.cfg.head_dim
        # prefill is quadratic until the budget is hit, then linear
        budget = min(seq_len, self.cache_budget())
        quad = (seq_len**2 - max(0, seq_len - budget) ** 2) / 2
        lin = max(0, seq_len - budget) * budget
        return int(2 * 2 * batch * h * d * (quad + lin))


class RandomEvictionBaseline(EvictionAttention):
    """Control: evict uniformly at random among the unprotected tokens.

    Implements the "no selection signal" baseline from
    Wang et al., *Random Attention: Rethinking KV Cache Eviction for Efficient
    Reasoning* (2026), https://arxiv.org/abs/2609.03430 -- it keeps the prompt and
    the recent window and randomises everything else.  Included because it is the
    most uncomfortable, and most reproducible, datapoint in the eviction
    literature: on reasoning benchmarks it matches tuned selectors.
    """

    name = "random_eviction"
    family = "baseline"
    paper = ("Wang et al., Random Attention (2026)", "https://arxiv.org/abs/2609.03430")

    def __init__(self, cfg=None, **kw):
        super().__init__(cfg, **kw)
        self._gen = torch.Generator().manual_seed(self.cfg.seed)

    def _compaction_keep(self, state, budget: int) -> torch.Tensor:
        b, h, t_k = state["score"].shape
        n_window = min(self.cfg.window_size, budget)
        n_sink = min(self.cfg.sink_size, budget - n_window)
        n_rand = max(1, budget - n_window - n_sink)
        cand = t_k - n_window - n_sink
        if cand <= 0:
            return torch.arange(t_k, device=state["score"].device).expand(b, h, t_k).contiguous()
        noise = torch.rand(b, h, cand, generator=self._gen, device=state["score"].device)
        order = torch.argsort(noise, dim=-1)[..., :n_rand]
        if n_sink:
            order = torch.cat([torch.arange(n_sink, device=order.device).expand(*order.shape[:-1], n_sink), order], -1)
        recent = torch.arange(t_k - n_window, t_k, device=order.device).expand(*order.shape[:-1], n_window)
        order = torch.cat([order, recent], dim=-1)
        order, _ = torch.sort(order, dim=-1)
        return order
