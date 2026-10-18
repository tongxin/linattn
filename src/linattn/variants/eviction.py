"""The eviction family: old baselines, the 2026 output-aware line, and the control.

Attention-mass eviction (H2O, SnapKV, PyramidKV, R-KV, Ada-KV) was the first
training-free KV family.  The 2026 Random Attention result showed that *that score*
is weak on reasoning traces once the prompt and recent window are protected
(arXiv:2609.03430).  It did not end eviction: DropKV, ReST-KV and BeaconKV replace
the score with output perturbation, reconstruction error or future-query coverage.
Those methods live in :mod:`linattn.variants.frontier`.

This module keeps the old lineage as explicit baselines and
:class:`RandomEvictionBaseline` as the control.  The comparison that matters is not
"eviction yes/no"; it is which selection objective earns its scoring cost.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from linattn.core import AttentionVariant, causal_mask, dense_attention, repeat_kv

__all__ = [
    "EvictionAttention",
    "RandomEvictionBaseline",
    "H2OAttention",
    "SnapKVAttention",
    "TOVAAttention",
]


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
        # Output-aware selectors need the observation-window outputs and queries,
        # not only the attention weights.  Keeping them here avoids duplicating the
        # attention pass in DropKV/ReST-KV and gives BeaconKV its query representatives.
        state["out_obs"] = out.detach()[..., -obs:, :]
        state["q_obs"] = q.detach()[..., -obs:, :]
        return out

    # -- score bookkeeping and compaction -----------------------------------------
    def _group_mass(self, state) -> torch.Tensor:
        mass = state["mass"]  # (B, H, T_k)
        b, h, t = mass.shape
        return mass.view(b, self.cfg.n_kv_heads, self.cfg.group_size, t).sum(2)

    def cache_budget(self) -> int:
        """Retained tokens after compaction (PyramidKV overrides this per layer)."""
        return int(self.cfg.cache_budget)

    def _keep_top(self, score: torch.Tensor, budget: int, t_k: int,
                  n_recent: int, n_sink: int) -> torch.Tensor:
        """Keep ``n_sink`` leading + ``n_recent`` trailing tokens, fill with top scores.

        Shared by every scored selector so the *protected* part is identical and
        only the score differs -- which is exactly the variable the literature argues
        about.
        """
        n_top = max(1, budget - n_recent - n_sink)
        masked = score.clone()
        if n_sink:
            masked[..., :n_sink] = float("inf")
        masked[..., max(0, t_k - n_recent):] = float("inf")
        order = torch.argsort(masked, dim=-1, descending=True)[..., : n_top + n_recent + n_sink]
        order, _ = torch.sort(order, dim=-1)
        return order

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


# --------------------------------------------------------------------------------------
# The attention-mass selectors, kept as academic lineage.
#
# The score barely beats random on reasoning traces once prompt/recent tokens are
# protected (Random Attention, 2026).  They stay here because new output-aware
# selectors need the old baselines: `random_eviction` is the control, and these three
# isolate what the mass score actually contributes.
# --------------------------------------------------------------------------------------
class H2OAttention(EvictionAttention):
    """Heavy-Hitter Oracle: keep tokens by *accumulated* attention mass.

    Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
    Large Language Models* (NeurIPS 2023), https://arxiv.org/abs/2306.14048

    The score is attention mass received so far -- a past statistic used to predict
    future usefulness. This is the canonical example of a heuristic that works on
    RULER and is fragile on reasoning chains.
    """

    name = "h2o"
    family = "eviction"
    paper = ("Zhang et al., H2O: Heavy-Hitter Oracle (NeurIPS 2023)", "https://arxiv.org/abs/2306.14048")

    def _compaction_keep(self, state, budget: int) -> torch.Tensor:
        state["score"] = state["score"] + self._group_mass(state)
        t_k = state["k"].shape[-2]
        n_recent = min(self.cfg.window_size, budget)
        n_sink = min(self.cfg.sink_size, budget - n_recent)
        return self._keep_top(state["score"], budget, t_k, n_recent, n_sink)


class SnapKVAttention(EvictionAttention):
    """SnapKV: score keys once from an observation window of recent queries.

    Li et al., *SnapKV: LLM Knows What You are Looking for Before Generation*
    (NeurIPS 2024), https://arxiv.org/abs/2404.14469

    The observation-window attention is average-pooled along the key axis (kernel 7
    for Llama-3 in the paper); the window itself is always retained so generation can
    continue.
    """

    name = "snapkv"
    family = "eviction"
    paper = ("Li et al., SnapKV (NeurIPS 2024)", "https://arxiv.org/abs/2404.14469")

    def _compaction_keep(self, state, budget: int) -> torch.Tensor:
        obs = state["obs"]                                  # (B,H,obs_len,Tk)
        score = obs.mean(dim=-2)                            # (B,H,Tk)
        kt = score.shape[-1]
        pool = int(self.cfg.extras.get("pool_kernel", 7))
        if pool > 1:
            score = torch.nn.functional.avg_pool1d(
                score, kernel_size=pool, stride=1, padding=pool // 2, count_include_pad=False
            )[..., :kt]
        b, h, _ = score.shape
        score = score.view(b, self.cfg.n_kv_heads, self.cfg.group_size, kt).mean(2)
        n_recent = min(self.cfg.obs_window, budget)
        n_sink = min(self.cfg.sink_size, budget - n_recent)
        return self._keep_top(score, budget, kt, n_recent, n_sink)


class TOVAAttention(EvictionAttention):
    """TOVA: keep the token the *latest* query attends to most, plus recent tokens.

    Oren et al., *Transformers are Multi-State RNNs* (ICLR 2024),
    https://arxiv.org/abs/2401.06104

    TOVA's point is that the query that matters is the current one, not the average
    over history -- a direct rebuttal of H2O's accumulated score.
    """

    name = "tova"
    family = "eviction"
    paper = ("Oren et al., Transformers are Multi-State RNNs (ICLR 2024)", "https://arxiv.org/abs/2401.06104")

    def _compaction_keep(self, state, budget: int) -> torch.Tensor:
        score = state["obs"][..., -1, :]                     # last query's attention
        b, h, kt = score.shape
        score = score.view(b, self.cfg.n_kv_heads, self.cfg.group_size, kt).mean(2)
        n_recent = min(self.cfg.window_size, budget)
        n_sink = min(self.cfg.sink_size, budget - n_recent)
        return self._keep_top(score, budget, kt, n_recent, n_sink)
