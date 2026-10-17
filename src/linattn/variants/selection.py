"""Query-aware sparse selection: Quest, MInference, FlexPrefill, XAttention,
MoBA, DeepSeek Sparse Attention (DSA/CSA) and SeerAttention-R.

Every variant in this module keeps the KV cache intact and instead decides *which
KV entries to read*.  That is the family that has reached production inference
stacks, because the selection decision can be made from a cheap proxy score and
the expensive part (the core attention) can stay a standard block-sparse kernel.

The implementations below are the *semantics*: they build a boolean read-mask and
evaluate exact attention under it, so that the error against dense attention is
attributable to the selection rather than to a numerical approximation.  The
accelerated counterparts live in :mod:`linattn.kernels` and are checked against
these classes.

Notation: ``block_size`` is the KV block (page) granularity, ``topk_blocks`` the
budget in blocks, ``topk_tokens`` the budget in tokens for token-level selectors.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from linattn.core import AttnConfig, AttentionVariant, block_reduce, causal_mask, dense_attention, repeat_kv

__all__ = [
    "BlockSelectionAttention",
    "QuestAttention",
    "MInferenceAttention",
    "FlexPrefillAttention",
    "XAttention",
    "MoBAAttention",
    "DSAAttention",
    "SeerAttentionR",
    "block_mask_to_token_mask",
    "topk_blocks",
]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def block_mask_to_token_mask(sel: torch.Tensor, block_q: int, block_k: int, t_q: int, t_k: int) -> torch.Tensor:
    """Expand a block-level boolean mask ``(B,H,n_qb,n_kb)`` to token level."""
    mask = sel.repeat_interleave(block_q, dim=-2).repeat_interleave(block_k, dim=-1)
    return mask[..., :t_q, :t_k]


def topk_blocks(scores: torch.Tensor, k: int, allowed: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Boolean mask of the ``k`` highest-scoring blocks (last dim), respecting ``allowed``."""
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float("-inf"))
    k = max(1, min(k, scores.shape[-1]))
    idx = torch.topk(scores, k, dim=-1).indices
    out = torch.zeros_like(scores, dtype=torch.bool)
    out.scatter_(-1, idx, True)
    return out


def _cumulative_select(probs: torch.Tensor, alpha: float) -> torch.Tensor:
    """Minimal set of entries whose descending-probability cumsum reaches ``alpha``."""
    order = torch.argsort(probs, dim=-1, descending=True)
    sorted_probs = torch.gather(probs, -1, order)
    n_keep = (sorted_probs.cumsum(-1) < alpha).sum(-1) + 1
    n_keep = n_keep.clamp(min=1, max=probs.shape[-1])
    rank = torch.argsort(order, dim=-1)
    return rank < n_keep.unsqueeze(-1)


class BlockSelectionAttention(AttentionVariant):
    """Shared plumbing for variants that read a subset of the KV cache."""

    family = "selection"

    def __init__(self, cfg: Optional[AttnConfig] = None, **kw):
        super().__init__(cfg, **kw)
        self._reads = 0.0
        self._extra_stats: Dict[str, float] = {}

    def select(self, q: torch.Tensor, state: Dict[str, Any], offset: int) -> torch.Tensor:
        """Return a token-level boolean read mask of shape ``(B, H, T_q, T_k)``."""
        raise NotImplementedError

    def effective_mask(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        """The mask actually read: the selection intersected with causality.

        Exposed (rather than buried in :meth:`attend`) because it is the object the
        validation suite checks: a selector may propose a future block or a
        partially-future page, and the invariant that has to hold is that nothing
        future is ever *read*.  ``tests/test_selection.py`` verifies both that this
        mask is causal and that the variant's output equals exact attention over
        it -- i.e. that the variant reads exactly what it says it reads.
        """
        k, v = state["k"], state["v"]
        allow = self.select(q, state, offset)
        causal = causal_mask(q.shape[-2], k.shape[-2], offset=offset, device=q.device)
        allow = allow & causal
        # a query whose selection is empty (or entirely future) still reads itself
        empty = ~allow.any(-1)
        if bool(empty.any()):
            diag = torch.arange(q.shape[-2], device=q.device).unsqueeze(-1) + offset
            fallback = torch.zeros_like(allow)
            diag_idx = diag.view(1, 1, -1, 1).expand(*allow.shape[:-1], -1).clamp(max=k.shape[-2] - 1)
            fallback.scatter_(-1, diag_idx, True)
            allow = allow | (fallback & empty.unsqueeze(-1))
        return allow

    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        allow = self.effective_mask(q, state, offset)
        if getattr(self, "_collect", False):
            self._masks.append(allow.detach())
        self._reads = float(allow.sum(-1).float().mean())
        return dense_attention(q, k, v, causal=False, mask=allow)

    def stats(self) -> Dict[str, float]:
        out = {"reads_per_query": self._reads}
        out.update(self._extra_stats)
        return out

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(min(seq_len, self.cfg.topk_blocks * self.cfg.block_size + self.cfg.window_size))

    def selection_flops(self, seq_len: int, batch: int = 1) -> int:
        """FLOPs spent deciding what to read: reported, never hidden."""
        return 0

    def flops(self, seq_len: int, batch: int = 1) -> int:
        h, d = self.cfg.n_heads, self.cfg.head_dim
        reads = self.read_budget(seq_len, batch)
        return int(2 * 2 * batch * h * seq_len * reads * d + self.selection_flops(seq_len, batch))


# --------------------------------------------------------------------------------------
# Quest
# --------------------------------------------------------------------------------------
class QuestAttention(BlockSelectionAttention):
    """Quest: page-level selection using channel-wise key bounds.

    For every page the cache stores the channel-wise minimum ``m`` and maximum
    ``M`` of its keys.  Because ``max(q_d*m_d, q_d*M_d)`` upper-bounds ``q_d*k_d``
    for every key in the page, ``sum_d max(q_d m_d, q_d M_d)`` upper-bounds the
    page's attention logit.  Selecting the pages with the largest upper bound is a
    query-aware approximation of selecting the pages that contain the
    highest-attention tokens.

    Tang et al., *Quest: Query-Aware Sparsity for Efficient Long-Context LLM
    Inference* (ICML 2024), https://arxiv.org/abs/2406.10774

    The paper's own caveats are reproduced: the first two layers are kept dense
    (their sparsity is low), and the selection is per query head, so under GQA a
    group must read the *union* of its heads' pages -- the token budget inflates
    by up to the group size.  ``extras["gqa_union"] = False`` switches to a
    max-pooled shared selection, which is the fair-comparison setting.
    """

    name = "quest"
    family = "selection"
    paper = ("Tang et al., Quest (ICML 2024)", "https://arxiv.org/abs/2406.10774")

    def __init__(self, cfg=None, **kw):
        super().__init__(cfg, **kw)
        self.dense_layers = int(self.cfg.extras.get("dense_layers", 2))
        self.layer_idx = int(self.cfg.extras.get("layer_idx", 2))
        self.page_size = int(self.cfg.extras.get("page_size", self.cfg.block_size))

    def prepare_state(self, batch, device, dtype) -> Dict[str, Any]:
        state = super().prepare_state(batch, device, dtype)
        state["page_min"] = torch.zeros(batch, self.cfg.n_kv_heads, 0, self.cfg.head_dim, device=device, dtype=dtype)
        state["page_max"] = torch.zeros(batch, self.cfg.n_kv_heads, 0, self.cfg.head_dim, device=device, dtype=dtype)
        return state

    def push_kv(self, state, k, v):
        start = state["k"].shape[-2]
        state = super().push_kv(state, k, v)
        self._refresh_pages(state, start)
        return state

    def _refresh_pages(self, state, from_pos: int) -> None:
        """Recompute page min/max only for pages touched by the new keys."""
        bs = self.page_size
        k = state["k"]
        t = k.shape[-2]
        first_page = max(0, from_pos // bs)
        n_pages = (t + bs - 1) // bs
        if n_pages <= first_page:
            return
        mins, maxs = [], []
        for page in range(first_page, n_pages):
            chunk = k[..., page * bs : min(t, (page + 1) * bs), :]
            mins.append(chunk.amin(dim=-2))
            maxs.append(chunk.amax(dim=-2))
        new_min = torch.stack(mins, dim=-2)
        new_max = torch.stack(maxs, dim=-2)
        if first_page == 0:
            state["page_min"], state["page_max"] = new_min, new_max
        else:
            state["page_min"] = torch.cat([state["page_min"][..., :first_page, :], new_min], dim=-2)
            state["page_max"] = torch.cat([state["page_max"][..., :first_page, :], new_max], dim=-2)

    def page_scores(self, q: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        """Upper-bound page scores, shape ``(B, H, T_q, n_pages)``."""
        # page metadata lives on the KV heads; under GQA each query head reads the
        # metadata of its own group's KV head
        k_min = repeat_kv(state["page_min"], self.cfg.group_size)        # (B,H,P,D)
        k_max = repeat_kv(state["page_max"], self.cfg.group_size)
        qe = q.unsqueeze(-2)                                             # (B,H,Tq,1,D)
        lo = qe * k_min.unsqueeze(2)
        hi = qe * k_max.unsqueeze(2)
        return torch.maximum(lo, hi).sum(-1)

    def page_selection(self, q: torch.Tensor, state: Dict[str, Any], offset: int) -> torch.Tensor:
        """Block-level boolean selection ``(B, H, T_q, n_pages)``."""
        t_q = q.shape[-2]
        bs = self.page_size
        scores = self.page_scores(q, state)
        n_pages = scores.shape[-1]
        page_start = torch.arange(n_pages, device=q.device) * bs
        q_pos = torch.arange(t_q, device=q.device) + offset
        visible = page_start.unsqueeze(0) <= q_pos.unsqueeze(1)
        scores = scores.masked_fill(~visible, float("-inf"))
        own_page = (q_pos // bs).clamp(max=n_pages - 1)
        # A query always reads its own page.  Note the shape: `scores[..., own_page]`
        # would index the last dim with a (T_q,) tensor and produce a (., T_q, T_q)
        # block, writing `inf` into *every* row for every query's own page -- which
        # silently collapsed the selection to pages 0..k.  Scatter on the diagonal.
        own_idx = own_page.view(1, 1, -1, 1).expand(q.shape[0], q.shape[1], t_q, 1)
        scores = scores.scatter(-1, own_idx, float("inf"))
        sel = topk_blocks(scores, self.cfg.topk_blocks)
        if self.cfg.group_size > 1 and self.cfg.extras.get("gqa_union", True):
            b, h, t, p = sel.shape
            grouped = sel.view(b, self.cfg.n_kv_heads, self.cfg.group_size, t, p).any(dim=2)
            sel = grouped.unsqueeze(2).expand(b, self.cfg.n_kv_heads, self.cfg.group_size, t, p).reshape(b, h, t, p)
        self._extra_stats = {"pages_per_query": float(sel.sum(-1).float().mean())}
        return sel

    def select(self, q: torch.Tensor, state: Dict[str, Any], offset: int) -> torch.Tensor:
        t_q, t_k = q.shape[-2], state["k"].shape[-2]
        if self.layer_idx < self.dense_layers:
            return torch.ones(q.shape[0], q.shape[1], t_q, t_k, dtype=torch.bool, device=q.device)
        sel = self.page_selection(q, state, offset)
        return block_mask_to_token_mask(sel, 1, self.page_size, t_q, t_k)

    def selection_flops(self, seq_len: int, batch: int = 1) -> int:
        n_pages = (seq_len + self.page_size - 1) // self.page_size
        return int(2 * batch * self.cfg.n_heads * seq_len * n_pages * self.cfg.head_dim)


# --------------------------------------------------------------------------------------
# MInference 1.0
# --------------------------------------------------------------------------------------
class MInferenceAttention(BlockSelectionAttention):
    """Pattern-based prefill sparsity: A-shape, Vertical-Slash, Block-Sparse heads.

    Jiang et al., *MInference 1.0: Accelerating Pre-filling for Long-Context LLMs
    via Dynamic Sparse Attention* (NeurIPS 2024), https://arxiv.org/abs/2407.02490

    Faithful points:

    * the Vertical-Slash estimator uses the **last 64 query rows** to locate the
      important vertical lines (columns) and slash lines (diagonals);
    * the Block-Sparse estimator mean-pools q and k into 64-token blocks and
      scores blocks with the pooled inner product -- equivalent to mean-pooling
      the attention matrix, because pooling and matmul commute;
    * the A-shape head is the static "initial tokens + local window" pattern;
    * ``pattern="auto"`` performs the kernel-aware search online, per head,
      picking whichever pattern maximises attention-mass recall at equal read
      budget.  The paper does this offline once per model; doing it per call keeps
      the trade-off measurable instead of assumed.
    """

    name = "minference"
    family = "selection"
    paper = ("Jiang et al., MInference 1.0 (NeurIPS 2024)", "https://arxiv.org/abs/2407.02490")

    def __init__(self, cfg=None, **kw):
        super().__init__(cfg, **kw)
        self.pattern = self.cfg.extras.get("pattern", "auto")
        self.last_q = int(self.cfg.extras.get("last_q", 64))
        self.n_vertical = int(self.cfg.extras.get("n_vertical", 4))
        self.n_slash = int(self.cfg.extras.get("n_slash", 4))
        self.resolved: Optional[str] = None

    def _a_shape(self, q: torch.Tensor, t_k: int, offset: int) -> torch.Tensor:
        t_q, device = q.shape[-2], q.device
        qi = torch.arange(t_q, device=device).unsqueeze(1) + offset
        kj = torch.arange(t_k, device=device).unsqueeze(0)
        sink = kj < (self.cfg.sink_size + self.cfg.block_size)
        local = (qi - kj) < self.cfg.window_size
        return (sink | local).expand(q.shape[0], q.shape[1], t_q, t_k)

    def _vertical_slash(self, q, k, offset) -> torch.Tensor:
        t_q, t_k, device = q.shape[-2], k.shape[-2], q.device
        probe = min(self.last_q, t_q)
        q_probe = q[..., t_q - probe :, :]
        k_rep = repeat_kv(k, self.cfg.group_size)
        scores = torch.matmul(q_probe, k_rep.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        rows = torch.arange(probe, device=device).unsqueeze(1) + (offset + t_q - probe)
        cols = torch.arange(t_k, device=device).unsqueeze(0)
        weights = torch.softmax(scores.masked_fill(cols > rows, float("-inf")), dim=-1)
        # vertical lines: columns that the probe rows attend to
        vertical = weights.sum(-2)
        # slash lines: diagonals j - i, summed over the probe rows
        offsets = cols.unsqueeze(0) - rows
        shift = offset + t_q - 1
        n_off = shift + t_k
        index = (offsets + shift).clamp(0, n_off - 1)
        flat_w = weights.reshape(*weights.shape[:-2], -1)
        flat_i = index.reshape(-1).view(1, 1, -1).expand(*weights.shape[:-2], -1)
        diag = torch.zeros(*weights.shape[:-2], n_off, device=device, dtype=weights.dtype)
        diag.scatter_add_(-1, flat_i, flat_w)
        v_top = torch.topk(vertical, min(self.n_vertical * self.cfg.block_size, t_k), dim=-1).indices
        s_top = torch.topk(diag, min(self.n_slash, n_off), dim=-1).indices
        s_off = (s_top - shift).unsqueeze(-1)      # diagonal offsets d = j - i
        qi = torch.arange(t_q, device=device).unsqueeze(1) + offset
        kj = torch.arange(t_k, device=device).unsqueeze(0)
        allow = torch.zeros(q.shape[0], q.shape[1], t_q, t_k, dtype=torch.bool, device=device)
        vertical_hits = torch.zeros_like(allow)
        # a vertical line is a global column: every query row reads it
        # v_top is (B, H, n_vertical): keep B and H, broadcast over queries.  Using
        # shape[:-2] here drops *two* dims and flattens (H, n_vertical) into one, which
        # only broadcast by accident at B == 1 and crashed at B > 1.
        v_idx = v_top.unsqueeze(-2).expand(*allow.shape[:-1], -1)
        vertical_hits.scatter_(-1, v_idx, True)
        allow = allow | vertical_hits
        for slot in range(s_off.shape[-2]):
            # an offset d selected on the probe rows is reused for every query: j = i + d
            j = (torch.arange(t_q, device=device).view(1, 1, -1, 1) + offset) \
                + s_off[..., slot, :].view(*s_off.shape[:-2], 1, 1)
            valid = (j >= 0) & (j < t_k)
            onehot = torch.zeros_like(allow)
            onehot.scatter_(-1, j.clamp(0, t_k - 1), valid)
            allow = allow | onehot
        allow |= (kj < self.cfg.sink_size)
        return allow

    def _block_sparse(self, q, k, offset) -> torch.Tensor:
        t_q, t_k, bs, device = q.shape[-2], k.shape[-2], self.cfg.block_size, q.device
        n_qb = (t_q + bs - 1) // bs
        n_kb = (t_k + bs - 1) // bs
        k_rep = repeat_kv(k, self.cfg.group_size)
        qp = block_reduce(q, bs, -2)
        kp = block_reduce(k_rep, bs, -2)
        block_scores = torch.matmul(qp, kp.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        qb = (torch.arange(n_qb, device=device) + offset // bs).unsqueeze(1)
        kb = torch.arange(n_kb, device=device).unsqueeze(0)
        allowed = (kb <= qb).expand(q.shape[0], q.shape[1], n_qb, n_kb)
        sel = topk_blocks(block_scores, self.cfg.topk_blocks, allowed)
        sel = sel | (kb == qb).expand_as(sel)
        mask = block_mask_to_token_mask(sel, bs, bs, t_q, t_k)
        return mask | (torch.arange(t_k, device=device).unsqueeze(0) < self.cfg.sink_size)

    def _candidates(self, q, state, offset) -> Dict[str, torch.Tensor]:
        k = state["k"]
        return {
            "a_shape": self._a_shape(q, k.shape[-2], offset),
            "vertical_slash": self._vertical_slash(q, k, offset),
            "block_sparse": self._block_sparse(q, k, offset),
        }

    def select(self, q: torch.Tensor, state: Dict[str, Any], offset: int) -> torch.Tensor:
        k, v = state["k"], state["v"]
        t_q, t_k, device = q.shape[-2], k.shape[-2], q.device
        if self.pattern != "auto":
            if self.pattern == "a_shape":
                return self._a_shape(q, t_k, offset)
            if self.pattern == "vertical_slash":
                return self._vertical_slash(q, k, offset)
            if self.pattern == "block_sparse":
                return self._block_sparse(q, k, offset)
            raise ValueError(f"unknown pattern {self.pattern!r}")
        if self.resolved is not None:
            return self._candidates(q, state, offset)[self.resolved]
        candidates = self._candidates(q, state, offset)
        with torch.no_grad():
            weights = dense_attention(q, k, v, causal=True, return_weights=True)[1]
        causal = causal_mask(t_q, t_k, offset=offset, device=device)
        scored = {}
        for name, mask in candidates.items():
            valid = mask & causal
            recall = float((weights * valid).sum() / weights.sum().clamp_min(1e-9))
            reads = float(valid.sum(-1).float().mean())
            scored[name] = (recall, reads)
        # The paper's search is *equal-budget*: build candidates that cost the same
        # and keep the one with the best recall.  Normalising recall by reads instead
        # (dividing by the token budget) systematically rewards patterns that read
        # almost nothing, which is not the objective.
        cheapest = min(reads for _, reads in scored.values())
        eligible = {n: r for n, (r, reads) in scored.items() if reads <= cheapest * 1.25 + 1}
        best = max(eligible, key=lambda n: eligible[n])
        self.resolved = best
        self._extra_stats = {f"pattern_{n}": v[0] for n, v in scored.items()}
        self._extra_stats["resolved_pattern_id"] = float(list(candidates).index(best))
        return candidates[best]

    def selection_flops(self, seq_len: int, batch: int = 1) -> int:
        probe = min(self.last_q, seq_len)
        bs = self.cfg.block_size
        vs = 2 * 2 * batch * self.cfg.n_heads * probe * seq_len * self.cfg.head_dim
        bs_cost = 2 * 2 * batch * self.cfg.n_heads * (seq_len // bs) ** 2 * self.cfg.head_dim
        return int(vs + bs_cost)


# --------------------------------------------------------------------------------------
# FlexPrefill
# --------------------------------------------------------------------------------------
class FlexPrefillAttention(BlockSelectionAttention):
    """FlexPrefill: context-aware, budget-adaptive block selection.

    Lai et al., *FlexPrefill: A Context-Aware Sparse Attention Mechanism for
    Efficient Long-Sequence Inference* (2025), https://arxiv.org/abs/2502.20766

    Two stages, both reproduced:

    1. **pattern estimation** -- a representative query block (the last
       ``block_size`` queries) is scored against the whole cache; if the block
       distribution implied by the probe is close (Jensen-Shannon distance) to the
       distribution implied by walking the candidate keys, the head is treated as
       *structured* (one global estimate reused for every query block), otherwise
       as *query-aware* (each query block estimates its own blocks from a
       subsample of keys per block);
    2. **cumulative-attention index selection** -- blocks are taken in descending
       estimated score until the accumulated estimated mass reaches
       ``threshold`` (0.9 for query-aware heads, 0.95 for structured ones in the
       paper).  The read budget is data-dependent, which is the point of the
       method.
    """

    name = "flexprefill"
    family = "selection"
    paper = ("Lai et al., FlexPrefill (2025)", "https://arxiv.org/abs/2502.20766")

    def __init__(self, cfg=None, **kw):
        super().__init__(cfg, **kw)
        self.tau_js = float(self.cfg.extras.get("tau_js", 0.05))
        self.alpha_query_aware = float(self.cfg.extras.get("alpha_query_aware", 0.9))
        self.alpha_structured = float(self.cfg.extras.get("alpha_structured", 0.95))
        self.resolved: Optional[str] = None

    def _pooled_block_scores(self, q, k_rep, bs):
        return torch.matmul(block_reduce(q, bs, -2), block_reduce(k_rep, bs, -2).transpose(-1, -2)) / (
            q.shape[-1] ** 0.5
        )

    def select(self, q: torch.Tensor, state: Dict[str, Any], offset: int) -> torch.Tensor:
        k, v = state["k"], state["v"]
        t_q, t_k, bs, device = q.shape[-2], k.shape[-2], self.cfg.block_size, q.device
        k_rep = repeat_kv(k, self.cfg.group_size)
        probe = min(bs, t_q)
        q_probe = q[..., t_q - probe :, :]
        est_logits = self._pooled_block_scores(q_probe, k_rep, bs)[..., -1, :]
        est = torch.softmax(est_logits, dim=-1)
        with torch.no_grad():
            w = dense_attention(q_probe, k, v, causal=True, return_weights=True)[1]
            true = block_reduce(w, bs, -1)[..., -1, :]
            true = true / true.sum(-1, keepdim=True).clamp_min(1e-9)
            mixture = 0.5 * (est + true)
            js = 0.5 * (est * (est.clamp_min(1e-9) / mixture.clamp_min(1e-9)).log()).sum(-1)
            js = js + 0.5 * (true * (true.clamp_min(1e-9) / mixture.clamp_min(1e-9)).log()).sum(-1)
            structured = bool((js.sqrt() <= self.tau_js).all())
        if self.resolved is None:
            self.resolved = "structured" if structured else "query_aware"
        alpha = self.alpha_structured if structured else self.alpha_query_aware
        n_kb = est.shape[-1]
        if structured:
            scores = est.unsqueeze(-2).expand(*est.shape[:-2], t_q, n_kb)
        else:
            sample = min(int(self.cfg.extras.get("sample_keys", 8)), bs)
            n_kb_full = (t_k + bs - 1) // bs
            base = (torch.arange(n_kb_full, device=device).unsqueeze(-1) * bs
                    + torch.arange(sample, device=device)).clamp(max=t_k - 1)
            keys = k_rep[..., base.reshape(-1), :].view(*k_rep.shape[:-2], n_kb_full, sample, k_rep.shape[-1])
            # query-aware estimate: mean over the sampled keys of each block
            scores = torch.einsum("bhtd,bhknd->bhtkn", q, keys).mean(-1) / (q.shape[-1] ** 0.5)
            scores = scores[..., :n_kb]
        qb = (torch.arange(t_q, device=device) + offset) // bs
        kb = torch.arange(n_kb, device=device).unsqueeze(0)
        allowed = (kb <= qb.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(torch.nan_to_num(scores, neginf=-1e30), dim=-1)
        sel = _cumulative_select(probs, alpha)
        sel = sel | (kb == qb.unsqueeze(-1)).unsqueeze(0).unsqueeze(0)   # own block always read
        self._extra_stats = {
            "blocks_per_query": float(sel.sum(-1).float().mean()),
            "structured": 1.0 if structured else 0.0,
        }
        return block_mask_to_token_mask(sel, 1, bs, t_q, t_k)

    def selection_flops(self, seq_len: int, batch: int = 1) -> int:
        bs = self.cfg.block_size
        probe = min(bs, seq_len)
        return int(2 * 2 * batch * self.cfg.n_heads * (probe + seq_len) * seq_len * self.cfg.head_dim / max(1, bs))


# --------------------------------------------------------------------------------------
# XAttention
# --------------------------------------------------------------------------------------
class XAttention(BlockSelectionAttention):
    """XAttention: strided antidiagonal block scoring with a single threshold.

    Xu et al., *XAttention: Block Sparse Attention with Antidiagonal Scoring*
    (ICML 2025), https://arxiv.org/abs/2503.16428

    Why antidiagonals: within a block, a vertical line, a slash line and a dense
    blob all intersect the antidiagonals, so summing *only* the strided
    antidiagonals (``1/stride`` of the block's entries) recovers a good proxy for
    the block's softmax mass at a fraction of the cost of pooling-based scores.
    The paper then softmax-normalises the block scores and keeps the minimal set
    of blocks whose cumulative score reaches ``threshold`` (0.9-0.95), with
    per-head thresholds found offline by dynamic programming; here the threshold
    is a config knob and ``threshold_per_head`` allows the offline-searched
    values to be supplied.
    """

    name = "xattention"
    family = "selection"
    paper = ("Xu et al., XAttention (ICML 2025)", "https://arxiv.org/abs/2503.16428")

    def __init__(self, cfg=None, **kw):
        super().__init__(cfg, **kw)
        self.r = int(self.cfg.extras.get("antidiagonal_stride", self.cfg.stride))
        self.block = int(self.cfg.extras.get("xattn_block", 8))
        # the paper ablates three block-selection rules; all three are supported here
        self.select_mode = self.cfg.extras.get("select_mode", "threshold")

    def antidiagonal_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Strided antidiagonal sums of the block-diagonal attention submatrices.

        Returns ``(B, H, n_qb, n_kb, n_antidiagonal)``.  Keeping the antidiagonals
        separate -- instead of summing them into a single block score -- is what the
        paper's selection operates on: the softmax is over all (key block,
        antidiagonal) pairs, and a block is read if one of its antidiagonals falls
        inside the prefix that reaches the threshold.  Summing first and *then*
        softmaxing is not equivalent, and in practice it degenerates: block scores
        spanning several hundred units make the softmax collapse onto one block.
        """
        b, h, t_q, d = q.shape
        t_k = k.shape[-2]
        blk = self.block
        qb, kb = (t_q + blk - 1) // blk, (t_k + blk - 1) // blk
        pad_q, pad_k = qb * blk - t_q, kb * blk - t_k
        qp = F.pad(q, (0, 0, 0, pad_q))
        kp = repeat_kv(k, self.cfg.group_size)
        kp = F.pad(kp, (0, 0, 0, pad_k))
        # Scores for every (query block, key block) pair, computed only on the
        # strided antidiagonals: 1/stride of the full block-gram cost.  Each
        # antidiagonal is evaluated as a batched matmul with the row pairs as the
        # reduction axis, so the peak memory stays at O(n_qb * n_kb) per head
        # instead of O(n_qb * n_kb * blk * d).
        qb_ = qp.reshape(b, h, qb, blk, d)
        kb_ = kp.reshape(b, h, kb, blk, d)
        offsets = list(range(0, 2 * blk - 1, self.r))
        out = torch.zeros(b, h, qb, kb, len(offsets), device=q.device, dtype=torch.float32)
        rows_all = torch.arange(blk, device=q.device)
        for slot, off in enumerate(offsets):
            cols = off - rows_all
            valid = (cols >= 0) & (cols < blk)
            if not bool(valid.any()):
                continue
            rows, cols = rows_all[valid], cols[valid]
            qs = qb_.index_select(-2, rows).to(torch.float32)        # (B,H,qb,n,D)
            ks = kb_.index_select(-2, cols).to(torch.float32)        # (B,H,kb,n,D)
            out[..., slot] = torch.einsum("bhqnd,bhknd->bhqk", qs, ks)
        return out / (q.shape[-1] ** 0.5)

    def select(self, q: torch.Tensor, state: Dict[str, Any], offset: int) -> torch.Tensor:
        k = state["k"]
        t_q, t_k, blk, device = q.shape[-2], k.shape[-2], self.block, q.device
        scores = self.antidiagonal_scores(q, k)                        # (B,H,nqb,nkb,n_anti)
        n_qb, n_kb, n_anti = scores.shape[-3], scores.shape[-2], scores.shape[-1]
        qb = (torch.arange(n_qb, device=device) * blk).unsqueeze(1) + offset
        kb = (torch.arange(n_kb, device=device) * blk).unsqueeze(0)
        allowed = (kb <= qb).unsqueeze(0).unsqueeze(0)                 # block-level causality
        flat = scores.masked_fill(~allowed.unsqueeze(-1), float("-inf"))
        flat = torch.nan_to_num(flat, neginf=-1e30).reshape(*scores.shape[:-2], n_kb * n_anti)
        probs = torch.softmax(flat, dim=-1).reshape(*scores.shape[:-2], n_kb, n_anti)
        block_mass = probs.sum(-1)                                     # what a block contributes
        threshold = self.cfg.extras.get("threshold_per_head", self.cfg.threshold)
        if self.select_mode == "topk":
            sel = topk_blocks(block_mass, self.cfg.topk_blocks, allowed)
        elif self.select_mode == "top_ratio":
            ratio = float(self.cfg.extras.get("top_ratio", 0.02))
            per_query = (allowed.sum(-1).float() * ratio).ceil().clamp(min=1)
            sel = self._select_with_per_head(block_mass.masked_fill(~allowed, 0.0), per_query.unsqueeze(-1))
        elif isinstance(threshold, (list, tuple)):
            thresh = torch.tensor(list(threshold), device=device, dtype=block_mass.dtype).view(1, -1, 1, 1)
            sel = self._select_with_per_head(block_mass, thresh)
        else:
            sel = _cumulative_select(block_mass, float(threshold))
        sel = sel | (kb == qb).unsqueeze(0).unsqueeze(0)
        sel = sel & allowed
        self._extra_stats = {"blocks_per_query": float(sel.sum(-1).float().mean())}
        return block_mask_to_token_mask(sel, blk, blk, t_q, t_k)

    @staticmethod
    def _select_with_per_head(probs: torch.Tensor, thresh: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(probs, dim=-1, descending=True)
        sorted_probs = torch.gather(probs, -1, order)
        reached = sorted_probs.cumsum(-1) >= thresh
        first = reached.float().argmax(-1)
        first = torch.where(reached.any(-1), first, torch.full_like(first, probs.shape[-1] - 1))
        n_keep = (first + 1).clamp(max=probs.shape[-1])
        rank = torch.argsort(order, dim=-1)
        return rank < n_keep.unsqueeze(-1)

    def selection_flops(self, seq_len: int, batch: int = 1) -> int:
        blk = self.block
        n_qb = (seq_len + blk - 1) // blk
        n_kb = n_qb
        return int(2 * 2 * batch * self.cfg.n_heads * n_qb * n_kb * blk * blk / self.r * self.cfg.head_dim / blk)


# --------------------------------------------------------------------------------------
# MoBA
# --------------------------------------------------------------------------------------
class MoBAAttention(BlockSelectionAttention):
    """Mixture of Block Attention: per-token top-k routing over KV blocks.

    Lu et al., *MoBA: Mixture of Block Attention for Long-Context LLMs* (2025),
    https://arxiv.org/abs/2502.13189

    The gate is the inner product between the query and the **mean-pooled key
    block**; each query routes to its ``topk_blocks`` predecessors, always
    including its own (causally masked) block.  MoBA can emulate sliding-window
    and sink attention as special cases of the routing function, which is the
    paper's "less structure" argument.
    """

    name = "moba"
    family = "selection"
    paper = ("Lu et al., MoBA (2025)", "https://arxiv.org/abs/2502.13189")

    def select(self, q: torch.Tensor, state: Dict[str, Any], offset: int) -> torch.Tensor:
        k = state["k"]
        t_q, t_k, bs, device = q.shape[-2], k.shape[-2], self.cfg.block_size, q.device
        n_kb = (t_k + bs - 1) // bs
        k_rep = repeat_kv(k, self.cfg.group_size)
        k_mean = block_reduce(k_rep, bs, -2)                                  # (B,H,nkb,D)
        gates = torch.matmul(q, k_mean.transpose(-1, -2)) / (q.shape[-1] ** 0.5)                     # (B,H,Tq,nkb)
        q_pos = torch.arange(t_q, device=device) + offset
        kb_start = torch.arange(n_kb, device=device) * bs
        allowed = kb_start.unsqueeze(0) <= q_pos.unsqueeze(1)          # a query never routes to a future block
        sel = topk_blocks(gates, self.cfg.topk_blocks, allowed.expand_as(gates))
        own = (q_pos // bs).clamp(max=n_kb - 1)
        sel = sel.scatter(-1, own.view(1, 1, -1, 1).expand_as(sel[..., :1]), True)
        self._extra_stats = {"blocks_per_query": float(sel.sum(-1).float().mean())}
        return block_mask_to_token_mask(sel, 1, bs, t_q, t_k)

    def selection_flops(self, seq_len: int, batch: int = 1) -> int:
        n_kb = (seq_len + self.cfg.block_size - 1) // self.cfg.block_size
        return int(2 * batch * self.cfg.n_heads * seq_len * n_kb * self.cfg.head_dim)


# --------------------------------------------------------------------------------------
# DeepSeek Sparse Attention (DSA) / Compressed Sparse Attention (CSA)
# --------------------------------------------------------------------------------------
class DSAAttention(BlockSelectionAttention):
    """DeepSeek Sparse Attention: learned lightning indexer + top-k token selection.

    DeepSeek-AI, *DeepSeek-V3.2* (2025), https://arxiv.org/abs/2512.02556 (tech
    report distributed with the release; see also *DeepSeek-V3.2-Exp*).

    The indexer score between query ``t`` and key ``s`` is

    .. math::  I_{t,s} = \\sum_{j=1}^{H_I} w_{t,j}\\,\\mathrm{ReLU}(q_{t,j}\\cdot k_s)

    with ``H_I`` indexer heads and learned, per-token head weights ``w``.  Keys
    with the top ``k`` scores (2048 in the V3.2 release) are then read by the core
    attention, which in DeepSeek's case is MLA over the selected tokens.

    ``compress_ratio > 1`` implements the **compressed-indexing part** of CSA/CSA2:
    the indexer scores pooled keys and the selected groups expand to the tokens they
    contain -- the mechanism that makes the indexer cheap enough to sit in front of
    every layer.

    It is **not** CSA2.  CSA2 (DeepSeek-V4.1-Flash, arXiv:2609.19969) also (a)
    reuses one cached KV copy across a group of layers and (b) stores the latent in
    FP4.  This variant does neither: it is single-layer and fp32, and the ``csa2``
    entry in ``kv/format.py`` is a cost-model constant taken from the report, not an
    implementation.  See ``docs/kv-layer.md`` for the component-by-component mapping
    and ``docs/roadmap.md`` for the gap.

    Because the indexer is a *learned* component, a randomly initialised one is
    meaningless; :meth:`fit` trains the indexer projections on calibration inputs
    against the dense attention mass, which is what makes the capacity question
    ("can a low-dimensional ReLU indexer rank keys the way softmax does?")
    testable.
    """

    name = "dsa"
    family = "selection"
    trained = True
    paper = ("DeepSeek-AI, DeepSeek-V3.2 / DSA + lightning indexer (2025)",
             "https://huggingface.co/deepseek-ai/DeepSeek-V3.2")

    def __init__(self, cfg=None, **kw):
        super().__init__(cfg, **kw)
        self.n_indexer_heads = int(self.cfg.extras.get("indexer_heads", 8))
        self.indexer_dim = int(self.cfg.extras.get("indexer_dim", 16))
        self.compress_ratio = int(self.cfg.extras.get("compress_ratio", self.cfg.compress_ratio))
        gen = torch.Generator().manual_seed(self.cfg.seed)
        # One small projection per (attention head, indexer head).  In DeepSeek's
        # implementation the indexer heads read the layer's hidden state directly and
        # are shared by the attention heads; our harness only has q/k/v, so the
        # indexer is parameterised per attention head instead.  The capacity
        # question -- can a low-dimensional ReLU score rank keys the way softmax
        # does? -- is unaffected by that difference.
        self.q_proj = torch.nn.Parameter(
            torch.randn(self.cfg.n_heads, self.n_indexer_heads, self.cfg.head_dim, self.indexer_dim,
                        generator=gen) * 0.02
        )
        self.head_weights = torch.nn.Parameter(torch.full((self.n_indexer_heads,), 1.0 / self.n_indexer_heads))

    # -- indexer -------------------------------------------------------------------
    def indexer_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """``(B, H, T_q, T_k)`` lightning-indexer scores."""
        b, h, t_q, d = q.shape
        t_k = k.shape[-2]
        proj = self.q_proj.to(q.dtype)                                              # (H, E, d_in, d_out)
        q_idx = torch.einsum("bhtd,hedf->bhtef", q, proj)                           # (B,H,Tq,E,F)
        k_idx = k.unsqueeze(2).expand(b, self.cfg.n_kv_heads, self.cfg.group_size, t_k, d)
        k_idx = k_idx.reshape(b, h, t_k, d)
        k_idx = torch.einsum("bhtd,hedf->bhtef", k_idx, proj)                       # (B,H,Tk,E,F)
        w = torch.softmax(self.head_weights.to(q.dtype), dim=-1)
        # I(t, s) = sum_e w_e ReLU(q_e . k_s).  Accumulating one indexer head at a
        # time keeps the peak memory at O(T_q * T_k) instead of O(T_q * T_k * H_I),
        # which is the same reason the real kernel streams the score computation.
        scores = torch.zeros(b, h, t_q, t_k, dtype=q.dtype, device=q.device)
        for e in range(self.n_indexer_heads):
            dots = torch.einsum("bhtf,bhsf->bhts", q_idx[..., e, :], k_idx[..., e, :])
            scores = scores + w[e] * torch.relu(dots)
        return scores

    def select(self, q: torch.Tensor, state: Dict[str, Any], offset: int) -> torch.Tensor:
        k = state["k"]
        t_q, t_k, device = q.shape[-2], k.shape[-2], q.device
        if self.compress_ratio > 1:
            m = self.compress_ratio
            k_pool = block_reduce(k, m, -2)
            scores = self.indexer_scores(q, k_pool)                    # over compressed groups
            n_groups = scores.shape[-1]
            q_pos = torch.arange(t_q, device=device) + offset
            grp_start = torch.arange(n_groups, device=device) * m
            allowed = grp_start.unsqueeze(0) <= q_pos.unsqueeze(1)
            scores = scores.masked_fill(~allowed, float("-inf"))
            n_keep = max(1, self.cfg.topk_tokens // m)
            sel_groups = topk_blocks(scores, n_keep)
            sel = sel_groups.repeat_interleave(m, dim=-1)[..., :t_k]
        else:
            scores = self.indexer_scores(q, k)
            n_keep = min(self.cfg.topk_tokens, t_k)
            sel = torch.zeros_like(scores, dtype=torch.bool)
            idx = torch.topk(scores, n_keep, dim=-1).indices
            sel.scatter_(-1, idx, True)
        # sinks and the local window are always read (the released implementation
        # keeps a fixed number of leading tokens and a sliding window)
        kj = torch.arange(t_k, device=device)
        qi = (torch.arange(t_q, device=device) + offset).unsqueeze(-1)
        sel = sel | (kj < self.cfg.sink_size) | ((qi - kj) < self.cfg.window_size)
        self._extra_stats = {"tokens_per_query": float(sel.sum(-1).float().mean())}
        return sel

    def fit(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, steps: int = 300, lr: float = 0.05) -> float:
        """Train the indexer to rank keys like the dense attention does.

        Returns the final cross-entropy between the indexer-induced distribution
        and the true attention distribution (lower is better).
        """
        with torch.no_grad():
            target = dense_attention(q, k, v, causal=True, return_weights=True)[1].mean(1, keepdim=True)
            target = target.expand(-1, self.cfg.n_heads, -1, -1)
        opt = torch.optim.Adam([self.q_proj, self.head_weights], lr=lr)
        loss_value = float("nan")
        for _ in range(steps):
            opt.zero_grad()
            scores = self.indexer_scores(q, k)
            logp = torch.log_softmax(scores, dim=-1)
            mask = causal_mask(q.shape[-2], k.shape[-2], 0, q.device)
            loss = -(target * logp).nan_to_num().sum(-1)[..., mask.any(-1)].mean()
            loss.backward()
            opt.step()
            loss_value = float(loss.detach())
        return loss_value

    def selection_flops(self, seq_len: int, batch: int = 1) -> int:
        m = max(1, self.compress_ratio)
        return int(2 * batch * self.cfg.n_heads * seq_len * (seq_len // m) * self.indexer_dim)


# --------------------------------------------------------------------------------------
# SeerAttention-R
# --------------------------------------------------------------------------------------
class SeerAttentionR(BlockSelectionAttention):
    """Learned block pooling gate, adapted for long decoding.

    Gao et al., *SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs*
    (ICLR 2025) and *SeerAttention-R: Sparse Attention Adaptation for Long
    Reasoning* (2025), https://arxiv.org/abs/2506.08889

    A ``BlockPool`` averages q and k inside each block, a learned projection maps
    the pooled vectors to a low-dimensional space, and a sigmoid gate decides
    whether a block is worth reading.  Unlike the training-free selectors above,
    the gate carries weights, so this class provides :meth:`fit_gate`, which
    trains the gate against the dense block mass.  That makes the capacity of the
    learned-gate parameterisation measurable, and it is also why this family is
    the one that needs a fine-tuning stage in practice.
    """

    name = "seerattention_r"
    family = "selection"
    trained = True
    paper = ("Gao et al., SeerAttention-R (2025)", "https://arxiv.org/abs/2506.08889")

    def __init__(self, cfg=None, **kw):
        super().__init__(cfg, **kw)
        self.pool_dim = int(self.cfg.extras.get("pool_dim", 16))
        self.bias = float(self.cfg.extras.get("gate_bias", 0.0))
        gen = torch.Generator().manual_seed(self.cfg.seed + 1)
        self.proj = torch.nn.Parameter(torch.randn(self.cfg.head_dim, self.pool_dim, generator=gen) * 0.02)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def block_gate(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Gate scores ``(B,H,n_qb,n_kb)`` from pooled q/k."""
        bs = self.cfg.block_size
        k_rep = repeat_kv(k, self.cfg.group_size)
        qp = block_reduce(q, bs, -2)
        kp = block_reduce(k_rep, bs, -2)
        proj = self.proj.to(q.dtype)
        q_proj = torch.matmul(qp, proj)
        k_proj = torch.matmul(kp, proj)
        return torch.sigmoid(self.scale.to(q.dtype) * torch.matmul(q_proj, k_proj.transpose(-1, -2)) + self.bias)

    def select(self, q: torch.Tensor, state: Dict[str, Any], offset: int) -> torch.Tensor:
        k, device = state["k"], q.device
        t_q, t_k, bs = q.shape[-2], k.shape[-2], self.cfg.block_size
        gates = self.block_gate(q, k)                                  # (B,H,nqb,nkb)
        n_qb, n_kb = gates.shape[-2:]
        qb = (torch.arange(n_qb, device=device) * bs + offset).unsqueeze(1)
        kb = (torch.arange(n_kb, device=device) * bs).unsqueeze(0)
        allowed = (kb <= qb).unsqueeze(0).unsqueeze(0)
        scores = gates.masked_fill(~allowed, float("-inf"))
        topn = int(self.cfg.extras.get("seer_topn", self.cfg.topk_blocks))
        sel = topk_blocks(scores, topn, allowed)
        sel = sel | (kb == qb).unsqueeze(0).unsqueeze(0)
        return block_mask_to_token_mask(sel, bs, bs, t_q, t_k)

    def fit_gate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        steps: int = 300,
        lr: float = 0.02,
    ) -> float:
        """Train the gate to reproduce the dense block-level attention mass."""
        bs = self.cfg.block_size
        with torch.no_grad():
            w = dense_attention(q, k, v, causal=True, return_weights=True)[1]     # (B,H,Tq,Tk)
            target = block_reduce(w, bs, -1)
            target = block_reduce(target, bs, -2)
            target = target / target.sum(-1, keepdim=True).clamp_min(1e-9)
        opt = torch.optim.Adam([self.proj, self.scale], lr=lr)
        loss_value = float("nan")
        for _ in range(steps):
            opt.zero_grad()
            gates = self.block_gate(q, k)
            logp = torch.log_softmax(gates, dim=-1)
            loss = -(target * logp).sum(-1).mean()
            loss.backward()
            opt.step()
            loss_value = float(loss.detach())
        return loss_value
