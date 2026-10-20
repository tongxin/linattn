"""Attention and KV-management variants from the 2025--2026 research frontier.

The earlier academic page established the vocabulary (low rank, random features,
fixed sparsity, eviction).  This module tracks the lines that are live now:

* **Selection is becoming output-aware.**  Attention mass is only one proxy for the
  quantity that matters, the error in ``A V``.  MASA scores sparse units by their
  reduction in matrix-product error; OVAL builds page bases from centred key--value
  geometry; DropKV and ReST-KV rank eviction candidates by output perturbation.
* **Selection is becoming hierarchical and asymmetric.**  PISA narrows a pyramid of
  key summaries instead of scoring every query-block pair.  SAGA exploits the fact
  that sparse attention shifts the decode bottleneck from value reads to QK scoring,
  so fewer key heads and more value heads can be the better allocation.
* **Compression is no longer only along the sequence axis.**  SFA sparsifies the
  feature axis while keeping every token, which is complementary to eviction and
  block selection.

These are reference semantics, not the authors' kernels.  Each class states which
part is faithful and which is an approximation.  Selection variants evaluate exact
softmax attention *under the mask they select*, so their measured error is
attributable to selection rather than to a second numerical approximation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from linattn.core import (
    AttentionVariant,
    AttnConfig,
    block_reduce,
    causal_mask,
    dense_attention,
    repeat_kv,
)
from linattn.variants.eviction import EvictionAttention
from linattn.variants.selection import BlockSelectionAttention

__all__ = [
    "SAGAAttention",
    "PISAAttention",
    "MASAAttention",
    "JSSAAttention",
    "OVALAttention",
    "SFAAttention",
    "DropKVAttention",
    "ReSTKVAttention",
    "BeaconKVAttention",
]


def _causal_token_mask(q: torch.Tensor, t_k: int, offset: int) -> torch.Tensor:
    return causal_mask(q.shape[-2], t_k, offset=offset, device=q.device)


def _expand_head_groups(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Repeat ``(B, H_kv, ...)`` tensors along the query-head axis."""
    if group_size == 1:
        return x
    b, h, *rest = x.shape
    return x[:, :, None].expand(b, h, group_size, *rest).reshape(b, h * group_size, *rest)


def _top_tokens(scores: torch.Tensor, k: int, allowed: torch.Tensor) -> torch.Tensor:
    scores = scores.masked_fill(~allowed, float("-inf"))
    k = max(1, min(k, scores.shape[-1]))
    idx = torch.topk(scores, k, dim=-1).indices
    out = torch.zeros_like(scores, dtype=torch.bool)
    out.scatter_(-1, idx, True)
    return out & allowed


class _FrontierSelection(BlockSelectionAttention):
    family = "frontier"

    def _finish(self, allow: torch.Tensor) -> torch.Tensor:
        self._reads = float(allow[0, 0].sum(-1).float().mean())
        return allow


# --------------------------------------------------------------------------------------
# SAGA / Atop-N (NeurIPS 2026)
# --------------------------------------------------------------------------------------
class SAGAAttention(_FrontierSelection):
    """Sparse Asymmetric Group-Query Attention with approximate top-N selection.

    Elata et al., *More Value per Key: Asymmetric Sparse Attention for Faster LLM
    Decoding* (NeurIPS 2026), https://arxiv.org/abs/2610.04753

    Faithful core: sparse token selection decouples the number of key heads from the
    number of value heads, and the memory model reports that asymmetry explicitly.
    The paper trains Atop-N and SAGA; this implementation uses the deployable
    inference semantics -- exact top-N by the current query -- and lets
    ``extras["key_heads"]`` choose the key-side allocation.  Value heads stay at
    ``cfg.n_kv_heads``.
    """

    name = "saga"
    trained = True
    paper = ("Elata et al., SAGA / More Value per Key (NeurIPS 2026)",
             "https://arxiv.org/abs/2610.04753")

    def __init__(self, cfg: Optional[AttnConfig] = None, **kw: Any):
        super().__init__(cfg, **kw)
        self.key_heads = max(1, min(int(self.cfg.extras.get("key_heads", 2)),
                                    self.cfg.n_kv_heads))

    def _key_index(self, device) -> torch.Tensor:
        h = self.cfg.n_heads
        return (torch.arange(h, device=device) * self.key_heads // h).clamp(max=self.key_heads - 1)

    def select(self, q: torch.Tensor, state: Dict[str, Any], offset: int) -> torch.Tensor:
        k = state["k"]
        t_k = k.shape[-2]
        idx = self._key_index(q.device)
        k_selected = k.index_select(1, idx)                       # (B,H,T,D)
        scores = torch.matmul(q, k_selected.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        causal = _causal_token_mask(q, t_k, offset)
        keep = max(1, min(self.cfg.topk_tokens, t_k))
        allow = _top_tokens(scores, keep, causal)
        if self.cfg.sink_size:
            allow |= (torch.arange(t_k, device=q.device) < self.cfg.sink_size).view(1, 1, 1, -1)
        self._extra_stats = {"key_heads": float(self.key_heads),
                             "value_heads": float(self.cfg.n_kv_heads)}
        return self._finish(allow & causal)

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(min(seq_len, self.cfg.topk_tokens + self.cfg.sink_size))

    def selection_flops(self, seq_len: int, batch: int = 1) -> int:
        return int(2 * 2 * batch * self.cfg.n_heads * seq_len * seq_len * self.cfg.head_dim)

    def memory_bytes(self, seq_len: int, batch: int = 1) -> int:
        d = self.cfg.head_dim
        heads = self.key_heads + self.cfg.n_kv_heads
        return int(2 * batch * heads * seq_len * d)


# --------------------------------------------------------------------------------------
# PISA / pyramid Top-K (2026)
# --------------------------------------------------------------------------------------
class PISAAttention(_FrontierSelection):
    """Pyramid Top-K block selection with log-linear routing.

    Tang et al., *Block Sparse Attention with Log-Linear Complexity* (2026),
    https://arxiv.org/abs/2609.31093

    Faithful core: build a coarse-to-fine hierarchy of pooled keys, select at the
    coarsest level, and only expand the children of selected nodes.  Each level
    scores a bounded candidate set, so routing does not require every query-block
    pair.  The paper uses learned/LogSumExp node scores and fused Triton kernels;
    here nodes are mean-pooled keys and the final attention is exact over the
    selected tokens.
    """

    name = "pisa"
    trained = True
    paper = ("Tang et al., Block Sparse Attention with Log-Linear Complexity (2026)",
             "https://arxiv.org/abs/2609.31093")

    def __init__(self, cfg: Optional[AttnConfig] = None, **kw: Any):
        super().__init__(cfg, **kw)
        self.branch = max(2, int(self.cfg.extras.get("branch", 2)))
        self.keep = max(1, int(self.cfg.extras.get("topk_tokens", self.cfg.topk_tokens)))

    def _levels(self, k_rep: torch.Tensor) -> List[torch.Tensor]:
        levels = [k_rep]
        while levels[-1].shape[-2] > self.keep and levels[-1].shape[-2] > 1:
            levels.append(block_reduce(levels[-1], 2, -2))
        return list(reversed(levels))

    def select(self, q: torch.Tensor, state: Dict[str, Any], offset: int) -> torch.Tensor:
        k = state["k"]
        k_rep = repeat_kv(k, self.cfg.group_size)
        t_q, t_k = q.shape[-2], k.shape[-2]
        causal = _causal_token_mask(q, t_k, offset)
        levels = self._levels(k_rep)
        node_idx = torch.arange(levels[0].shape[-2], device=q.device)
        node_idx = node_idx.view(1, 1, 1, -1).expand(q.shape[0], q.shape[1], t_q, -1)
        stride = max(1, 2 ** (len(levels) - 1))
        target = min(self.keep, k_rep.shape[-2])
        final: Optional[torch.Tensor] = None
        for level, keys in enumerate(levels):
            if level == 0:
                keys_sel = keys.unsqueeze(2).expand(-1, -1, t_q, -1, -1)
            else:
                gather_idx = node_idx.clamp(max=keys.shape[-2] - 1).unsqueeze(-1)
                gather_idx = gather_idx.expand(*gather_idx.shape[:-1], keys.shape[-1])
                keys_sel = torch.gather(keys.unsqueeze(2).expand(-1, -1, t_q, -1, -1),
                                        3, gather_idx)
            score = torch.einsum("bhtd,bhtcd->bhtc", q, keys_sel) / math.sqrt(q.shape[-1])
            starts = node_idx * stride
            qpos = torch.arange(t_q, device=q.device).unsqueeze(1) + offset
            visible = starts <= qpos.view(1, 1, -1, 1)
            score = score.masked_fill(~visible, float("-inf"))
            keep = min(node_idx.shape[-1], max(target, target * 2 ** (len(levels) - level - 1)))
            idx = torch.topk(score, keep, dim=-1).indices
            node_idx = torch.gather(node_idx, -1, idx)
            if level == len(levels) - 1:
                final = torch.zeros(q.shape[0], q.shape[1], t_q, t_k,
                                    dtype=torch.bool, device=q.device)
                final.scatter_(-1, node_idx.clamp(max=t_k - 1), True)
                break
            node_idx = (node_idx.unsqueeze(-1) * self.branch +
                        torch.arange(self.branch, device=q.device)).reshape(*node_idx.shape[:-1], -1)
            node_idx = node_idx.clamp(max=levels[level + 1].shape[-2] - 1)
            stride //= self.branch
        assert final is not None
        allow = final & causal
        if self.cfg.sink_size:
            allow |= (torch.arange(t_k, device=q.device) < self.cfg.sink_size).view(1, 1, 1, -1)
        self._extra_stats = {"pyramid_levels": float(len(levels))}
        return self._finish(allow & causal)

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(min(seq_len, self.keep + self.cfg.sink_size))

    def selection_flops(self, seq_len: int, batch: int = 1) -> int:
        levels = max(1, math.ceil(math.log2(max(1, seq_len // max(1, self.keep)))))
        candidates = sum(self.keep * self.branch ** max(0, levels - i) for i in range(levels))
        candidates = min(candidates, seq_len * max(1, levels))
        return int(2 * 2 * batch * self.cfg.n_heads * seq_len * candidates * self.cfg.head_dim / 4)


# --------------------------------------------------------------------------------------
# MASA (2026)
# --------------------------------------------------------------------------------------
class MASAAttention(_FrontierSelection):
    """Matrix Approximation Sparse Attention: score units by output-error reduction.

    Wan et al., *Sparse Attention Is Matrix Approximation, Not Choosing from a Bag
    of Values* (2026), https://arxiv.org/abs/2610.10871

    For a candidate block ``u`` with contribution ``C_u = (M_u * A) V``, MASA uses
    ``2 <O, C_u> - ||C_u||^2``, the reduction in squared ``||A V||_F`` error.  The
    implementation uses the exact profile attention map as the reference's analysis
    oracle and selects top blocks per query.  It is deliberately expensive: the point
    is to measure the attainable gain of an output-aware score before building a
    cheap estimator for it.
    """

    name = "masa"
    trained = False
    paper = ("Wan et al., MASA: Sparse Attention Is Matrix Approximation (2026)",
             "https://arxiv.org/abs/2610.10871")

    def __init__(self, cfg: Optional[AttnConfig] = None, **kw: Any):
        super().__init__(cfg, **kw)
        self.block = int(self.cfg.extras.get("block", self.cfg.block_size))
        self.topk = int(self.cfg.extras.get("topk_blocks", self.cfg.topk_blocks))

    def select(self, q: torch.Tensor, state: Dict[str, Any], offset: int) -> torch.Tensor:
        k, v = state["k"], state["v"]
        t_q, t_k = q.shape[-2], k.shape[-2]
        out, weights = dense_attention(q, k, v, causal=True, return_weights=True)
        v_rep = repeat_kv(v, self.cfg.group_size)
        n_blocks = math.ceil(t_k / self.block)
        scores = []
        for b in range(n_blocks):
            lo, hi = b * self.block, min(t_k, (b + 1) * self.block)
            contrib = torch.matmul(weights[..., lo:hi], v_rep[..., lo:hi, :])
            score = 2.0 * (out * contrib).sum(-1) - contrib.square().sum(-1)
            scores.append(score)
        score = torch.stack(scores, dim=-1)
        qpos = torch.arange(t_q, device=q.device).unsqueeze(1) + offset
        block_start = torch.arange(n_blocks, device=q.device) * self.block
        visible = block_start.view(1, 1, 1, -1) <= qpos.view(1, 1, -1, 1)
        score = score.masked_fill(~visible, float("-inf"))
        keep = max(1, min(self.topk, n_blocks))
        idx = torch.topk(score, keep, dim=-1).indices
        chosen = torch.zeros(q.shape[0], q.shape[1], t_q, n_blocks,
                             dtype=torch.bool, device=q.device)
        chosen.scatter_(-1, idx, True)
        allow = chosen.repeat_interleave(self.block, dim=-1)[..., :t_k]
        allow &= _causal_token_mask(q, t_k, offset)
        if self.cfg.sink_size:
            allow |= (torch.arange(t_k, device=q.device) < self.cfg.sink_size).view(1, 1, 1, -1)
        self._extra_stats = {"blocks": float(keep), "masa_oracle": 1.0}
        return self._finish(allow)

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(min(seq_len, self.topk * self.block + self.cfg.sink_size))

    def selection_flops(self, seq_len: int, batch: int = 1) -> int:
        h, d = self.cfg.n_heads, self.cfg.head_dim
        blocks = (seq_len + self.block - 1) // self.block
        dense = 2 * 2 * batch * h * seq_len * seq_len * d
        correction = 2 * batch * h * seq_len * blocks * d
        return int(dense + correction)


# --------------------------------------------------------------------------------------
# JSSA (2026)
# --------------------------------------------------------------------------------------
class JSSAAttention(_FrontierSelection):
    """Joint Subspace Sparse Attention: rank tokens in a shared QK subspace.

    Song et al., *Joint Subspace Sparse Attention for Efficient Long-Context
    Decoding* (2026), https://arxiv.org/abs/2610.19026

    Faithful core: learn a shared low-dimensional query--key subspace and use it for
    token ranking, then run exact attention over selected tokens.  The reference
    derives the subspace from the leading key covariance directions; the paper adds
    a score-preservation objective and a gather-free kernel.  Those training and
    kernel details are not reproduced.
    """

    name = "jssa"
    trained = True
    paper = ("Song et al., JSSA: Joint Subspace Sparse Attention (2026)",
             "https://arxiv.org/abs/2610.19026")

    def __init__(self, cfg: Optional[AttnConfig] = None, **kw: Any):
        super().__init__(cfg, **kw)
        self.rank = max(1, int(self.cfg.extras.get("rank", min(16, self.cfg.head_dim))))

    def select(self, q: torch.Tensor, state: Dict[str, Any], offset: int) -> torch.Tensor:
        k = state["k"]
        k_rep = repeat_kv(k, self.cfg.group_size)
        t_q, t_k, d = q.shape[-2], k.shape[-2], q.shape[-1]
        cov = torch.einsum("bhtd,bhte->bhde", k_rep, k_rep)
        _, vecs = torch.linalg.eigh(cov)
        rank = min(self.rank, d)
        basis = vecs[..., -rank:]
        qp = torch.matmul(q, basis)
        kp = torch.matmul(k_rep, basis)
        score = torch.matmul(qp, kp.transpose(-1, -2)) / math.sqrt(d)
        causal = _causal_token_mask(q, t_k, offset)
        allow = _top_tokens(score, self.cfg.topk_tokens, causal)
        if self.cfg.sink_size:
            allow |= (torch.arange(t_k, device=q.device) < self.cfg.sink_size).view(1, 1, 1, -1)
        self._extra_stats = {"subspace_rank": float(rank)}
        return self._finish(allow & causal)

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(min(seq_len, self.cfg.topk_tokens + self.cfg.sink_size))

    def selection_flops(self, seq_len: int, batch: int = 1) -> int:
        h, d, r = self.cfg.n_heads, self.cfg.head_dim, min(self.rank, self.cfg.head_dim)
        cov = 2 * batch * h * seq_len * d * d
        eig = 4 * batch * h * d ** 3
        rank_scores = 2 * 2 * batch * h * seq_len * seq_len * r
        return int(cov + eig + rank_scores)


# --------------------------------------------------------------------------------------
# OVAL (2026)
# --------------------------------------------------------------------------------------
class OVALAttention(_FrontierSelection):
    """Output-aware local page bases for sparse KV retrieval.

    Shahbazi et al., *OVAL: Output-Aware Local Page Bases for KV Cache Retrieval*
    (2026), https://arxiv.org/abs/2610.06686

    Faithful core: centre each page's keys and values, form
    ``M_eta = (1-eta)c_k X^T X + eta c_o X^T Y Y^T X``, retain its leading
    eigenvectors, and score pages with the page mean plus LogSumExp of the projected
    key logits.  The reference uses a per-head output projector for ``G``; this
    implementation uses ``G=I`` and a page-local SVD, which preserves the mechanism
    while avoiding an unstated model-specific ``W_O``.
    """

    name = "oval"
    trained = False
    paper = ("Shahbazi et al., OVAL: Output-Aware Local Page Bases (2026)",
             "https://arxiv.org/abs/2610.06686")

    def __init__(self, cfg: Optional[AttnConfig] = None, **kw: Any):
        super().__init__(cfg, **kw)
        self.page = int(self.cfg.extras.get("page_size", self.cfg.block_size))
        self.rank = max(1, int(self.cfg.extras.get("rank", 8)))
        self.eta = float(self.cfg.extras.get("eta", 0.5))

    def _summaries(self, state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        k, v = state["k"], state["v"]
        b, hkv, t_k, d = k.shape
        n_pages = math.ceil(t_k / self.page)
        mus, bases, residuals = [], [], []
        for p in range(n_pages):
            lo, hi = p * self.page, min(t_k, (p + 1) * self.page)
            kp = k[:, :, lo:hi, :]
            vp = v[:, :, lo:hi, :]
            mu = kp.mean(dim=-2)
            x = kp - mu.unsqueeze(-2)
            y = vp - vp.mean(dim=-2, keepdim=True)
            t = torch.matmul(x, x.transpose(-1, -2))
            w = torch.matmul(y, y.transpose(-1, -2))
            eps = 1e-8
            ck = (torch.diagonal(t, dim1=-2, dim2=-1).sum(-1) + eps).reciprocal()
            co = ((w * t).sum(dim=(-1, -2)) + eps).reciprocal()
            eye = torch.eye(hi - lo, dtype=k.dtype, device=k.device)
            a = ((1.0 - self.eta) * ck.view(b, hkv, 1, 1) * eye +
                 self.eta * co.view(b, hkv, 1, 1) * w)
            evals, evecs = torch.linalg.eigh(a)
            sqrt_a = torch.matmul(evecs * evals.clamp_min(0).sqrt().unsqueeze(-2),
                                  evecs.transpose(-1, -2))
            c = torch.matmul(sqrt_a, x)
            _, _, vh = torch.linalg.svd(c, full_matrices=False)
            r = min(self.rank, hi - lo, d)
            basis = vh[..., :r, :].transpose(-1, -2)
            mus.append(mu)
            bases.append(basis)
            residuals.append(torch.matmul(x, basis))
        return torch.stack(mus, -2), torch.stack(bases, -3), torch.stack(residuals, -3), n_pages

    def select(self, q: torch.Tensor, state: Dict[str, Any], offset: int) -> torch.Tensor:
        k = state["k"]
        t_q, t_k, d = q.shape[-2], k.shape[-2], q.shape[-1]
        mu, basis, residual, n_pages = self._summaries(state)
        g = self.cfg.group_size
        mu_q = _expand_head_groups(mu, g)
        basis_q = _expand_head_groups(basis, g)
        residual_q = _expand_head_groups(residual, g)
        mean_score = torch.einsum("bhtd,bhpd->bhtp", q, mu_q) / math.sqrt(d)
        qb = torch.einsum("bhtd,bhpdr->bhtpr", q, basis_q)
        z = torch.einsum("bhpmr,bhtpr->bhtpm", residual_q, qb)
        score = mean_score + torch.logsumexp(z, dim=-1)
        qpos = torch.arange(t_q, device=q.device).unsqueeze(1) + offset
        starts = torch.arange(n_pages, device=q.device) * self.page
        visible = starts.view(1, 1, 1, -1) <= qpos.view(1, 1, -1, 1)
        score = score.masked_fill(~visible, float("-inf"))
        idx = torch.topk(score, max(1, min(self.cfg.topk_blocks, n_pages)), dim=-1).indices
        chosen = torch.zeros(q.shape[0], q.shape[1], t_q, n_pages,
                             dtype=torch.bool, device=q.device)
        chosen.scatter_(-1, idx, True)
        allow = chosen.repeat_interleave(self.page, dim=-1)[..., :t_k]
        allow &= _causal_token_mask(q, t_k, offset)
        if self.cfg.sink_size:
            allow |= (torch.arange(t_k, device=q.device) < self.cfg.sink_size).view(1, 1, 1, -1)
        self._extra_stats = {"page_rank": float(min(self.rank, self.page)),
                             "oval_eta": self.eta}
        return self._finish(allow)

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(min(seq_len, self.cfg.topk_blocks * self.page + self.cfg.sink_size))

    def selection_flops(self, seq_len: int, batch: int = 1) -> int:
        pages = math.ceil(seq_len / self.page)
        m, d, r = min(self.page, seq_len), self.cfg.head_dim, min(self.rank, self.page)
        summary = batch * self.cfg.n_kv_heads * pages * (2 * m * m * d + 2 * m * d * r)
        score = 2 * batch * self.cfg.n_heads * seq_len * pages * (d + m * r)
        return int(summary + score)


# --------------------------------------------------------------------------------------
# SFA / feature sparsity (ICLR 2026)
# --------------------------------------------------------------------------------------
class SFAAttention(AttentionVariant):
    """Sparse Feature Attention: sparsify the feature axis, not the token axis.

    Xie et al., *Scaling Attention via Feature Sparsity* (ICLR 2026),
    https://arxiv.org/abs/2603.22300

    Faithful core: represent queries and keys as ``k``-sparse codes and keep every
    token.  The reference learns the codes and provides FlashSFA; here the code is
    the top-``k`` magnitude coordinates, which is the inference-time semantics and
    gives an exact reduction to dense attention when ``k == d``.
    """

    name = "sfa"
    family = "frontier"
    trained = True
    storage_precision = "sparse fp16 KV"
    paper = ("Xie et al., Sparse Feature Attention (ICLR 2026)",
             "https://arxiv.org/abs/2603.22300")

    def __init__(self, cfg: Optional[AttnConfig] = None, **kw: Any):
        super().__init__(cfg, **kw)
        self.keep = max(1, min(int(self.cfg.extras.get("feature_keep", self.cfg.head_dim)),
                               self.cfg.head_dim))

    def _sparsify(self, x: torch.Tensor) -> torch.Tensor:
        if self.keep >= x.shape[-1]:
            return x
        idx = torch.topk(x.abs(), self.keep, dim=-1).indices
        out = torch.zeros_like(x)
        out.scatter_(-1, idx, x.gather(-1, idx))
        return out

    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        qs = self._sparsify(q)
        ks = self._sparsify(k)
        allow = _causal_token_mask(q, k.shape[-2], offset)
        self._reads = float(k.shape[-2])
        return dense_attention(qs, ks, v, causal=False, mask=allow)

    def stats(self) -> Dict[str, float]:
        return {"reads_per_query": self._reads, "feature_keep": float(self.keep)}

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(seq_len)

    def memory_bytes(self, seq_len: int, batch: int = 1) -> int:
        d = self.cfg.head_dim
        values = 2 * batch * self.cfg.n_kv_heads * seq_len * d
        sparse_keys = batch * self.cfg.n_kv_heads * seq_len * (2 * self.keep + 4)
        return int(values + sparse_keys)

    def flops(self, seq_len: int, batch: int = 1) -> int:
        h = self.cfg.n_heads
        return int(2 * 2 * batch * h * seq_len * seq_len * self.keep)


# --------------------------------------------------------------------------------------
# Output-aware eviction: DropKV / ReST-KV / BeaconKV
# --------------------------------------------------------------------------------------
class DropKVAttention(EvictionAttention):
    """DropKV: evict tokens by decoupled residual-output perturbation.

    Zhang et al., *DropKV: Decoupling Residual-Output Perturbation for Near-Optimal
    KV-Cache Eviction* (NeurIPS 2026), https://arxiv.org/abs/2610.22130

    For each observation-window query, the score is
    ``(p_j / (1-p_j))^2 * ||a - v_j||^2``; tokens with the smallest score are
    evicted.  The paper's fused Triton scorer is not reproduced, but the reference
    computes the same quantity without materialising a ``w x n x d`` tensor.
    """

    name = "dropkv"
    family = "eviction"
    paper = ("Zhang et al., DropKV (NeurIPS 2026)", "https://arxiv.org/abs/2610.22130")

    def _perturbation_scores(self, state: Dict[str, Any]) -> torch.Tensor:
        weights = state["obs"]
        out = state["out_obs"]
        v_rep = repeat_kv(state["v"], self.cfg.group_size)
        p = weights.clamp(max=1.0 - 1e-6)
        dist = (out.square().sum(-1).unsqueeze(-1) +
                v_rep.square().sum(-1).unsqueeze(-2) -
                2.0 * torch.einsum("bhod,bhtd->bhot", out, v_rep))
        score = ((p / (1.0 - p)).square() * dist).sum(dim=-2)
        return score

    def _compaction_keep(self, state, budget: int) -> torch.Tensor:
        score = self._perturbation_scores(state)
        kt = state["k"].shape[-2]
        score = score.view(score.shape[0], self.cfg.n_kv_heads, self.cfg.group_size, kt).mean(2)
        pool = int(self.cfg.extras.get("pool_kernel", 11))
        if pool > 1:
            score = F.max_pool1d(score, pool, stride=1, padding=pool // 2)[..., :kt]
        n_recent = min(self.cfg.window_size, budget)
        n_sink = min(self.cfg.sink_size, budget - n_recent)
        return self._keep_top(-score, budget, kt, n_recent, n_sink)


class ReSTKVAttention(EvictionAttention):
    """ReST-KV: output reconstruction with spatial--temporal smoothing.

    An et al., *ReST-KV: Robust KV Cache Eviction with Layer-wise Output
    Reconstruction and Spatial-Temporal Smoothing* (ICLR 2026),
    https://arxiv.org/abs/2605.08840

    Faithful core: rank by the output discrepancy caused by removing a token,
    ``p_j ||a - v_j||^2``, smooth the score over time, and pool neighbouring token
    scores.  The layer-wise reconstruction solve and adaptive window are simplified
    to an EMA plus max-pooling; the paper's kernel is not reproduced.
    """

    name = "restkv"
    family = "eviction"
    paper = ("An et al., ReST-KV (ICLR 2026)", "https://arxiv.org/abs/2605.08840")

    def prepare_state(self, batch, device, dtype) -> Dict[str, Any]:
        state = super().prepare_state(batch, device, dtype)
        state["smooth"] = None
        return state

    def _compaction_keep(self, state, budget: int) -> torch.Tensor:
        weights, out = state["obs"], state["out_obs"]
        v_rep = repeat_kv(state["v"], self.cfg.group_size)
        dist = (out.square().sum(-1).unsqueeze(-1) +
                v_rep.square().sum(-1).unsqueeze(-2) -
                2.0 * torch.einsum("bhod,bhtd->bhot", out, v_rep))
        score = (weights * dist).sum(dim=-2)
        kt = state["k"].shape[-2]
        score = score.view(score.shape[0], self.cfg.n_kv_heads, self.cfg.group_size, kt).mean(2)
        if state["smooth"] is not None and state["smooth"].shape == score.shape:
            score = 0.7 * state["smooth"] + 0.3 * score
        state["smooth"] = score.detach()
        pool = int(self.cfg.extras.get("pool_kernel", 7))
        if pool > 1:
            score = F.max_pool1d(score, pool, stride=1, padding=pool // 2)[..., :kt]
        n_recent = min(self.cfg.window_size, budget)
        n_sink = min(self.cfg.sink_size, budget - n_recent)
        return self._keep_top(-score, budget, kt, n_recent, n_sink)


class BeaconKVAttention(EvictionAttention):
    """BeaconKV: keep compact representatives of historical query clusters.

    Kim et al., *BeaconKV: Key-Value Cache Compression Guided by Beacon Queries
    for Efficient Large Reasoning Model Inference* (ICML 2026),
    https://arxiv.org/abs/2609.04971

    The reference keeps one representative per query cluster so that a later
    "thought revisiting" query does not have to be predicted from the most recent
    window alone.  This implementation uses farthest-point representatives from the
    observation window, scores keys by their attention mass under those beacons, and
    retains the prompt, sinks and recent window.  The learned clustering and
    long-generation calibration are not reproduced.
    """

    name = "beaconkv"
    family = "eviction"
    paper = ("Kim et al., BeaconKV (ICML 2026)", "https://arxiv.org/abs/2609.04971")

    def __init__(self, cfg=None, **kw):
        super().__init__(cfg, **kw)
        self.n_beacons = max(1, int(self.cfg.extras.get("n_beacons", 4)))

    def prepare_state(self, batch, device, dtype) -> Dict[str, Any]:
        state = super().prepare_state(batch, device, dtype)
        state["beacons"] = None
        return state

    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        out = super().attend(q, state, offset)
        points = state["q_obs"]
        b, h, n, d = points.shape
        if n == 0:
            return out
        reps = [points[..., 0, :]]
        for _ in range(1, min(self.n_beacons, n)):
            stack = torch.stack(reps, dim=-2)
            dist = (points.unsqueeze(-2) - stack.unsqueeze(-3)).square().sum(-1).amin(-1)
            idx = dist.argmax(-1)
            reps.append(points.gather(2, idx.view(b, h, 1, 1).expand(b, h, 1, d)).squeeze(2))
        state["beacons"] = torch.stack(reps, dim=-2).detach()
        return out

    def _compaction_keep(self, state, budget: int) -> torch.Tensor:
        k = state["k"]
        kt = k.shape[-2]
        if state["beacons"] is None:
            score = torch.zeros(k.shape[0], self.cfg.n_kv_heads, kt, device=k.device, dtype=k.dtype)
        else:
            bq = state["beacons"]                         # already per query head
            k_rep = repeat_kv(k, self.cfg.group_size)
            logits = torch.einsum("bhjd,bhtd->bhjt", bq, k_rep) / math.sqrt(k.shape[-1])
            weights = torch.softmax(logits, dim=-1)
            score = weights.sum(-2)                         # (B,H,T), mass over beacons
            score = score.view(score.shape[0], self.cfg.n_kv_heads,
                               self.cfg.group_size, kt).mean(2)
        n_recent = min(self.cfg.window_size, budget)
        n_sink = min(self.cfg.sink_size, budget - n_recent)
        return self._keep_top(score, budget, kt, n_recent, n_sink)
