"""Attention variants from the academic literature, including lines no frontier
model adopted.

The industry-facing half of the survey asks "what ships". This module is the other
half: the ideas that shaped how the field *thinks* about attention -- low-rank
projections, random features, Nystrom approximations, structured sparsity, hashing,
routing, compressive memory, retrieval memory. Several predate the KV-cache era; some
were never adopted at scale. They are the cleanest statements of the trade-offs the
production systems later re-derive.

Four canonical milestones are implemented here: Linformer (low-rank KV, ICML'20),
Performer (random features, ICML'21), BigBird (window + random + global, NeurIPS'20)
and Reformer (LSH, ICLR'20). Three more were implemented and then *left out*, each for
a documented reason in ``docs/academic-attention.md``: Nystromformer (the Moore-Penrose
inverse of the landmark matrix is numerically unstable at small landmark counts -- our
causal and non-causal versions both blew the output up 4-10x on random inputs),
Longformer/HyperAttention/Infini/kNN memory (lower marginal value per line of code than
the modern set in ``modern.py``).

Each class declares whether the published method is causal. Where it is not, the
class sets ``default_causal = False`` and its docstring says so -- a reference
implementation that silently leaks the future would be worse than none.

``tests/test_academic.py`` checks degenerate configurations against exact attention
(one bucket, full window, identity projection, all landmarks); that is the strongest
evidence a reference implementation of an approximate method can offer.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch

from linattn.core import AttentionVariant, block_reduce, causal_mask, dense_attention, repeat_kv

__all__ = [
    "LinformerAttention",
    "PerformerAttention",
    "BigBirdAttention",
    "ReformerLSHAttention",
]


class _Academic(AttentionVariant):
    """Per-variant causal switch plus read accounting."""

    family = "academic"
    default_causal = True

    def __init__(self, cfg=None, **kw: Any):
        super().__init__(cfg, **kw)
        self.causal = bool(self.cfg.extras.get("causal", self.default_causal))
        self._reads = 0.0
        self._extra: Dict[str, float] = {}

    def stats(self) -> Dict[str, float]:
        out = {"reads_per_query": self._reads}
        out.update(self._extra)
        return out

    def _causal(self, t_q: int, t_k: int, offset: int, device) -> torch.Tensor:
        return causal_mask(t_q, t_k, offset=offset, device=device)


# --------------------------------------------------------------------------------------
# Linformer (ICML 2020)
# --------------------------------------------------------------------------------------
class LinformerAttention(_Academic):
    """Low-rank keys/values: ``softmax(Q (E K)^T) (F V)`` in ``O(T E d)``.

    Sun et al., *Linformer: Self-Attention with Linear Complexity* (ICML 2020),
    https://arxiv.org/abs/2006.04768

    **Not causal**: the projection mixes positions, so a query would see a
    low-rank summary of future keys. It is the canonical statement of the low-rank
    hypothesis that every later feature-axis compression inherits.
    """

    name = "linformer"
    default_causal = False
    paper = ("Sun et al., Linformer (ICML 2020)", "https://arxiv.org/abs/2006.04768")

    def __init__(self, cfg=None, **kw: Any):
        super().__init__(cfg, **kw)
        self.k_dim = int(self.cfg.extras.get("linformer_k", 64))
        self.identity = bool(self.cfg.extras.get("identity", False))
        self._cache: Dict[tuple, torch.Tensor] = {}

    def _projection(self, t_k: int, dtype, device) -> torch.Tensor:
        key = (t_k, dtype, str(device))
        if key in self._cache:
            return self._cache[key]
        if self.identity and t_k <= self.k_dim:
            proj = torch.zeros(self.k_dim, t_k, dtype=dtype, device=device)
            proj[:t_k, :t_k] = torch.eye(t_k, dtype=dtype, device=device)
        else:
            g = torch.Generator().manual_seed(self.cfg.seed + t_k)
            proj = torch.randn(self.k_dim, t_k, generator=g).to(dtype) / math.sqrt(t_k)
        self._cache[key] = proj
        return proj

    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        proj = self._projection(k.shape[-2], q.dtype, q.device)
        k_p = repeat_kv(torch.einsum("et,bhtd->bhed", proj, k), self.cfg.group_size)
        v_p = repeat_kv(torch.einsum("et,bhtd->bhed", proj, v), self.cfg.group_size)
        scores = torch.matmul(q, k_p.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        self._reads = float(self.k_dim)
        return torch.softmax(scores, dim=-1) @ v_p

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(self.k_dim)

    def flops(self, seq_len: int, batch: int = 1) -> int:
        return int(2 * 2 * batch * self.cfg.n_heads * seq_len * self.k_dim * self.cfg.head_dim)


# --------------------------------------------------------------------------------------
# Performer (ICML 2021)
# --------------------------------------------------------------------------------------
class PerformerAttention(_Academic):
    """FAVOR+ positive random features: linear time, causal, no attention matrix.

    Choromanski et al., *Rethinking Attention with Performers* (ICML 2021),
    https://arxiv.org/abs/2009.14794

    ``phi(x) = exp(-||x||^2/2) * exp(x W)`` with ``W ~ N(0, I)`` is an unbiased
    estimator of ``exp(q.k)`` without the ``exp(+||x||^2/2)`` blow-up of the original
    formulation. The causal path uses prefix sums of ``phi(k) v^T`` and ``phi(k)``, so
    cost is ``O(T m d)`` and the *reads* are the feature dimension, not the cache.
    """

    name = "performer"
    default_causal = True
    paper = ("Choromanski et al., Performer (ICML 2021)", "https://arxiv.org/abs/2009.14794")

    def __init__(self, cfg=None, **kw: Any):
        super().__init__(cfg, **kw)
        self.features = int(self.cfg.extras.get("features", 32))
        self._w: Optional[torch.Tensor] = None

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        if self._w is None or self._w.shape[0] != d:
            g = torch.Generator().manual_seed(self.cfg.seed)
            # x arrives already scaled by 1/sqrt(d), so ||x|| ~ 1 and W ~ N(0, I)
            self._w = torch.randn(d, self.features, generator=g)
        scale = torch.exp(-0.5 * x.square().sum(-1, keepdim=True)).clamp_min(1e-30)
        exponent = torch.matmul(x, self._w.to(x.dtype)).clamp(-20.0, 20.0)
        return torch.exp(exponent) * scale

    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        k_rep = repeat_kv(k, self.cfg.group_size)
        v_rep = repeat_kv(v, self.cfg.group_size)
        scale = math.sqrt(q.shape[-1])
        phi_q, phi_k = self._phi(q / scale), self._phi(k_rep / scale)   # (B,H,T,m)
        self._reads = float(self.features)
        if not self.causal:
            kv = torch.einsum("bhtm,bhtd->bhmd", phi_k, v_rep)
            num = torch.einsum("bhtm,bhmd->bhtd", phi_q, kv)
            den = torch.einsum("bhtm,bhm->bht", phi_q, phi_k.sum(-2)).unsqueeze(-1)
            return num / den.clamp_min(1e-9)
        # causal: prefix sums over keys, then gather at each query position.  Memory
        # is O(T m d) here; a kernel keeps only the running sum.
        cs_kv = (phi_k.unsqueeze(-1) * v_rep.unsqueeze(-2)).cumsum(dim=2)
        cs_k = phi_k.cumsum(dim=2)
        t_q, t_k = q.shape[-2], k.shape[-2]
        pos = (torch.arange(t_q, device=q.device) + offset).clamp(max=t_k - 1)
        sel_kv = cs_kv.index_select(2, pos)
        sel_k = cs_k.index_select(2, pos)
        num = torch.einsum("bhtm,bhtmd->bhtd", phi_q, sel_kv)
        den = torch.einsum("bhtm,bhtm->bht", phi_q, sel_k).unsqueeze(-1)
        return num / den.clamp_min(1e-9)

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(self.features)


# --------------------------------------------------------------------------------------
# Nystromformer (NeurIPS 2021)
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# BigBird (NeurIPS 2020)
# --------------------------------------------------------------------------------------
class BigBirdAttention(_Academic):
    """Window + random + global blocks as an explicit mask.

    Zaheer et al., *Big Bird* (NeurIPS 2020), https://arxiv.org/abs/2007.14062

    The pattern (not their block-sparse kernel) is what makes BigBird a universal
    approximator of sequence functions; the random component is what later work
    replaced with learned routing.
    """

    name = "bigbird"
    paper = ("Zaheer et al., Big Bird (NeurIPS 2020)", "https://arxiv.org/abs/2007.14062")

    def __init__(self, cfg=None, **kw: Any):
        super().__init__(cfg, **kw)
        self.window = int(self.cfg.extras.get("window", self.cfg.window_size))
        self.n_random = int(self.cfg.extras.get("n_random", 3))
        self.n_global = int(self.cfg.extras.get("n_global", 2))
        self._rand: Dict[tuple, torch.Tensor] = {}

    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        t_q, t_k, device = q.shape[-2], k.shape[-2], q.device
        qi = (torch.arange(t_q, device=device) + offset).unsqueeze(1)
        kj = torch.arange(t_k, device=device).unsqueeze(0)
        causal = (kj <= qi) if self.causal else torch.ones(t_q, t_k, dtype=torch.bool, device=device)
        allow = causal | (((qi - kj) < self.window) & causal)
        if self.n_random > 0:
            key = (t_k, t_q, str(device))
            if key not in self._rand:
                g = torch.Generator().manual_seed(self.cfg.seed)
                self._rand[key] = torch.stack(
                    [torch.randperm(t_k, generator=g)[: self.n_random] for _ in range(t_q)]
                ).to(device)
            rand = torch.zeros(t_q, t_k, dtype=torch.bool, device=device)
            rand.scatter_(-1, self._rand[key], True)
            allow = allow | (rand & causal)
        allow = (allow | (kj < self.n_global)).expand(q.shape[0], q.shape[1], t_q, t_k)
        self._reads = float(allow[0, 0].sum(-1).float().mean())
        return dense_attention(q, k, v, causal=False, mask=allow)

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(min(seq_len, self.window + self.n_random + self.n_global))


# --------------------------------------------------------------------------------------
# Reformer LSH (ICLR 2020)
# --------------------------------------------------------------------------------------
class ReformerLSHAttention(_Academic):
    """Angular-LSH buckets: a query reads the keys hashed into its bucket.

    Kitaev et al., *Reformer* (ICLR 2020), https://arxiv.org/abs/2001.04451

    This is the *semantics* (bucket-equality mask), not the chunked kernel; one
    bucket makes it exact, which the tests use. The hash is shared across heads, as a
    fixed random projection would be.
    """

    name = "reformer_lsh"
    paper = ("Kitaev et al., Reformer (ICLR 2020)", "https://arxiv.org/abs/2001.04451")

    def __init__(self, cfg=None, **kw: Any):
        super().__init__(cfg, **kw)
        self.buckets = int(self.cfg.extras.get("buckets", 8))
        self._proj: Optional[torch.Tensor] = None

    def _hash(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        if self._proj is None or self._proj.shape[0] != d:
            g = torch.Generator().manual_seed(self.cfg.seed)
            self._proj = torch.randn(d, max(1, self.buckets), generator=g)
        # random-projection argmax: with one bucket every token lands in bucket 0
        return torch.matmul(x, self._proj.to(x.dtype)).argmax(-1)

    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        k_rep = repeat_kv(k, self.cfg.group_size)
        v_rep = repeat_kv(v, self.cfg.group_size)
        same = self._hash(q).unsqueeze(-1) == self._hash(k_rep).unsqueeze(-2)
        # a bucket may be empty for a query (especially under causal masking), so the
        # query always keeps itself -- the standard "window before chunk" fallback
        t_q, t_k = q.shape[-2], k.shape[-2]
        own = torch.zeros(t_q, t_k, dtype=torch.bool, device=q.device)
        pos = (torch.arange(t_q, device=q.device) + offset).clamp(max=t_k - 1)
        own.scatter_(-1, pos.unsqueeze(-1), True)
        same = same | own.unsqueeze(0).unsqueeze(0)
        if self.causal:
            same = same & self._causal(q.shape[-2], k.shape[-2], offset, q.device)
        out = dense_attention(q, k_rep, v_rep, causal=False, mask=same)
        self._reads = float(same[0, 0].sum(-1).float().mean())
        return out

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(max(1.0, seq_len / max(1, self.buckets)))
