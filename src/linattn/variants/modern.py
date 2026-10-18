"""The current academic frontier (2025-2026): what researchers argue *now*.

The canon in :mod:`linattn.variants.academic` established the vocabulary. This module
covers the lines that are live in 2026 and that have no frontier-model landing yet:

=========================  ==========================================================
class                       claim in the paper
=========================  ==========================================================
``FreeSoloAttention``       a sliding window plus *keypoint cells* selected by score;
                            linear time with zero-shot length extrapolation
``NAMOHAttention``          activate K of H heads per token; each head owns a token
                            subset and attends causally within it
``HLAMixingAttention``      chunk-wise dynamic mixing between linear and full attention
``MassOracleAttention``     the COBS analysis: block selection *is* attention-mass
                            ranking; this is the exact-mass oracle upper bound
=========================  ==========================================================

These are implemented from the published descriptions (abstracts/papers), not from
released code. Each docstring says which part is faithful and which is an
approximation; the tests check invariants and degenerate cases, never claim parity
with the authors' kernels.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch

from linattn.core import AttentionVariant, block_reduce, causal_mask, dense_attention, repeat_kv

__all__ = ["FreeSoloAttention", "NAMOHAttention", "HLAMixingAttention", "MassOracleAttention"]


class _Modern(AttentionVariant):
    family = "modern"

    def __init__(self, cfg=None, **kw: Any):
        super().__init__(cfg, **kw)
        self._reads = 0.0
        self._extra: Dict[str, float] = {}

    def stats(self) -> Dict[str, float]:
        out = {"reads_per_query": self._reads}
        out.update(self._extra)
        return out


class FreeSoloAttention(_Modern):
    """Sliding window + keypoint cells, and nothing else.

    Shi & Price, *FreeSolo: Linear-time Sparse Attention with Zero-Shot Length
    Extrapolation* (2026), https://arxiv.org/abs/2610.17191

    Faithful part: the two branches (local window, long-range *cell* selection) and
    the cell-level keypoint score. Approximated part: the paper's FS-L prunes the
    prefix with a learned cell scorer and FS-B searches a balanced tree; here the
    scorer is ``q . mean(k_cell)`` and the selection is a top-k, which is the
    deployable core without the learned pruning.
    """

    name = "free_solo"
    paper = ("Shi & Price, FreeSolo (2026)", "https://arxiv.org/abs/2610.17191")

    def __init__(self, cfg=None, **kw: Any):
        super().__init__(cfg, **kw)
        self.cell = int(self.cfg.extras.get("cell", self.cfg.block_size))
        self.topk_cells = int(self.cfg.extras.get("topk_cells", self.cfg.topk_blocks))
        self.window = int(self.cfg.extras.get("window", self.cfg.window_size))

    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        t_q, t_k, d = q.shape[-2], k.shape[-2], q.shape[-1]
        k_rep = repeat_kv(k, self.cfg.group_size)
        v_rep = repeat_kv(v, self.cfg.group_size)
        n_cells = math.ceil(t_k / self.cell)
        k_mean = block_reduce(k_rep, self.cell, -2)
        scores = torch.matmul(q, k_mean.transpose(-1, -2)) / math.sqrt(d)       # (B,H,Tq,C)
        qi = (torch.arange(t_q, device=q.device) + offset).unsqueeze(1)
        cell_end = (torch.arange(n_cells, device=q.device) + 1) * self.cell - 1
        visible = cell_end.unsqueeze(0) <= qi
        scores = scores.masked_fill(~visible.unsqueeze(0).unsqueeze(0), float("-inf"))
        keep = max(1, min(self.topk_cells, n_cells))
        sel = torch.topk(scores, keep, -1).indices
        cell_mask = torch.zeros(q.shape[0], q.shape[1], t_q, n_cells, dtype=torch.bool, device=q.device)
        cell_mask.scatter_(-1, sel, True)
        cell_mask &= visible.unsqueeze(0).unsqueeze(0)
        token_mask = cell_mask.repeat_interleave(self.cell, dim=-1)[..., :t_k]
        kj = torch.arange(t_k, device=q.device)
        window = (kj.unsqueeze(0) <= qi) & ((qi - kj.unsqueeze(0)) < self.window)
        allow = token_mask | window.expand(q.shape[0], q.shape[1], t_q, t_k)
        self._reads = float(allow[0, 0].sum(-1).float().mean())
        self._extra = {"cells": float(keep)}
        return dense_attention(q, k_rep, v_rep, causal=False, mask=allow)

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(min(seq_len, self.window + self.topk_cells * self.cell))


class NAMOHAttention(_Modern):
    """Head-owned token subsets: each head attends only its assigned subsequence.

    Fu et al., *Scaling Parameter and Context in Attention: Native Sparse Attention
    from Mixture-of-Head* (2026), https://arxiv.org/abs/2609.38832

    NAMOH activates ``K`` of ``H`` heads per token and lets each head keep only its
    assigned tokens. Here the assignment is a deterministic stride (``token_index %
    H``) with a local window so every query has context; the paper learns the
    assignment. With ``H = 1`` this is exact causal attention, which the tests use.
    """

    name = "namoh"
    paper = ("Fu et al., NAMOH: native sparse attention from mixture-of-heads (2026)",
             "https://arxiv.org/abs/2609.38832")

    def __init__(self, cfg=None, **kw: Any):
        super().__init__(cfg, **kw)
        self.window = int(self.cfg.extras.get("window", self.cfg.window_size))
        self.assign = str(self.cfg.extras.get("assign", "stride"))

    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        t_q, t_k = q.shape[-2], k.shape[-2]
        h = self.cfg.n_heads
        k_rep = repeat_kv(k, self.cfg.group_size)
        v_rep = repeat_kv(v, self.cfg.group_size)
        qi = (torch.arange(t_q, device=q.device) + offset).unsqueeze(1)
        kj = torch.arange(t_k, device=q.device)
        causal = kj.unsqueeze(0) <= qi
        window = (qi - kj.unsqueeze(0)) < self.window
        if self.assign == "stride":
            owner = (kj % h).view(1, 1, 1, t_k) == torch.arange(h, device=q.device).view(1, h, 1, 1)
        else:
            g = torch.Generator().manual_seed(self.cfg.seed)
            perm = torch.randperm(h, generator=g)
            owner = (perm[kj % h]).view(1, 1, 1, t_k) == torch.arange(h, device=q.device).view(1, h, 1, 1)
        allow = (owner.expand(q.shape[0], h, t_q, t_k) | window.expand(q.shape[0], h, t_q, t_k)) & causal
        self._reads = float(allow[0, 0].sum(-1).float().mean())
        return dense_attention(q, k_rep, v_rep, causal=False, mask=allow)

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(min(seq_len, seq_len / max(1, self.cfg.n_heads) + self.window))


class HLAMixingAttention(_Modern):
    """Chunk-wise dynamic mixing of full attention and a linear state.

    Chen et al., *HLA: Expressive Hybrid Linear Attention via Chunk-Wise Dynamic
    Mixing* (2026), https://arxiv.org/abs/2610.05842

    Within a chunk the layer computes exact causal attention and, in parallel, a
    read from a linear-attention state written by previous chunks; a *data-dependent*
    scalar gate (from the chunk's mean query norm) mixes the two. This is the
    Infini-attention family with the gate made per-chunk and data-dependent, which is
    HLA's stated contribution.
    """

    name = "hla_mixing"
    paper = ("Chen et al., HLA: hybrid linear attention via chunk-wise dynamic mixing (2026)",
             "https://arxiv.org/abs/2610.05842")

    def __init__(self, cfg=None, **kw: Any):
        super().__init__(cfg, **kw)
        self.tau = float(self.cfg.extras.get("gate_tau", 1.0))

    def prepare_state(self, batch: int, device, dtype) -> Dict[str, Any]:
        state = super().prepare_state(batch, device, dtype)
        d = self.cfg.head_dim
        state["S"] = torch.zeros(batch, self.cfg.n_heads, d, d, device=device, dtype=dtype)
        state["z"] = torch.zeros(batch, self.cfg.n_heads, d, device=device, dtype=dtype)
        return state

    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        t_q, t_k = q.shape[-2], k.shape[-2]
        k_chunk = k[:, :, t_k - t_q :, :] if t_q <= t_k else k
        v_chunk = v[:, :, t_k - t_q :, :] if t_q <= t_k else v
        k_rep = repeat_kv(k_chunk, self.cfg.group_size)
        v_rep = repeat_kv(v_chunk, self.cfg.group_size)
        local = dense_attention(q, k_rep, v_rep, causal=True)
        memory_num = torch.einsum("bhtd,bhde->bhte", q, state["S"])
        memory_den = torch.einsum("bhtd,bhd->bht", q, state["z"]).unsqueeze(-1)
        memory = memory_num / memory_den.abs().clamp_min(1e-6)
        gate = torch.sigmoid(q.abs().mean(dim=(-1, -2), keepdim=True) / self.tau)   # (B,H,1,1)
        self._reads = float(min(t_q, t_k) + self.cfg.head_dim)
        self._extra = {"gate": float(gate.mean())}
        return gate * local + (1.0 - gate) * memory

    def finalize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        k, v = state["k"], state["v"]
        k_rep = repeat_kv(k, self.cfg.group_size)
        v_rep = repeat_kv(v, self.cfg.group_size)
        state["S"] = state["S"] + torch.einsum("bhtd,bhte->bhde", k_rep, v_rep)
        state["z"] = state["z"] + k_rep.sum(-2)
        return state

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(seq_len + self.cfg.head_dim)      # chunk-local exact attention

    def memory_bytes(self, seq_len: int, batch: int = 1) -> int:
        d = self.cfg.head_dim
        return int(4 * batch * self.cfg.n_heads * d * d + 4 * batch * self.cfg.n_heads * d)


class MassOracleAttention(_Modern):
    """The COBS oracle: select blocks by their *exact* attention mass.

    Tian et al., *COBS: Cumulant Order Block Sparse Attention* (2026),
    https://arxiv.org/abs/2607.09052

    COBS formalises block selection as ranking blocks by attention mass and shows
    that *if* selection were exact, block-sparse attention would match dense. This
    class implements that oracle: it computes dense attention to measure each block's
    mass, then reads only the top blocks. It is an **analysis instrument**, not a
    method -- it peeks at the answer -- and exists to bound what any selector could
    achieve at a given budget.
    """

    name = "mass_oracle"
    trained = False
    paper = ("Tian et al., COBS: block selection is attention-mass ranking (2026)",
             "https://arxiv.org/abs/2607.09052")

    def __init__(self, cfg=None, **kw: Any):
        super().__init__(cfg, **kw)
        self.block = int(self.cfg.extras.get("block", self.cfg.block_size))
        self.topk = int(self.cfg.extras.get("topk_blocks", self.cfg.topk_blocks))
        self.sink = int(self.cfg.extras.get("sinks", self.cfg.sink_size))

    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        t_q, t_k = q.shape[-2], k.shape[-2]
        with torch.no_grad():
            weights = dense_attention(q, k, v, causal=True, return_weights=True)[1]   # (B,H,Tq,Tk)
            n_blocks = math.ceil(t_k / self.block)
            mass = block_reduce(weights, self.block, -1)[..., :n_blocks]
            keep = max(1, min(self.topk, n_blocks))
            sel = torch.topk(mass, keep, -1).indices
            chosen = torch.zeros(q.shape[0], q.shape[1], t_q, n_blocks, dtype=torch.bool, device=q.device)
            chosen.scatter_(-1, sel, True)
        mask = chosen.repeat_interleave(self.block, dim=-1)[..., :t_k]
        mask = mask | (torch.arange(t_k, device=q.device) < self.sink).view(1, 1, 1, t_k)
        mask = mask & causal_mask(t_q, t_k, offset=offset, device=q.device)
        self._reads = float(mask[0, 0].sum(-1).float().mean())
        return dense_attention(q, k, v, causal=False, mask=mask)

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(min(seq_len, self.topk * self.block + self.sink))
