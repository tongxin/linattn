"""CSA2-style compressed sparse attention, as a single layer.

DeepSeek-V4.1-Flash (arXiv:2609.19969) describes CSA2 as three mechanisms composed:
compressed indexing, cross-layer KV reuse and FP4 KV storage.  This class implements
the **per-layer** part -- compression + low-precision storage + sparse selection --
and takes a ``cross_layer_group`` knob that only affects the memory accounting (the
multi-layer reuse itself lives in ``kv/reuse.py`` and ``kv/csa2.py``).

Mechanism
---------

* keys/values are mean-pooled into groups of ``compress_ratio`` tokens;
* the pooled cache is quantised to ``bits`` (4/8/16) with per-group scales -- the
  *stored* bytes are the quantised ones, and attention sees the dequantised values,
  so quantisation error appears in the output instead of being hidden;
* the lightning indexer scores pooled groups; each query reads its top-k groups;
* a short **raw** window handles the current tokens (a pooled group that contains
  future tokens would leak, so only *fully past* groups are eligible).  The window
  and the groups are concatenated into one key set and attended in a single softmax,
  so the union is exact rather than a post-hoc mixture.

What this is not: the released CSA2 layer.  There is no trained checkpoint, no exact
layer schedule, and the FP4 path is a simulated codec (``kv/quant.py``) rather than a
hardware kernel.  See ``docs/kv-layer.md`` for the honest mapping.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from linattn.core import AttnConfig, block_reduce, causal_mask, dense_attention
from linattn.kv.quant import QuantSpec, dequantize, quantize
from linattn.variants.selection import BlockSelectionAttention

__all__ = ["CSA2Attention"]


class CSA2Attention(BlockSelectionAttention):
    """Compressed, low-precision, sparsely-selected attention (one layer)."""

    name = "csa2"
    family = "native"
    trained = True
    paper = ("DeepSeek-AI, DeepSeek-V4.1-Flash: CSA2 + FP4 KV (2026)",
             "https://arxiv.org/abs/2609.19969")

    def __init__(self, cfg: Optional[AttnConfig] = None, **kw: Any):
        super().__init__(cfg, **kw)
        self.compress_ratio = int(self.cfg.extras.get("compress_ratio", self.cfg.compress_ratio))
        self.topk_groups = int(self.cfg.extras.get("topk_groups", self.cfg.topk_blocks))
        self.bits = int(self.cfg.extras.get("bits", 4))
        self.quant_group = int(self.cfg.extras.get("quant_group", 32))
        self.cross_layer_group = int(self.cfg.extras.get("cross_layer_group", 1))
        self.window = int(self.cfg.extras.get("window", self.cfg.window_size))
        self.n_indexer_heads = int(self.cfg.extras.get("indexer_heads", 4))
        self.indexer_dim = int(self.cfg.extras.get("indexer_dim", 16))
        self.quant = QuantSpec(bits=self.bits, group_size=self.quant_group)
        gen = torch.Generator().manual_seed(self.cfg.seed)
        self.q_proj = torch.nn.Parameter(
            torch.randn(self.cfg.n_heads, self.n_indexer_heads, self.cfg.head_dim, self.indexer_dim,
                        generator=gen) * 0.02
        )
        self.head_weights = torch.nn.Parameter(torch.full((self.n_indexer_heads,), 1.0 / self.n_indexer_heads))
        self._extra: Dict[str, float] = {}

    @property
    def storage_precision(self) -> str:
        return f"fp{self.bits} compressed + fp16 window"

    # -- storage ------------------------------------------------------------------
    def compressed_kv(self, k: torch.Tensor, v: torch.Tensor):
        """Pool to groups, then quantise the *stored* copy (dequantised for compute)."""
        kc = block_reduce(k, self.compress_ratio, -2)
        vc = block_reduce(v, self.compress_ratio, -2)
        if self.quant.enabled:
            # the codec works in fp32; restore the caller's dtype afterwards
            kc = dequantize(*quantize(kc, self.quant), self.quant).to(k.dtype)
            vc = dequantize(*quantize(vc, self.quant), self.quant).to(v.dtype)
        return kc, vc

    def indexer_scores(self, q: torch.Tensor, kc: torch.Tensor) -> torch.Tensor:
        b, h, t_q, d = q.shape
        t_g = kc.shape[-2]
        proj = self.q_proj.to(q.dtype)
        q_idx = torch.einsum("bhtd,hedf->bhtef", q, proj)
        k_rep = kc.unsqueeze(2).expand(b, self.cfg.n_kv_heads, self.cfg.group_size, t_g, d)
        k_idx = torch.einsum("bhtd,hedf->bhtef", k_rep.reshape(b, h, t_g, d), proj)
        dots = torch.einsum("bhtef,bhsef->bhtse", q_idx, k_idx)
        w = torch.softmax(self.head_weights.to(q.dtype), dim=-1)
        return (torch.relu(dots) * w.view(1, 1, 1, 1, -1)).sum(-1)      # (B,H,Tq,G)

    # -- attention ----------------------------------------------------------------
    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        b, h, t_q, d = q.shape
        t_k = k.shape[-2]
        m = self.compress_ratio
        kc, vc = self.compressed_kv(k, v)
        n_groups = kc.shape[-2]

        # Per query: the raw window is [p-w+1, p]; pooled groups are eligible only if
        # they end before that window (disjoint branches, so no token is counted
        # twice) and are fully in the past (a pooled group containing future tokens
        # would leak them into the pooled value).
        q_pos = torch.arange(t_q, device=q.device) + offset
        w = max(1, min(self.window, t_k))
        group_end = (torch.arange(n_groups, device=q.device) + 1) * m - 1
        eligible = (group_end.unsqueeze(0) <= q_pos.unsqueeze(1)) & (
            group_end.unsqueeze(0) < (q_pos.unsqueeze(1) - w + 1)
        )
        scores = self.indexer_scores(q, kc).masked_fill(~eligible.unsqueeze(0).unsqueeze(0), float("-inf"))
        keep = max(1, min(self.topk_groups, n_groups))
        sel = torch.topk(scores, keep, dim=-1).indices
        group_mask = torch.zeros(b, h, t_q, n_groups, dtype=torch.bool, device=q.device)
        group_mask.scatter_(-1, sel, True)
        # topk still returns indices when every score is -inf, so intersect with
        # eligibility: an ineligible (future / window-overlapping) group must not be read
        group_mask &= eligible.unsqueeze(0).unsqueeze(0)

        # raw local window, appended as extra keys so everything goes through one softmax
        kj = torch.arange(t_k, device=q.device)
        window_mask = ((kj.unsqueeze(0) <= q_pos.unsqueeze(1))
                       & ((q_pos.unsqueeze(1) - kj.unsqueeze(0)) < w)).expand(b, h, t_q, t_k)

        k_all = torch.cat([k, kc], dim=-2)
        v_all = torch.cat([v, vc], dim=-2)
        allow = torch.cat([window_mask, group_mask], dim=-1)
        out = dense_attention(q, k_all, v_all, causal=False, mask=allow)

        groups = float(group_mask.sum(-1).float().mean())
        window_reads = float(window_mask.sum(-1).float().mean())
        self._extra = {
            "groups_per_query": groups,
            "tokens_equivalent": groups * m + window_reads,
            "bits": float(self.bits),
        }
        self._reads = groups + window_reads
        return out

    # -- accounting ---------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        out = {"reads_per_query": self._reads}
        out.update(self._extra)
        return out

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        groups = max(1, min(self.topk_groups, (seq_len + self.compress_ratio - 1) // self.compress_ratio))
        return float(groups + min(self.window, seq_len))

    def memory_bytes(self, seq_len: int, batch: int = 1) -> int:
        n_groups = (seq_len + self.compress_ratio - 1) // self.compress_ratio
        compressed_values = 2 * self.cfg.n_kv_heads * n_groups * self.cfg.head_dim
        raw_window_values = 2 * self.cfg.n_kv_heads * min(self.window, seq_len) * self.cfg.head_dim
        total = self.quant.bytes_for(compressed_values) + raw_window_values * 2
        return int(total * batch / self.cross_layer_group)

    def selection_flops(self, seq_len: int, batch: int = 1) -> int:
        n_groups = (seq_len + self.compress_ratio - 1) // self.compress_ratio
        return int(2 * batch * self.cfg.n_heads * seq_len * n_groups * self.indexer_dim)
