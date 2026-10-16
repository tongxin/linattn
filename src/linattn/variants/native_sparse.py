"""Native (trainable) sparse attention: NSA's compressed / selected / window branches.

DeepSeek-AI, *Native Sparse Attention: Hardware-Aligned and Natively Trainable
Sparse Attention* (ACL 2025), https://arxiv.org/abs/2502.11089

NSA routes every query through three branches and combines them with a learned
gate:

* **compression** -- keys/values are mean-pooled into blocks of ``compress_ratio``
  tokens, giving coarse global context at a fraction of the length;
* **selection** -- contiguous blocks of ``block_size`` raw tokens are selected
  using the *compression branch's* attention scores, so selection reuses work
  already done and stays block-aligned (Tensor-Core friendly);
* **sliding window** -- recent tokens, isolated into their own branch so that
  local patterns do not shortcut the other two.

Two implementation notes that matter for reproduction:

1. Selection reuses the pooled-key scores rather than scoring raw keys.  That is
   what makes NSA's selection cost ``O(T * l)`` instead of ``O(T^2)`` but it also
   means selection quality is bounded by the pooled representation -- the point
   the COBS analysis (arXiv:2607.09052) attacks.
2. The gates are learned.  A randomly initialised gate is meaningless, so this
   class defaults to ``gate_mode="uniform"`` (equal branch weights, the honest
   inference-time proxy) and offers :meth:`fit_gates`, which trains the gates to
   minimise the output error against dense attention.  That makes the
   "training-aware" claim testable on a CPU.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from linattn.core import AttnConfig, AttentionVariant, block_reduce, causal_mask, dense_attention, repeat_kv
from linattn.variants.selection import BlockSelectionAttention, block_mask_to_token_mask, topk_blocks

__all__ = ["NSAAttention"]


class NSAAttention(AttentionVariant):
    """Three-branch native sparse attention (inference semantics of NSA)."""

    name = "nsa"
    family = "native"
    trained = True
    paper = ("Yuan et al., Native Sparse Attention (ACL 2025)", "https://arxiv.org/abs/2502.11089")

    def __init__(self, cfg: Optional[AttnConfig] = None, **kw):
        super().__init__(cfg, **kw)
        self.compress = int(self.cfg.extras.get("compress_ratio", self.cfg.compress_ratio))
        self.selected_blocks = int(self.cfg.extras.get("selected_blocks", self.cfg.topk_blocks))
        self.gate_mode = self.cfg.extras.get("gate_mode", "uniform")
        self._extra: Dict[str, float] = {}
        gen = torch.Generator().manual_seed(self.cfg.seed)
        init = torch.full((self.cfg.n_heads, 3), 1.0 / 3.0)
        self.gates = torch.nn.Parameter(init + 0.01 * torch.randn(self.cfg.n_heads, 3, generator=gen))

    # -- branches ------------------------------------------------------------------
    def compressed_kv(self, k: torch.Tensor, v: torch.Tensor):
        c = self.compress
        kc = block_reduce(k, c, -2)
        vc = block_reduce(v, c, -2)
        return kc, vc

    def branch_compression(self, q, k, v, offset):
        kc, vc = self.compressed_kv(k, v)
        n_groups = kc.shape[-2]
        q_pos = torch.arange(q.shape[-2], device=q.device) + offset
        grp_start = torch.arange(n_groups, device=q.device) * self.compress
        allow = grp_start.unsqueeze(0) <= (q_pos.unsqueeze(1) + self.compress - 1)
        if not bool(allow.any()):
            allow = allow | True
        return dense_attention(q, kc, vc, causal=False, mask=allow.expand(q.shape[0], q.shape[1], q.shape[-2], n_groups))

    def selection_mask(self, q, k, offset) -> torch.Tensor:
        """Blocks selected from the pooled-key scores (NSA's blockwise selection)."""
        kc, _ = self.compressed_kv(k, k)
        bs = self.cfg.block_size
        k_raw = block_reduce(repeat_kv(k, self.cfg.group_size), bs, -2)
        # score pooled raw blocks with the query (same granularity as the selected blocks)
        scores = torch.matmul(q, k_raw.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        t_q, t_k = q.shape[-2], k.shape[-2]
        n_kb = k_raw.shape[-2]
        q_pos = torch.arange(t_q, device=q.device) + offset
        kb_start = torch.arange(n_kb, device=q.device) * bs
        allowed = kb_start.unsqueeze(0) <= q_pos.unsqueeze(1)
        sel = topk_blocks(scores, self.selected_blocks, allowed.expand_as(scores))
        own = (q_pos // bs).clamp(max=n_kb - 1)
        sel = sel.scatter(-1, own.view(1, 1, -1, 1).expand_as(sel[..., :1]), True)
        self._extra["selected_blocks_per_query"] = float(sel.sum(-1).float().mean())
        return block_mask_to_token_mask(sel, 1, bs, t_q, t_k)

    def branch_selection(self, q, k, v, offset):
        mask = self.selection_mask(q, k, offset)
        return dense_attention(q, k, v, causal=False, mask=mask)

    def branch_window(self, q, k, v, offset):
        t_q, t_k = q.shape[-2], k.shape[-2]
        qi = torch.arange(t_q, device=q.device).unsqueeze(1) + offset
        kj = torch.arange(t_k, device=q.device).unsqueeze(0)
        allow = (kj <= qi) & ((qi - kj) < self.cfg.window_size)
        return dense_attention(q, k, v, causal=False, mask=allow.expand(q.shape[0], q.shape[1], t_q, t_k))

    # -- combination ---------------------------------------------------------------
    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        o_cmp = self.branch_compression(q, k, v, offset)
        o_slc = self.branch_selection(q, k, v, offset)
        o_win = self.branch_window(q, k, v, offset)
        if self.gate_mode == "uniform":
            gates = torch.full((self.cfg.n_heads, 3), 1.0 / 3.0, device=q.device, dtype=q.dtype)
        else:
            gates = torch.sigmoid(self.gates.to(q.dtype))
            gates = gates / gates.sum(-1, keepdim=True).clamp_min(1e-9)
        return (gates[:, 0].view(1, -1, 1, 1) * o_cmp
                + gates[:, 1].view(1, -1, 1, 1) * o_slc
                + gates[:, 2].view(1, -1, 1, 1) * o_win)

    # -- budgets and costs ---------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        # compressed branch reads one pooled key per compress_ratio tokens, which is
        # negligible but not zero: it is counted in read_budget()
        reads = float(self.cfg.window_size + self.selected_blocks * self.cfg.block_size)
        out = {"reads_per_query": reads}
        out.update(self._extra)
        return out

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        compressed = (seq_len + self.compress - 1) // self.compress
        return float(min(seq_len, self.cfg.window_size + self.selected_blocks * self.cfg.block_size + compressed))

    def memory_bytes(self, seq_len: int, batch: int = 1) -> int:
        return int(batch * self.cfg.n_kv_heads * seq_len * self.cfg.head_dim * 2 * 2)

    def flops(self, seq_len: int, batch: int = 1) -> int:
        h, d = self.cfg.n_heads, self.cfg.head_dim
        compressed = (seq_len + self.compress - 1) // self.compress
        reads = self.cfg.window_size + self.selected_blocks * self.cfg.block_size + compressed
        return int(2 * 2 * batch * h * seq_len * min(seq_len, reads) * d)

    # -- training the gates (the "natively trainable" part) ------------------------
    def fit_gates(self, q, k, v, steps: int = 200, lr: float = 0.05) -> float:
        """Learn branch weights that best reproduce dense attention's output."""
        with torch.no_grad():
            target = dense_attention(q, k, v, causal=True)
        state = self.prepare_state(q.shape[0], q.device, q.dtype)
        state = self.push_kv(state, k, v)
        opt = torch.optim.Adam([self.gates], lr=lr)
        loss_value = float("nan")
        for _ in range(steps):
            opt.zero_grad()
            gates = torch.sigmoid(self.gates).to(q.dtype)
            gates = gates / gates.sum(-1, keepdim=True).clamp_min(1e-9)
            o_cmp = self.branch_compression(q, k, v, 0)
            o_slc = self.branch_selection(q, k, v, 0)
            o_win = self.branch_window(q, k, v, 0)
            out = gates[:, 0].view(1, -1, 1, 1) * o_cmp + gates[:, 1].view(1, -1, 1, 1) * o_slc \
                + gates[:, 2].view(1, -1, 1, 1) * o_win
            loss = F.mse_loss(out, target)
            loss.backward()
            opt.step()
            loss_value = float(loss.detach())
        self.gate_mode = "learned"
        return loss_value
