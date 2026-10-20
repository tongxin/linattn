"""Multi-head Latent Attention (MLA): the compressed KV state DeepSeek ships.

DeepSeek-AI, *DeepSeek-V2* (2024), https://arxiv.org/abs/2405.04434

MLA is not a sparsity method: it shrinks every cached KV *vector* to a shared
low-rank latent ``c`` (512 values per token in DeepSeek-V2/V3 instead of
``2 * n_heads * head_dim``), and materialises per-head keys and values from that
latent inside the attention kernel.  It is the compressed-state branch of the
taxonomy, and it is a prerequisite for understanding DSA/CSA: DSA's indexer runs
*on top of* MLA, and the V4 lineage compresses the latent further with FP4 and
cross-layer reuse.

The class here takes the latent as its ``k`` input (``n_kv_heads == 1`` is the
natural setting) and reports the cache-footprint reduction, which is the number
that matters at serving time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from linattn.core import AttnConfig, AttentionVariant, causal_mask, dense_attention

__all__ = ["MLAttention"]


class MLAttention(AttentionVariant):
    """Latent KV cache with per-head up-projections."""

    name = "mla"
    family = "compressed"
    storage_precision = "fp16 latent"
    paper = ("DeepSeek-AI, DeepSeek-V2 (2024)", "https://arxiv.org/abs/2405.04434")

    def __init__(self, cfg: Optional[AttnConfig] = None, **kw):
        super().__init__(cfg, **kw)
        self.latent_dim = int(self.cfg.extras.get("latent_dim", self.cfg.head_dim // 4))
        self.w_k: Optional[torch.nn.Parameter] = None
        self.w_v: Optional[torch.nn.Parameter] = None
        self._build_projections()

    def _build_projections(self) -> None:
        gen = torch.Generator().manual_seed(self.cfg.seed)
        self.w_k = torch.nn.Parameter(
            torch.randn(self.cfg.n_heads, self.latent_dim, self.cfg.head_dim, generator=gen) * 0.05
        )
        self.w_v = torch.nn.Parameter(
            torch.randn(self.cfg.n_heads, self.latent_dim, self.cfg.head_dim, generator=gen) * 0.05
        )

    def materialize(self, latent: torch.Tensor):
        """``(B, 1, T, d_c)`` -> per-head keys and values ``(B, H, T, d_h)``."""
        if latent.shape[-1] != self.latent_dim:
            # the caller's latent dimension wins: in a harness that only has q/k/v,
            # the "latent" is whatever is cached, and the up-projection adapts
            self.latent_dim = latent.shape[-1]
            self._build_projections()
        c = latent[:, 0]                                              # (B, T, d_c)
        k = torch.einsum("btd,hde->bhte", c, self.w_k.to(c.dtype))    # (B,H,T,d_h)
        v = torch.einsum("btd,hde->bhte", c, self.w_v.to(c.dtype))
        return k, v

    def prepare_state(self, batch: int, device, dtype) -> Dict[str, Any]:
        """The cache holds *latents*, so its last dimension is ``latent_dim``."""
        return {
            "k": torch.zeros(batch, self.cfg.n_kv_heads, 0, self.latent_dim, device=device, dtype=dtype),
            "v": torch.zeros(batch, self.cfg.n_kv_heads, 0, self.latent_dim, device=device, dtype=dtype),
            "pos": 0,
        }

    def push_kv(self, state, k, v):
        if state["k"].shape[-2] == 0 and state["k"].shape[-1] != k.shape[-1]:
            self.latent_dim = k.shape[-1]
            self._build_projections()
            state["k"] = torch.zeros_like(k[..., :0, :])
            state["v"] = torch.zeros_like(v[..., :0, :])
        return super().push_kv(state, k, v)

    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = self.materialize(state["k"])
        allow = causal_mask(q.shape[-2], k.shape[-2], offset=offset, device=q.device)
        return dense_attention(q, k, v, causal=False, mask=allow)

    def memory_bytes(self, seq_len: int, batch: int = 1) -> int:
        """Cached latent only: ``T * d_c`` values per token (no per-head fan-out)."""
        return int(batch * seq_len * self.latent_dim * 2)

    def dense_equivalent_bytes(self, seq_len: int, batch: int = 1) -> int:
        return int(batch * self.cfg.n_kv_heads * seq_len * self.cfg.head_dim * 2 * 2)

    def compression_ratio(self) -> float:
        return self.dense_equivalent_bytes(1024) / max(1, self.memory_bytes(1024))

    def flops(self, seq_len: int, batch: int = 1) -> int:
        h, d = self.cfg.n_heads, self.cfg.head_dim
        core = 2 * 2 * batch * h * seq_len * (seq_len + 1) / 2 * d
        up = 2 * 2 * batch * seq_len * self.latent_dim * d * h
        return int(core + up)
