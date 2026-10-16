"""Structural / local sparsity: sliding window attention and attention sinks.

This is the cheapest family and the one that is actually deployed almost
everywhere: streaming heads in long-context models attend to a bounded window,
and the first few tokens act as a "sink" that absorbs probability mass.  The
implementation here is the *oracle* version (an explicit boolean mask), which
keeps it exact and makes it usable as the local branch inside the more complex
variants.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from linattn.core import AttentionVariant, causal_mask, dense_attention

__all__ = ["SlidingWindowAttention", "StreamingSinkAttention"]


class SlidingWindowAttention(AttentionVariant):
    """Causal sliding window of ``window_size`` tokens plus ``sink_size`` leading tokens.

    References
    ----------
    Beltagy et al., *Longformer* (2020), https://arxiv.org/abs/2004.05150
    Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (2024),
    https://arxiv.org/abs/2309.17453
    """

    name = "sliding_window"
    family = "local"
    paper = ("Xiao et al., StreamingLLM: attention sinks (ICLR 2024)", "https://arxiv.org/abs/2309.17453")

    def __init__(self, cfg=None, **kw):
        super().__init__(cfg, **kw)
        self._reads = 0.0

    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        k, v = state["k"], state["v"]
        t_q, t_k = q.shape[-2], k.shape[-2]
        allow = self.window_mask(t_q, t_k, offset, q.device)
        self._reads = float(allow.sum(-1).float().mean())
        return dense_attention(q, k, v, causal=False, mask=allow)

    def window_mask(self, t_q: int, t_k: int, offset: int, device) -> torch.Tensor:
        qi = torch.arange(t_q, device=device).unsqueeze(1) + offset
        kj = torch.arange(t_k, device=device).unsqueeze(0)
        causal = kj <= qi
        recent = (qi - kj) < self.cfg.window_size
        sink = kj < self.cfg.sink_size
        return causal & (recent | sink)

    def stats(self) -> Dict[str, float]:
        return {"reads_per_query": self._reads}

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        return float(min(seq_len, self.cfg.window_size + self.cfg.sink_size))

    def memory_bytes(self, seq_len: int, batch: int = 1) -> int:
        # a real streaming implementation keeps only window+sink entries
        kept = min(seq_len, self.cfg.window_size + self.cfg.sink_size)
        return int(batch * self.cfg.n_kv_heads * kept * self.cfg.head_dim * 2 * 2)


class StreamingSinkAttention(SlidingWindowAttention):
    """Attention sinks + window, restricted to exactly ``window_size`` recent tokens.

    Same computation as :class:`SlidingWindowAttention`; kept as a separate name
    because the literature uses "StreamingLLM-style" for the eviction contract
    (drop evicted tokens permanently) rather than for a mask applied at read time.
    """

    name = "streaming_sink"
    paper = ("Xiao et al., StreamingLLM (ICLR 2024)", "https://arxiv.org/abs/2309.17453")

    def memory_bytes(self, seq_len: int, batch: int = 1) -> int:
        kept = min(seq_len, self.cfg.window_size + self.cfg.sink_size)
        return int(batch * self.cfg.n_kv_heads * kept * self.cfg.head_dim * 2 * 2)
