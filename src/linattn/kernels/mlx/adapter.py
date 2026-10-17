"""Run an MLX variant behind the repository's torch :class:`AttentionVariant` API.

This is the "framework adaptation" piece: the selection and the attention math run
on Metal through :func:`linattn.kernels.mlx.variants.mlx_forward`, while the outer
contract (``forward(q, k, v, state=...) -> (out, state)``) stays identical to the
torch variants.  The existing harness, CLI, benchmarks and tests therefore drive the
MLX path without knowing about it::

    from linattn.kernels.mlx.adapter import build_mlx
    from linattn.harness import compare

    compare([build_mlx("quest"), build("quest")], seq=1024)   # torch vs Metal

Costs to be aware of: every chunk crosses the torch<->numpy<->MLX boundary, so this
path is for **validation**, not for serving.  ``bench/bench_mlx.py`` measures the
MLX-only path (no conversion) for latency.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from linattn.core import AttentionVariant, AttnConfig, dense_attention
from linattn.kernels.mlx import mlx_available, mx, ops
from linattn.kernels.mlx.variants import SUPPORTED, mlx_forward

__all__ = ["MlxAttention", "build_mlx", "MLX_SUPPORTED"]

MLX_SUPPORTED = SUPPORTED


class MlxAttention(AttentionVariant):
    """MLX implementation of one variant, exposed as a torch ``AttentionVariant``."""

    family = "mlx"

    def __init__(self, cfg: Optional[AttnConfig] = None, variant: str = "quest", **kw: Any):
        super().__init__(cfg, **kw)
        if variant not in SUPPORTED:
            raise KeyError(f"mlx backend supports {SUPPORTED}, not {variant!r}")
        self.variant = variant
        self.name = f"mlx_{variant}"
        self._extra: Dict[str, float] = {}
        self._reference = self._torch_reference()

    # -- selection/execution ------------------------------------------------------
    def attend(self, q: torch.Tensor, state: Dict[str, Any], offset: int = 0) -> torch.Tensor:
        if not mlx_available():
            from linattn.kernels.mlx import mlx_import_error

            raise RuntimeError(f"MLX is not available: {mlx_import_error()}")
        cfg = self.cfg
        q_mx = ops.to_mlx(q)
        k_mx = ops.to_mlx(state["k"])
        v_mx = ops.to_mlx(state["v"])
        out, stats = mlx_forward(
            self.variant,
            q_mx,
            k_mx,
            v_mx,
            window=cfg.window_size,
            sinks=cfg.sink_size,
            page_size=cfg.block_size,
            topk_pages=cfg.topk_blocks,
            topk_tokens=cfg.topk_tokens,
            indexer_dim=int(cfg.extras.get("indexer_dim", 16)),
            n_indexer_heads=int(cfg.extras.get("indexer_heads", 4)),
            compress_ratio=int(cfg.extras.get("compress_ratio", 1)),
            seed=cfg.seed,
            indexer_proj=cfg.extras.get("indexer_proj"),
        )
        self._extra = {k: float(v) for k, v in stats.items()}
        return ops.to_torch(out, like=q)

    def stats(self) -> Dict[str, float]:
        return dict(self._extra)

    # -- cost model: reuse the torch variant's analytic accounting ----------------
    def _torch_reference(self):
        from linattn.registry import build

        try:
            return build(self.variant, self.cfg)
        except Exception:
            return None

    def read_budget(self, seq_len: int, batch: int = 1) -> float:
        if self._reference is not None:
            return self._reference.read_budget(seq_len, batch)
        return float(seq_len)

    def memory_bytes(self, seq_len: int, batch: int = 1) -> int:
        if self._reference is not None:
            return self._reference.memory_bytes(seq_len, batch)
        return super().memory_bytes(seq_len, batch)

    def flops(self, seq_len: int, batch: int = 1) -> int:
        if self._reference is not None:
            return self._reference.flops(seq_len, batch)
        return super().flops(seq_len, batch)


def build_mlx(variant: str, cfg: Optional[AttnConfig] = None, **kw: Any) -> MlxAttention:
    """Factory mirroring :func:`linattn.registry.build`, for the Metal backend."""
    return MlxAttention(cfg, variant=variant, **kw)
