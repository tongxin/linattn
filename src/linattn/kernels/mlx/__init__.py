"""MLX backend: run the attention variants on Apple Silicon (Metal).

The rest of the repository is PyTorch-based, which on a MacBook means CPU: PyTorch's
MPS path has no fused attention kernels and no fast top-k, so the sparse-attention
variants cannot be validated at a realistic scale there.  MLX targets Metal directly
and has a fused SDPA (``mx.fast.scaled_dot_product_attention``), a real ``topk`` and
a custom-kernel API (``mx.fast.metal_kernel``) -- which makes it the right place to
check that these variants still hold up on a different accelerator with a different
memory hierarchy.

What is here:

``ops``           MLX-native primitives (dense/sliding/block-sparse attention,
                  indexer top-k, chunked delta rule), each mirroring a torch reference
``metal_decode``  a hand-written Metal kernel for paged decode attention
``variants``      MLX-native dense / sliding-window / Quest / DSA implementations
``adapter``       :class:`MlxAttention`, an ``AttentionVariant`` that runs the MLX
                  implementation behind the usual torch API, so the existing
                  harness, CLI, benchmarks and tests drive it unchanged

Availability is tolerant by design: importing this package on Linux, or in a
headless macOS session with no Metal device, must not break the torch paths.
"""

from __future__ import annotations

from typing import Optional

_IMPORT_ERROR: Optional[str] = None
try:  # pragma: no cover - environment dependent
    import mlx.core as mx
except Exception as exc:  # pragma: no cover - Linux, or no Metal device
    mx = None  # type: ignore
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

__all__ = ["mlx_available", "metal_available", "mlx_import_error", "mx"]


def mlx_available() -> bool:
    """True when ``mlx.core`` imported successfully in this process."""
    return mx is not None


def metal_available() -> bool:
    """True when MLX can see a Metal device (i.e. the GPU path is usable)."""
    if mx is None:
        return False
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


def mlx_import_error() -> Optional[str]:
    """Why MLX is unavailable, when it is (used as the test skip reason)."""
    return _IMPORT_ERROR
