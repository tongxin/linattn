"""Accelerated backends and the dispatch layer.

Three backends are provided, in increasing order of how much hardware they assume:

``torch``
    vectorised PyTorch.  Always available, and the reference every other backend
    is checked against.
``flex``
    PyTorch's ``flex_attention`` with a :class:`~torch.nn.attention.flex_attention.BlockMask`.
    This is the "software half" of co-design: the sparse pattern becomes a block
    mask, and the compiler generates the fused kernel.  Requires CUDA or a
    CPU-capable build of flex attention; the mask construction itself is portable.
``triton``
    hand-written Triton kernels for the operations where the compiler cannot win:
    fused indexer+top-k, paged selection, and the chunked delta-rule scan.
    Requires a CUDA GPU; each kernel ships with a ``*_torch`` fallback so the
    interface is testable everywhere.

``linattn.kernels.describe_backends()`` reports what the current machine can run,
which is what the benchmark scripts print in their report headers.
"""

from __future__ import annotations

import importlib.util
import platform
from typing import Dict

import torch

__all__ = ["describe_backends", "has_triton", "has_cuda", "has_mps", "backend_for"]


def has_triton() -> bool:
    return importlib.util.find_spec("triton") is not None


def has_cuda() -> bool:
    return bool(torch.cuda.is_available())


def has_mps() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def has_flex_attention() -> bool:
    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401

        return True
    except Exception:
        return False


def describe_backends() -> Dict[str, object]:
    return {
        "platform": f"{platform.system()} {platform.machine()}",
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "cuda": has_cuda(),
        "mps": has_mps(),
        "triton_installed": has_triton(),
        "flex_attention_importable": has_flex_attention(),
    }


def backend_for(preferred: str = "auto") -> str:
    """Pick the fastest usable backend for the current machine."""
    if preferred != "auto":
        return preferred
    if has_cuda() and has_triton():
        return "triton"
    if has_cuda() and has_flex_attention():
        return "flex"
    return "torch"
