"""ctypes wrapper around the C paged-decode kernels in ``kernels/c/decode_attn.c``.

Why a C kernel in a repo about GPU kernels: because the central claim of the
sparse-attention literature -- "decode attention is memory bound, so reading
fewer KV bytes is what matters" -- is measurable on any machine, and a laptop CPU
shows the same effect as an H100 at a smaller scale.  The C kernel is compiled
at import time with the system compiler and then compared against the torch
reference in ``tests/test_kernels.py`` and timed in ``bench/bench_decode.py``.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

__all__ = ["C_SOURCE", "build_library", "paged_decode", "paged_decode_torch", "is_available"]

C_SOURCE = Path(__file__).parent / "c" / "decode_attn.c"
_LIB: Optional[ctypes.CDLL] = None
_BUILD_ERROR: Optional[str] = None


def _compiler() -> Optional[str]:
    for candidate in (os.environ.get("CC"), "cc", "clang", "gcc"):
        if candidate and shutil.which(candidate):
            return candidate
    return None


def build_library(force: bool = False) -> Optional[ctypes.CDLL]:
    """Compile the C kernel once per process; returns ``None`` if no compiler."""
    global _LIB, _BUILD_ERROR
    if _LIB is not None and not force:
        return _LIB
    cc = _compiler()
    if cc is None:
        _BUILD_ERROR = "no C compiler found"
        return None
    suffix = ".dylib" if os.uname().sysname == "Darwin" else ".so"
    out = Path(tempfile.gettempdir()) / f"linattn_decode_attn{suffix}"
    cmd = [cc, "-O3", "-ffast-math", "-shared", "-fPIC", "-o", str(out), str(C_SOURCE)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except Exception as exc:  # pragma: no cover - depends on the machine
        _BUILD_ERROR = f"compile failed: {exc}"
        return None
    lib = ctypes.CDLL(str(out))
    f32 = ctypes.POINTER(ctypes.c_float)
    i32 = ctypes.POINTER(ctypes.c_int32)
    for name in ("paged_decode_naive", "paged_decode_tiled"):
        fn = getattr(lib, name)
        fn.argtypes = [f32, f32, f32, i32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_float, f32]
        fn.restype = None
    lib.dense_decode_reference.argtypes = [f32, f32, f32, ctypes.c_int, ctypes.c_int, ctypes.c_float, f32]
    lib.dense_decode_reference.restype = None
    _LIB = lib
    return lib


def is_available() -> bool:
    return build_library() is not None


def _ptr(t: torch.Tensor):
    return ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_float))


def paged_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_ids: torch.Tensor,
    *,
    page_size: int,
    kernel: str = "tiled",
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Attention for a single ``(head, query)`` pair over selected pages.

    ``q``: ``(dim,)``; ``k``/``v``: ``(n_tokens, dim)``; ``page_ids``: ``(n_pages,)``.
    Returns the output vector and a dict with the number of KV entries and bytes
    actually read -- the accounting that makes the sparsity argument concrete.
    """
    lib = build_library()
    if lib is None:
        raise RuntimeError(f"C kernel unavailable: {_BUILD_ERROR}")
    dim = q.numel()
    if dim > 512:
        raise ValueError("the C kernel supports head_dim <= 512")
    qf = q.detach().to(torch.float32).contiguous()
    kf = k.detach().to(torch.float32).contiguous()
    vf = v.detach().to(torch.float32).contiguous()
    ids = page_ids.detach().to(torch.int32).contiguous()
    out = torch.zeros(dim, dtype=torch.float32)
    fn = lib.paged_decode_tiled if kernel == "tiled" else lib.paged_decode_naive
    fn(_ptr(qf), _ptr(kf), _ptr(vf), ctypes.cast(ids.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
       int(ids.numel()), int(page_size), int(dim), int(kf.shape[0]), float(1.0 / dim**0.5), _ptr(out))
    read_entries = int(min(kf.shape[0], ids.numel() * page_size))
    stats = {
        "kv_entries": read_entries,
        "bytes": read_entries * dim * 2 * 4,  # K and V, fp32
        "dense_entries": int(kf.shape[0]),
    }
    return out, stats


def paged_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_ids: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    """Torch reference for the same paged decode (gather the pages, then attend)."""
    dim = q.shape[-1]
    offsets = torch.arange(page_size, device=q.device)
    idx = (page_ids.view(-1, 1) * page_size + offsets.view(1, -1)).reshape(-1).clamp(max=k.shape[0] - 1)
    ks, vs = k[idx], v[idx]
    scores = (ks @ q) / dim**0.5
    return torch.softmax(scores, dim=-1) @ vs
