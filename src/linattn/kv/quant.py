"""Low-precision KV storage: FP8 and FP4 (E2M1) with per-group scales.

Quantisation is the cheapest way to cut KV bytes -- it needs no selector, no
training and no schedule -- so it belongs in the storage layer rather than in any
one attention variant. Both codecs here are *group-wise*: one scale per
``group_size`` values along the last axis, which is what makes 4-bit usable at all
(a single tensor-wide scale destroys the small channels).

The FP4 codec is the E2M1 grid actually used for 4-bit inference: the eight
magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 with a sign. Bytes are accounted as
``4 bits/value + 2 bytes per group`` (two values per byte, as in packed storage).

``FP8`` uses PyTorch's real ``float8_e4m3fn`` cast rather than a simulated grid, so
its error matches what a kernel would see.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

__all__ = [
    "FP4_GRID",
    "QuantSpec",
    "quantize",
    "dequantize",
    "bytes_for",
    "roundtrip",
    "relative_error",
]

# E2M1 magnitudes; codes 0..7 are positive, 8..15 negative (sign bit).
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


@dataclass(frozen=True)
class QuantSpec:
    """``bits`` is 16 (no-op), 8 (e4m3) or 4 (E2M1)."""

    bits: int = 16
    group_size: int = 32

    def __post_init__(self) -> None:
        if self.bits not in (4, 8, 16):
            raise ValueError("bits must be 4, 8 or 16")
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")

    @property
    def enabled(self) -> bool:
        return self.bits < 16

    def bytes_for(self, n_values: int) -> float:
        if not self.enabled:
            return n_values * 2.0
        n_groups = math.ceil(n_values / self.group_size)
        return n_values * self.bits / 8.0 + n_groups * 2.0   # scales in fp16


def _groupify(x: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    """``(..., n) -> (..., n_groups, group)``, padding the tail with zeros."""
    n = x.shape[-1]
    pad = (-n) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x.reshape(*x.shape[:-1], -1, group_size), n


def _ungroupify(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.reshape(*x.shape[:-2], -1)[..., :n]


def quantize(x: torch.Tensor, spec: QuantSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(codes, scales)``. Codes are int8 (fp4 index) or float8 (fp8)."""
    if not spec.enabled:
        return x.clone(), torch.ones(*x.shape[:-1], 1, dtype=torch.float32, device=x.device)
    grouped, _ = _groupify(x.float(), spec.group_size)
    if spec.bits == 4:
        # absmax scale so the largest magnitude maps to 6 (the top of the grid)
        scale = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-8) / FP4_GRID[-1]
        normalised = grouped / scale
        # nearest grid point, preserving sign
        idx = (normalised.abs().unsqueeze(-1) - FP4_GRID.view(*[1] * normalised.dim(), -1)).abs().argmin(-1)
        negative = normalised < 0
        codes = torch.where(negative, idx + 8, idx).to(torch.int8)
        return _ungroupify(codes, x.shape[-1]), _ungroupify(scale, -(-x.shape[-1] // spec.group_size))
    # fp8 e4m3: per-group scale to the representable max (448)
    scale = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 448.0
    codes = (grouped / scale).to(torch.float8_e4m3fn)
    return _ungroupify(codes, x.shape[-1]), _ungroupify(scale, -(-x.shape[-1] // spec.group_size))


def dequantize(codes: torch.Tensor, scales: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """Inverse of :func:`quantize`."""
    if not spec.enabled:
        return codes.clone()
    grouped, n = _groupify(codes, spec.group_size)
    if spec.bits == 4:
        code = grouped.to(torch.int64)
        magnitude = FP4_GRID[code & 7].to(torch.float32)
        value = torch.where(code >= 8, -magnitude, magnitude)
    else:
        value = grouped.to(torch.float32)
    # one scale per group, broadcast over the group axis
    scale = scales.float()[..., : value.shape[-2]].unsqueeze(-1)
    return _ungroupify(value * scale, n)


def bytes_for(n_values: int, spec: QuantSpec) -> float:
    """Bytes for ``n_values`` cached numbers under ``spec`` (including scales)."""
    return spec.bytes_for(n_values)


def roundtrip(x: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """Quantise then dequantise, for error measurement."""
    codes, scales = quantize(x, spec)
    return dequantize(codes, scales, spec)


def relative_error(x: torch.Tensor, hat: torch.Tensor) -> float:
    denom = x.float().norm().clamp_min(1e-12)
    return float((x.float() - hat.float()).norm() / denom)
