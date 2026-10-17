"""KV representations, in bytes per token.

Every number here is per layer and per sequence token unless noted; multiply by the
layer count (and batch) for a footprint. The presets are shapes that actually ship,
so the cost model can compare them without pretending to know one exact model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import math

from linattn.kv.quant import QuantSpec

__all__ = ["KVFormat", "KVSpec", "PRESETS", "format_by_name"]


@dataclass(frozen=True)
class KVFormat:
    """A cached representation.

    ``bytes_per_token_per_layer`` is for representations whose footprint grows with
    context (KV caches). ``fixed_bytes_per_sequence_per_layer`` is for recurrent
    state, whose footprint does **not** grow -- the key economic difference between
    the two families in the survey.
    """

    name: str
    bytes_per_token_per_layer: Optional[float]
    fixed_bytes_per_sequence_per_layer: float = 0.0
    note: str = ""

    def bytes_for(self, seq_len: int, n_layers: int, batch: int = 1) -> float:
        if self.bytes_per_token_per_layer is not None:
            per_seq = self.bytes_per_token_per_layer * seq_len
        else:
            per_seq = 0.0
        per_seq += self.fixed_bytes_per_sequence_per_layer
        return per_seq * n_layers * batch

    @classmethod
    def from_spec(cls, spec: "KVSpec", n_layers: int) -> "KVFormat":
        """Collapse a composed :class:`KVSpec` into the per-layer linear form.

        ``KVFormat`` is linear in sequence length and multiplies by ``n_layers``, so
        the cross-layer multiplier is folded into the per-layer rate here. Using
        ``ceil(n_layers / group)`` (not ``n_layers / group``) keeps partial groups
        honest.
        """
        if spec.fixed_values_per_sequence:
            return cls(spec.name, None, fixed_bytes_per_sequence_per_layer=spec.bytes_for(1, 1, 1),
                       note=spec.describe())
        per_token = spec.bytes_for(4096, 1, 1) / 4096.0        # per stored copy
        copies = max(1, math.ceil(n_layers / spec.cross_layer_group))
        return cls(spec.name, per_token * copies / n_layers, note=spec.describe())

    @property
    def is_state_based(self) -> bool:
        return self.bytes_per_token_per_layer is None


@dataclass(frozen=True)
class KVSpec:
    """A composed storage spec: representation x compression x precision x reuse.

    This is the reusable building block behind the presets. ``values_per_token`` is
    the number of cached numbers per token **before** compression and precision
    (e.g. 2 x 8 heads x 128 = 2048 for a dense GQA layer, 576 for an MLA latent).
    """

    name: str
    values_per_token_per_layer: float
    quant: QuantSpec = QuantSpec(16, 32)
    compress_group: int = 1            # sequence-axis pooling stride (CSA "m")
    cross_layer_group: int = 1         # layers sharing one stored copy
    fixed_values_per_sequence: float = 0.0

    def __post_init__(self) -> None:
        if self.compress_group < 1 or self.cross_layer_group < 1:
            raise ValueError("compress_group and cross_layer_group must be >= 1")

    def bytes_for(self, seq_len: int, n_layers: int, batch: int = 1) -> float:
        """Exact bytes for one request, including quantisation scales and copying."""
        values = (self.values_per_token_per_layer / self.compress_group) * seq_len
        values += self.fixed_values_per_sequence
        per_copy = self.quant.bytes_for(max(1, int(math.ceil(values))))
        copies = max(1, math.ceil(n_layers / self.cross_layer_group))
        return per_copy * copies * batch

    def describe(self) -> str:
        bits = self.quant.bits
        parts = [f"{self.values_per_token_per_layer:g} values/token"]
        if self.compress_group > 1:
            parts.append(f"pooled x{self.compress_group}")
        parts.append(f"fp{bits}" if bits < 16 else "fp16")
        if self.cross_layer_group > 1:
            parts.append(f"shared over {self.cross_layer_group} layers")
        return ", ".join(parts)


def _dense(heads: int, dim: int, dtype_bytes: int) -> float:
    """K and V, per layer, per token."""
    return 2 * heads * dim * dtype_bytes


PRESETS: Dict[str, KVFormat] = {
    # DeepSeek-V3-like MLA: one ~576-wide latent per token in fp16.
    "mla": KVFormat("mla", bytes_per_token_per_layer=576 * 2,
                    note="MLA latent (~576 values/token, fp16), DeepSeek-V2/V3 shapes"),
    # V4.1-Flash: CSA2 + cross-layer reuse + FP4 KV -> 890 B/token across all layers.
    "csa2": KVFormat("csa2", bytes_per_token_per_layer=890 / 61,
                     note="CSA2 + cross-layer reuse + FP4 KV: 890 B/token across ~61 layers"),
    "dense_fp16": KVFormat("dense_fp16", bytes_per_token_per_layer=_dense(8, 128, 2),
                           note="dense GQA (8 KV heads x 128) fp16: 4 KiB/token/layer"),
    "dense_fp8": KVFormat("dense_fp8", bytes_per_token_per_layer=_dense(8, 128, 1),
                          note="dense GQA fp8: 2 KiB/token/layer"),
    # Kimi-Linear-like: KDA state is d_state x d_value per head, constant in T.
    "gdn_state": KVFormat("gdn_state", None,
                          fixed_bytes_per_sequence_per_layer=16 * 128 * 128 * 2,
                          note="KDA/GDN recurrent state (16 heads x 128 x 128, fp16), constant in T"),
}

# ---------------------------------------------------------------------------------------
# Composed specs: the CSA2 pieces, as reusable knobs rather than a magic number.
# ---------------------------------------------------------------------------------------
CSA2_VALUES_PER_TOKEN = 576.0        # MLA latent, DeepSeek-V2/V3 shapes

SPECS: Dict[str, KVSpec] = {
    # a naive composition: fp4 latent, no extra sequence compression, 4-layer sharing
    "csa2": KVSpec("csa2", values_per_token_per_layer=CSA2_VALUES_PER_TOKEN,
                   quant=QuantSpec(bits=4, group_size=32), compress_group=1,
                   cross_layer_group=4),
    # what it takes to *approach* 890 B/token with the same latent and precision:
    # ~3 stored copies across 61 layers (group 21) rather than one per 4 layers
    "csa2_reach890": KVSpec("csa2_reach890", values_per_token_per_layer=CSA2_VALUES_PER_TOKEN,
                            quant=QuantSpec(bits=4, group_size=32), compress_group=1,
                            cross_layer_group=21),
}

for _name, _spec in SPECS.items():
    PRESETS[_name] = KVFormat.from_spec(_spec, n_layers=61)

# The published number, kept as a literal: it is a report value, not a composition.
PRESETS["csa2_reported"] = KVFormat("csa2_reported", bytes_per_token_per_layer=890 / 61,
                                    note="DeepSeek-V4.1-Flash reported global KV: 890 B/token across ~61 layers")


def format_by_name(name: str) -> KVFormat:
    if name not in PRESETS:
        raise KeyError(f"unknown kv format {name!r}; known: {sorted(PRESETS)}")
    return PRESETS[name]
