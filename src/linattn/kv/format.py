"""KV representations, in bytes per token.

Every number here is per layer and per sequence token unless noted; multiply by the
layer count (and batch) for a footprint. The presets are shapes that actually ship,
so the cost model can compare them without pretending to know one exact model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

__all__ = ["KVFormat", "PRESETS", "format_by_name"]


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

    @property
    def is_state_based(self) -> bool:
        return self.bytes_per_token_per_layer is None


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


def format_by_name(name: str) -> KVFormat:
    if name not in PRESETS:
        raise KeyError(f"unknown kv format {name!r}; known: {sorted(PRESETS)}")
    return PRESETS[name]
