"""Cross-layer KV reuse: one stored copy per group of layers.

Two things are being modelled, and they are separable:

* **storage** -- ``group_size`` layers share one cached copy, so the footprint is
  divided by the group size (modulo the partial last group);
* **access** -- the shared layers *read the representative layer's* KV instead of
  computing their own. Whether that is acceptable is a property of the trained
  model: neighbouring layers can be trained to share (the architectural choice in
  CSA2 / IndexCache), while independently trained layers cannot.

``share_layers`` makes the access side explicit so an experiment can measure the
divergence instead of assuming it. ``CrossLayerPlan`` makes the storage side
explicit so the cost model can use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

__all__ = ["CrossLayerPlan", "share_layers"]


@dataclass(frozen=True)
class CrossLayerPlan:
    """Partition layers into groups that share one cached KV copy."""

    n_layers: int = 61
    group_size: int = 4

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")

    @property
    def groups(self) -> List[List[int]]:
        return [
            list(range(start, min(start + self.group_size, self.n_layers)))
            for start in range(0, self.n_layers, self.group_size)
        ]

    @property
    def stored_layers(self) -> List[int]:
        """The representative ("full") layer of each group."""
        return [g[0] for g in self.groups]

    @property
    def copies(self) -> int:
        return len(self.groups)

    @property
    def bytes_multiplier(self) -> float:
        """Fraction of a per-layer cache that still has to be stored."""
        return self.copies / self.n_layers

    def representative(self, layer: int) -> int:
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"layer {layer} out of range")
        return self.stored_layers[layer // self.group_size]

    def is_full(self, layer: int) -> bool:
        """True for the layer in each group that computes its own KV."""
        return self.representative(layer) == layer

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_layers": self.n_layers,
            "group_size": self.group_size,
            "copies": self.copies,
            "bytes_multiplier": self.bytes_multiplier,
        }


def share_layers(layers: Sequence, plan: CrossLayerPlan) -> Tuple[Dict[int, object], Dict[int, int]]:
    """Replace each layer's KV with its group representative's.

    ``layers`` is a sequence of per-layer tensors (or anything indexable by layer).
    Returns ``(stored, mapping)`` where ``stored`` holds only the representatives and
    ``mapping`` says which representative each layer reads.
    """
    if len(layers) != plan.n_layers:
        raise ValueError(f"expected {plan.n_layers} layers, got {len(layers)}")
    stored = {layer: layers[layer] for layer in plan.stored_layers}
    mapping = {layer: plan.representative(layer) for layer in range(plan.n_layers)}
    return stored, mapping
