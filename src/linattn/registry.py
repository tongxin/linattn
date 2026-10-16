"""Name -> variant registry.

``linattn.list_variants()`` is the entry point used by the CLI, the benchmark
scripts and the docs generator: it returns the canonical variant list in
taxonomic order, which is the order used throughout ``docs/``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from linattn.core import AttentionVariant, AttnConfig, DenseAttention
from linattn.variants.eviction import (
    H2OAttention,
    PyramidKVAttention,
    RandomEvictionAttention,
    SnapKVAttention,
)
from linattn.variants.local import SlidingWindowAttention, StreamingSinkAttention
from linattn.variants.mla import MLAttention
from linattn.variants.native_sparse import NSAAttention
from linattn.variants.selection import (
    DSAAttention,
    FlexPrefillAttention,
    MInferenceAttention,
    MoBAAttention,
    QuestAttention,
    SeerAttentionR,
    XAttention,
)

__all__ = ["VARIANTS", "build", "list_variants", "describe"]

VARIANTS: Dict[str, Type[AttentionVariant]] = {
    # dense oracle
    "dense": DenseAttention,
    # structural / local sparsity
    "sliding_window": SlidingWindowAttention,
    "streaming_sink": StreamingSinkAttention,
    # KV eviction / compaction
    "h2o": H2OAttention,
    "snapkv": SnapKVAttention,
    "pyramidkv": PyramidKVAttention,
    "random_eviction": RandomEvictionAttention,
    # query-aware selection
    "quest": QuestAttention,
    "minference": MInferenceAttention,
    "flexprefill": FlexPrefillAttention,
    "xattention": XAttention,
    "moba": MoBAAttention,
    "dsa": DSAAttention,
    "seerattention_r": SeerAttentionR,
    # native (trainable) sparse attention
    "nsa": NSAAttention,
    # compressed latent state
    "mla": MLAttention,
}


def build(name: str, cfg: AttnConfig | None = None, **kw: Any) -> AttentionVariant:
    """Instantiate a variant by name (``cfg`` fields can be passed as keywords)."""
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; known: {sorted(VARIANTS)}")
    return VARIANTS[name](cfg, **kw)


def list_variants() -> List[str]:
    """Canonical variant names, in the order used by the documentation."""
    return list(VARIANTS)


def describe(name: str) -> Dict[str, Any]:
    """Metadata for one variant: family, whether it needs training, its citation."""
    cls = VARIANTS[name]
    return {
        "name": name,
        "class": cls.__name__,
        "family": getattr(cls, "family", "?"),
        "needs_training": getattr(cls, "trained", False),
        "citation": getattr(cls, "paper", ()),
        "docstring": (cls.__doc__ or "").strip().split("\n\n")[0].replace("\n", " "),
    }
