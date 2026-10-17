"""Name -> implementation registries.

Two catalogues, because the two families have different interfaces:

``VARIANTS``
    attention variants.  They take pre-projected ``(q, k, v)`` and implement the
    ``AttentionVariant`` prefill/decode contract, so the harness, CLI, benchmarks
    and tests can drive any of them identically.
``MIXERS``
    fixed-size-state sequence mixers (SSD / Gated DeltaNet / KDA / Gated
    DeltaNet-2).  They take a token embedding ``x`` of shape ``(B, T, d_model)`` and
    own their projections, so they cannot implement the ``(q, k, v)`` contract --
    registering them in ``VARIANTS`` would make ``build()`` and ``compare()`` lie.

Keeping them in separate registries is right; keeping them *only* in the code (as
happened until this was noticed) is not.  ``list_variants()`` and
``list_mixers()`` are both surfaced by ``python -m linattn.cli variants`` so the
catalogue you can read matches the catalogue you can run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

from linattn.core import AttentionVariant, AttnConfig, DenseAttention
from linattn.variants.eviction import RandomEvictionBaseline
from linattn.variants.local import SlidingWindowAttention, StreamingSinkAttention
from linattn.variants.mla import MLAttention
from linattn.variants.native_sparse import NSAAttention
from linattn.variants.recurrent import STATE_KINDS, LinearAttentionConfig, LinearAttentionLayer
from linattn.variants.selection import (
    DSAAttention,
    FlexPrefillAttention,
    MInferenceAttention,
    MoBAAttention,
    QuestAttention,
    SeerAttentionR,
    XAttention,
)

__all__ = [
    "VARIANTS",
    "MIXERS",
    "build",
    "build_mixer",
    "list_variants",
    "list_mixers",
    "describe",
    "describe_mixer",
    "MixerSpec",
]

VARIANTS: Dict[str, Type[AttentionVariant]] = {
    # dense oracle
    "dense": DenseAttention,
    # structural / local sparsity
    "sliding_window": SlidingWindowAttention,
    "streaming_sink": StreamingSinkAttention,
    # control baseline for the retired eviction family
    "random_eviction": RandomEvictionBaseline,
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


@dataclass(frozen=True)
class MixerSpec:
    """What a state mixer is, for the CLI, the docs generator and the tests."""

    kind: str
    summary: str
    recurrence: str
    decay: str
    gates: str
    citation: Tuple[str, str]
    needs_training: bool = True


MIXERS: Dict[str, MixerSpec] = {
    "ssd": MixerSpec(
        kind="ssd",
        summary="Mamba-2 / semiseparable state-space layer (no erase term)",
        recurrence="S_t = diag(a_t) S_{t-1} + b_t k_t v_t^T",
        decay="scalar per head (learned A_log)",
        gates="write only (beta)",
        citation=("Dao & Gu, Transformers are SSMs (2024)", "https://arxiv.org/abs/2405.21060"),
    ),
    "gdn": MixerSpec(
        kind="gdn",
        summary="Gated DeltaNet: gated decay plus the delta-rule erase",
        recurrence="S_t = (I - b_t k_t k_t^T) diag(a_t) S_{t-1} + b_t k_t v_t^T",
        decay="scalar per head",
        gates="one scalar beta for erase and write",
        citation=("Yang et al., Gated Delta Networks (ICLR 2025)", "https://arxiv.org/abs/2412.06464"),
    ),
    "kda": MixerSpec(
        kind="kda",
        summary="Kimi Delta Attention: channel-wise decay plus the delta-rule erase",
        recurrence="S_t = (I - b_t k_t k_t^T) diag(a_t) S_{t-1} + b_t k_t v_t^T, a_t per channel",
        decay="channel-wise (per feature dimension)",
        gates="one scalar beta for erase and write",
        citation=("Kimi Team, Kimi Linear (2025)", "https://arxiv.org/abs/2510.26692"),
    ),
    "gdn2": MixerSpec(
        kind="gdn2",
        summary="Gated DeltaNet-2: decoupled channel-wise erase and write gates",
        recurrence="S_t = (I - diag(b_t) k_t k_t^T) diag(a_t) S_{t-1} + diag(w_t) k_t v_t^T",
        decay="channel-wise",
        gates="separate channel-wise erase (b) and write (w) gates",
        citation=("Yang et al., Gated DeltaNet-2 (2026)", "https://arxiv.org/abs/2605.22791"),
    ),
}

# guard against the exact drift this registry exists to prevent
assert set(MIXERS) == set(STATE_KINDS), f"MIXERS {set(MIXERS)} != STATE_KINDS {set(STATE_KINDS)}"


def build(name: str, cfg: AttnConfig | None = None, **kw: Any) -> AttentionVariant:
    """Instantiate a variant by name (``cfg`` fields can be passed as keywords)."""
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; known: {sorted(VARIANTS)}")
    return VARIANTS[name](cfg, **kw)


def list_variants() -> List[str]:
    """Canonical variant names, in the order used by the documentation."""
    return list(VARIANTS)


def build_mixer(kind: str, **config_kwargs: Any) -> LinearAttentionLayer:
    """Instantiate a state mixer by kind (``ssd`` | ``gdn`` | ``kda`` | ``gdn2``)."""
    if kind not in MIXERS:
        raise KeyError(f"unknown mixer {kind!r}; known: {sorted(MIXERS)}")
    config_kwargs.setdefault("kind", kind)
    return LinearAttentionLayer(LinearAttentionConfig(**config_kwargs))


def list_mixers() -> List[str]:
    """Canonical state-mixer kinds, in taxonomic order."""
    return list(MIXERS)


def describe_mixer(kind: str) -> Dict[str, Any]:
    """Metadata for one state mixer."""
    spec = MIXERS[kind]
    return {
        "name": kind,
        "class": "LinearAttentionLayer",
        "family": "compressed-state",
        "needs_training": spec.needs_training,
        "citation": spec.citation,
        "summary": spec.summary,
        "recurrence": spec.recurrence,
        "decay": spec.decay,
        "gates": spec.gates,
    }


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
