"""Name -> implementation registries.

Three catalogues, because the families have different interfaces:

``VARIANTS``
    attention variants.  They take pre-projected ``(q, k, v)`` and implement the
    ``AttentionVariant`` prefill/decode contract, so the harness, CLI, benchmarks
    and tests can drive any of them identically.
``MIXERS``
    fixed-size-state sequence mixers (SSD / GDN / KDA / GDN-2 / Q-Delta / KDN /
    SwiLA).  They take a token embedding ``x`` of shape ``(B, T, d_model)`` and
    own their projections, so they cannot implement the ``(q, k, v)`` contract --
    registering them in ``VARIANTS`` would make ``build()`` and ``compare()`` lie.
``KERNEL_VARIANTS``
    backend-native implementations behind the same ``AttentionVariant`` API, named
    ``<backend>_<variant>`` (currently ``mlx_dense``, ``mlx_quest``, ...).  These are
    backends for an algorithm, not extra algorithm families, so the CLI catalogue
    lists each variant once while ``build()`` can select a backend explicitly.
    Their analytic cost model delegates to the matching torch variant, so they can
    be costed without the accelerator runtime being installed.

Keeping them in separate registries is right; keeping them *only* in the code (as
happened until this was noticed) is not.  ``list_variants()`` and
``list_mixers()`` are surfaced by ``python -m linattn.cli variants`` so the
catalogue you can read matches the catalogue you can run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

from linattn.core import AttentionVariant, AttnConfig, DenseAttention
from linattn.variants.eviction import RandomEvictionBaseline
from linattn.variants.eviction import H2OAttention, SnapKVAttention, TOVAAttention
from linattn.variants.academic import (
    BigBirdAttention,
    LinformerAttention,
    PerformerAttention,
    ReformerLSHAttention,
)
from linattn.variants.modern import (
    FreeSoloAttention,
    HLAMixingAttention,
    MassOracleAttention,
    NAMOHAttention,
)
from linattn.variants.frontier import (
    BeaconKVAttention,
    DropKVAttention,
    JSSAAttention,
    MASAAttention,
    OVALAttention,
    PISAAttention,
    ReSTKVAttention,
    SAGAAttention,
    SFAAttention,
)
from linattn.variants.csa2 import CSA2Attention
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
    "KERNEL_VARIANTS",
    "MIXERS",
    "build",
    "build_any",
    "build_mixer",
    "list_variants",
    "list_kernel_variants",
    "list_mixers",
    "describe",
    "describe_kernel",
    "describe_mixer",
    "MixerSpec",
]

VARIANTS: Dict[str, Type[AttentionVariant]] = {
    # dense oracle
    "dense": DenseAttention,
    # structural / local sparsity
    "sliding_window": SlidingWindowAttention,
    "streaming_sink": StreamingSinkAttention,
    # control baseline for attention-mass eviction
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
    "csa2": CSA2Attention,
    # compressed latent state
    "mla": MLAttention,
    # canonical academic milestones (2020-2021)
    "linformer": LinformerAttention,
    "performer": PerformerAttention,
    "bigbird": BigBirdAttention,
    "reformer_lsh": ReformerLSHAttention,
    # current academic lines (2025-2026)
    "free_solo": FreeSoloAttention,
    "namoh": NAMOHAttention,
    "hla_mixing": HLAMixingAttention,
    "mass_oracle": MassOracleAttention,
    # 2025-2026 frontier: routing, output-aware selection, feature sparsity
    "saga": SAGAAttention,
    "pisa": PISAAttention,
    "jssa": JSSAAttention,
    "masa": MASAAttention,
    "oval": OVALAttention,
    "sfa": SFAAttention,
    # output-aware and long-horizon KV eviction
    "dropkv": DropKVAttention,
    "restkv": ReSTKVAttention,
    "beaconkv": BeaconKVAttention,
    # academic eviction lineage, kept as baselines for the random-eviction result
    "h2o": H2OAttention,
    "snapkv": SnapKVAttention,
    "tova": TOVAAttention,
}


def _discover_kernel_variants() -> Dict[str, str]:
    """Names of backend-native implementations, mapped to their torch reference."""
    from linattn.kernels.mlx.variants import SUPPORTED as MLX_SUPPORTED

    return {f"mlx_{name}": name for name in MLX_SUPPORTED}


KERNEL_VARIANTS: Dict[str, str] = _discover_kernel_variants()


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
    "qdelta": MixerSpec(
        kind="qdelta",
        summary="Q-Delta: the query's own state prediction corrects the delta update",
        recurrence="S_t = (I - b_t(k_t + λ_t q_t)k_t^T) diag(a_t) S_{t-1} + b_t k_t v_t^T",
        decay="scalar per head",
        gates="key-write strength plus a learned query-feedback coefficient",
        citation=("Park et al., Q-Delta (ICML 2026)", "https://arxiv.org/abs/2606.08804"),
    ),
    "kdn": MixerSpec(
        kind="kdn",
        summary="Diagonal Kalman DeltaNet: uncertainty-aware associative memory",
        recurrence="S_t = (I - κ_t k_t^T) diag(a_t) S_{t-1} + κ_t v_t^T, κ_t = p_t ⊙ k_t / (r_t + k_t^T p_t k_t)",
        decay="channel-wise",
        gates="uncertainty-derived Kalman gain; diagonal covariance state",
        citation=("Bui et al., Kalman Delta Networks (2026)", "https://arxiv.org/abs/2609.07816"),
    ),
    "swila": MixerSpec(
        kind="swila",
        summary="Switching Linear Attention: a routed mixture of delta-rule memories",
        recurrence="S_t^(j) = a_tj S_{t-1}^(j) + b_tj π_tj (v_t - a_tj k_t^T S_{t-1}^(j)) k_t",
        decay="per expert, per head",
        gates="soft posterior responsibilities over J memory experts",
        citation=("Lee et al., Switching Linear Attention (COLM 2026)", "https://arxiv.org/abs/2609.39034"),
    ),
}

# guard against the exact drift this registry exists to prevent
assert set(MIXERS) == set(STATE_KINDS), f"MIXERS {set(MIXERS)} != STATE_KINDS {set(STATE_KINDS)}"


def build(name: str, cfg: AttnConfig | None = None, **kw: Any) -> AttentionVariant:
    """Instantiate a torch or backend-native variant by name."""
    if name in VARIANTS:
        return VARIANTS[name](cfg, **kw)
    if name in KERNEL_VARIANTS:
        backend, _, base = name.partition("_")
        if backend == "mlx":
            from linattn.kernels.mlx.adapter import build_mlx

            return build_mlx(base, cfg, **kw)
    known = sorted([*VARIANTS, *KERNEL_VARIANTS])
    raise KeyError(f"unknown variant {name!r}; known: {known}")


def build_any(name: str, cfg: AttnConfig | None = None, **kw: Any) -> AttentionVariant:
    """Alias for :func:`build` that makes the kernel dispatch explicit."""
    return build(name, cfg, **kw)


def list_variants() -> List[str]:
    """Canonical variant names, in the order used by the documentation."""
    return list(VARIANTS)


def list_kernel_variants() -> List[str]:
    """Canonical ``<backend>_<variant>`` names for backend-native implementations."""
    return list(KERNEL_VARIANTS)


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
    if name in KERNEL_VARIANTS:
        return describe_kernel(name)
    cls = VARIANTS[name]
    return {
        "name": name,
        "class": cls.__name__,
        "family": getattr(cls, "family", "?"),
        "needs_training": getattr(cls, "trained", False),
        "citation": getattr(cls, "paper", ()),
        "docstring": (cls.__doc__ or "").strip().split("\n\n")[0].replace("\n", " "),
    }


def describe_kernel(name: str) -> Dict[str, Any]:
    """Metadata for a backend-native variant, using its torch reference's citation."""
    if name not in KERNEL_VARIANTS:
        raise KeyError(f"unknown kernel variant {name!r}; known: {sorted(KERNEL_VARIANTS)}")
    backend, _, base = name.partition("_")
    meta = describe(base)
    return {
        **meta,
        "name": name,
        "class": f"{backend}_forward",
        "family": backend,
        "reference": base,
    }
