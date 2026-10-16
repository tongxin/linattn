"""linattn: sparse, compressed-state and co-designed attention variants.

The package is organised in three layers:

* :mod:`linattn.core` -- shared primitives, the reference (dense) attention oracle
  and the :class:`AttentionVariant` interface used by every implementation.
* :mod:`linattn.variants` -- one module per attention variant, each implementing
  the same prefill/decode contract so that variants are drop-in comparable.
* :mod:`linattn.kernels` -- accelerated implementations (Triton, PyTorch
  FlexAttention block masks, C/OpenMP) plus the dispatch layer that picks a
  backend and reports whether it is available on the current machine.

Every variant exposes the same three things, which is what the validation suite
in ``tests/`` keys off:

``forward(q, k, v, ...)``
    chunked causal attention with explicit cache semantics;
``memory_bytes(T)`` / ``flops(T)``
    analytic cost model, used for the efficiency accounting in the docs;
``read_budget(T)``
    number of KV entries actually read, used to check that a variant honours its
    sparsity budget.
"""

from linattn.core import AttentionVariant, AttnConfig, DenseAttention, dense_attention
from linattn.registry import VARIANTS, build, describe, list_variants

__all__ = [
    "AttentionVariant",
    "AttnConfig",
    "DenseAttention",
    "dense_attention",
    "VARIANTS",
    "build",
    "describe",
    "list_variants",
]

__version__ = "0.1.0"
