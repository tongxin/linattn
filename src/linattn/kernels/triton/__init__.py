"""Triton kernels (CUDA).  Each kernel has a portable torch fallback.

The kernels here are the *accelerated* counterparts of the semantics implemented
in ``linattn.variants``.  They are written so that the torch fallback and the
Triton path share a signature: ``tests/test_kernels.py`` runs the fallback on any
machine and asserts parity between the two on a GPU.

``block_sparse``      exact attention over a selected set of KV blocks (all
                      selection variants: Quest, MInference, XAttention, MoBA, NSA,
                      SeerAttention-R, DSA)
``fused_indexer``     lightning-indexer scoring fused with a two-pass top-k
                      (DSA/CSA), avoiding the O(L^2) score materialisation
``quest_pages``       per-page key min/max maintenance and query-aware page scoring
``delta_chunk``       chunkwise delta-rule scan (SSD / Gated DeltaNet / KDA)
"""

from __future__ import annotations

import importlib.util

HAS_TRITON = importlib.util.find_spec("triton") is not None

__all__ = ["HAS_TRITON"]
