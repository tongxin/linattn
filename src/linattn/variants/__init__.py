"""One module per attention variant, all sharing the :class:`~linattn.core.AttentionVariant` contract.

Reading order for the codebase follows the taxonomy in ``docs/survey.md``:

``local``        structural sparsity (sliding window, attention sinks)
``eviction``     the eviction lineage, output-aware 2026 methods, and the random control
``selection``    query-aware block/token selection (Quest, MInference, FlexPrefill,
                 XAttention, MoBA, DSA, SeerAttention-R)
``native_sparse``trainable native sparse attention (NSA)
``mla``          low-rank latent KV (MLA), the base layer DSA sits on
``recurrent``    fixed-size state / linear attention (Mamba-2 SSD, Gated DeltaNet, KDA)
``hybrid``       layer interleaving of the two families
"""

from linattn.variants import eviction, hybrid, local, mla, native_sparse, recurrent, selection

__all__ = ["eviction", "hybrid", "local", "mla", "native_sparse", "recurrent", "selection"]
