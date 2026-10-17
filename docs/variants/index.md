# Variant index

One page per implemented variant, generated from the code by `scripts/gen_variant_docs.py`. The taxonomy behind the grouping is in [the survey](../survey.md#3-taxonomy).

The fixed-size-state models are documented separately, because they take `x` instead of `(q, k, v)`: **[state mixers](state_mixers.md)** (SSD / Gated DeltaNet / KDA / Gated DeltaNet-2).

| variant | family | training | mechanism |
| --- | --- | --- | --- |
| [`dense`](dense.md) | dense | no | Full softmax attention: the oracle, and the reference point for cost models. |
| [`sliding_window`](sliding_window.md) | local | no | Causal sliding window of ``window_size`` tokens plus ``sink_size`` leading tokens. |
| [`streaming_sink`](streaming_sink.md) | local | no | Attention sinks + window, restricted to exactly ``window_size`` recent tokens. |
| [`random_eviction`](random_eviction.md) | baseline | no | Control: evict uniformly at random among the unprotected tokens. |
| [`quest`](quest.md) | selection | no | Quest: page-level selection using channel-wise key bounds. |
| [`minference`](minference.md) | selection | no | Pattern-based prefill sparsity: A-shape, Vertical-Slash, Block-Sparse heads. |
| [`flexprefill`](flexprefill.md) | selection | no | FlexPrefill: context-aware, budget-adaptive block selection. |
| [`xattention`](xattention.md) | selection | no | XAttention: strided antidiagonal block scoring with a single threshold. |
| [`moba`](moba.md) | selection | no | Mixture of Block Attention: per-token top-k routing over KV blocks. |
| [`dsa`](dsa.md) | selection | yes | DeepSeek Sparse Attention: learned lightning indexer + top-k token selection. |
| [`seerattention_r`](seerattention_r.md) | selection | yes | Learned block pooling gate, adapted for long decoding. |
| [`nsa`](nsa.md) | native | yes | Three-branch native sparse attention (inference semantics of NSA). |
| [`csa2`](csa2.md) | native | yes | Compressed, low-precision, sparsely-selected attention (one layer). |
| [`mla`](mla.md) | compressed | no | Latent KV cache with per-head up-projections. |
