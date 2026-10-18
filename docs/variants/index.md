# Variant index

One page per implemented variant, generated from the code by `scripts/gen_variant_docs.py`. The taxonomy behind the grouping is in [the survey](../survey.md#3-taxonomy).

The fixed-size-state models are documented separately, because they take `x` instead of `(q, k, v)`: **[state mixers](state_mixers.md)** (SSD / GDN / KDA / GDN-2 / Q-Delta / KDN / SwiLA).

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
| [`linformer`](linformer.md) | academic | no | Low-rank keys/values: ``softmax(Q (E K)^T) (F V)`` in ``O(T E d)``. |
| [`performer`](performer.md) | academic | no | FAVOR+ positive random features: linear time, causal, no attention matrix. |
| [`bigbird`](bigbird.md) | academic | no | Window + random + global blocks as an explicit mask. |
| [`reformer_lsh`](reformer_lsh.md) | academic | no | Angular-LSH buckets: a query reads the keys hashed into its bucket. |
| [`free_solo`](free_solo.md) | modern | no | Sliding window + keypoint cells, and nothing else. |
| [`namoh`](namoh.md) | modern | no | Head-owned token subsets: each head attends only its assigned subsequence. |
| [`hla_mixing`](hla_mixing.md) | modern | no | Chunk-wise dynamic mixing of full attention and a linear state. |
| [`mass_oracle`](mass_oracle.md) | modern | no | The COBS oracle: select blocks by their *exact* attention mass. |
| [`saga`](saga.md) | frontier | yes | Sparse Asymmetric Group-Query Attention with approximate top-N selection. |
| [`pisa`](pisa.md) | frontier | yes | Pyramid Top-K block selection with log-linear routing. |
| [`jssa`](jssa.md) | frontier | yes | Joint Subspace Sparse Attention: rank tokens in a shared QK subspace. |
| [`masa`](masa.md) | frontier | no | Matrix Approximation Sparse Attention: score units by output-error reduction. |
| [`oval`](oval.md) | frontier | no | Output-aware local page bases for sparse KV retrieval. |
| [`sfa`](sfa.md) | frontier | yes | Sparse Feature Attention: sparsify the feature axis, not the token axis. |
| [`dropkv`](dropkv.md) | eviction | no | DropKV: evict tokens by decoupled residual-output perturbation. |
| [`restkv`](restkv.md) | eviction | no | ReST-KV: output reconstruction with spatial--temporal smoothing. |
| [`beaconkv`](beaconkv.md) | eviction | no | BeaconKV: keep compact representatives of historical query clusters. |
| [`h2o`](h2o.md) | eviction | no | Heavy-Hitter Oracle: keep tokens by *accumulated* attention mass. |
| [`snapkv`](snapkv.md) | eviction | no | SnapKV: score keys once from an observation window of recent queries. |
| [`tova`](tova.md) | eviction | no | TOVA: keep the token the *latest* query attends to most, plus recent tokens. |
