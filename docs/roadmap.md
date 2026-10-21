# Roadmap / deliberately deferred

Items that are understood and agreed but **out of current scope**. Keeping them here
turns a limitation into a decision.

## 1. Variable-length (ragged) batching

Every variant supports *equal-length* batches (`tests/test_batching.py`). There is no
`lengths` argument and no padding mask, so a padded short sequence would attend to
padding through the shared causal mask.

What it needs:

* a `lengths` (or mask) argument threaded through `AttentionVariant.forward` and every
  `attend`, including the eviction paths (whose compaction must not mix padding into
  the retained window);
* a per-item causal mask in the reference path, and a block-sparse equivalent in the
  kernels (skip blocks that are entirely padding);
* tests that a ragged batch equals one call per sequence, plus a check that padding is
  never read;
* a decision on where padding enters: caller-provided padded tensors, or per-item
  slicing in the driver (simpler, slower).

Continuous batching / request scheduling remains out of scope entirely: this is a
kernel-and-semantics repository, not a serving engine.

## 2. Batch invariance as a contract

All deterministic variants are pinned batch-invariant (batched == per-item to <1e-9).
Two gaps remain before that is a *contract* rather than a test result:

* `random_eviction` draws from one RNG stream per call, so a batched run is not
  item-wise equal to single runs. Options: derive a per-item RNG from the item index
  (probably right), or document it as intentionally non-invariant across shapes
  (current behaviour, tested only for seed determinism).
* floating-point summation order can differ with batch size in BLAS paths; the 1e-9
  tolerance covers this today, but a strict guarantee needs fixed-order kernels --
  a different performance point.

## 3. CSA2: what remains after the modelling pass

Implemented: compressed indexing (`variants/csa2.py`), an E2M1/FP8 group-wise codec
(`kv/quant.py`), a cross-layer sharing plan (`kv/reuse.py`), a multi-layer stack that
actually reads a representative's cache (`kv/csa2.py`), a synthetic error decomposition
(`experiments/csa2_approximation.py`), and an official-config comparison
(`experiments/csa2_official_check.py`).

What a *released-architecture* claim would still need:

* **trained weights** -- the indexer projections are random unless fitted, so
  selection quality is not DeepSeek's;
* **the exact layer schedule** -- the 1017 model card and `config.json` now disclose
  40 layers, encoder `m=2`, decoder `m=1`, and the Full/Reindex/Reuse source layers.
  Our stack still uses a uniform `group_size` and does not model CED prefill or the
  hierarchical candidate pool;
* **a hardware FP4 path** -- the codec simulates packed 4-bit accounting but the
  arithmetic is FP32 after dequantisation; a real kernel keeps FP4 in registers/Tensor
  Cores (and the numerics differ);
* **real-workload validation** -- the sharing-divergence sweep uses synthetic layers;
  the trained regime (neighbouring layers trained to share) is the one that matters.

### Open problem: CSA2 algorithmic sensitivity

CSA2 combines CED, compressed indexing, cross-layer Full/Reindex/Reuse modes,
hierarchical candidate selection, FP4 main KV and FP8 SWA. The public report gives
whole-model benchmark results and says FP4 caused “only marginal degradation”, but it
does not publish a layerwise dense-versus-CSA2 attention/hidden-state ablation.

Do not close this with a synthetic surrogate or another whole-model benchmark. Revisit
only when the following are available:

1. a runnable checkpoint and a matched dense/reference configuration;
2. declared targets: attention operator, value-weighted output, hidden state or final
   task behavior;
3. ablation of selection, pooling, FP4 and SWA precision one factor at a time;
4. evidence-sparse workloads as well as RULER-style retrieval;
5. end-to-end cache bytes and FLOPs reported alongside quality.

The general framework that should precede that study is in
[`research/approximation-research-plan.md`](research/approximation-research-plan.md).
The current official-config comparison is
[`csa2-official-config.md`](csa2-official-config.md).

## 4. Triton-on-Metal end-to-end run

`docs/triton-on-apple.md` verifies the `triton-msl` plugin, its entry point, and the
unofficial Triton wheel for cp314/macOS 15+. The GPU run is deferred: it needs a 3.14
venv plus a 142 MB Triton wheel and a 127 MB torch wheel (see the doc for the exact
commands).
