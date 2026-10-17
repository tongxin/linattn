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
actually reads a representative's cache (`kv/csa2.py`), and composed footprints that
reproduce/bracket the reported 890 B/token (`experiments/csa2_model.py`).

What a *released-architecture* claim would still need:

* **trained weights** -- the indexer projections are random unless fitted, so
  selection quality is not DeepSeek's;
* **the exact layer schedule** -- which layers are "full" in V4.1 is not public; our
  plan shares uniformly every `group_size` layers;
* **a hardware FP4 path** -- the codec simulates packed 4-bit accounting but the
  arithmetic is FP32 after dequantisation; a real kernel keeps FP4 in registers/Tensor
  Cores (and the numerics differ);
* **real-workload validation** -- the sharing-divergence sweep uses synthetic layers;
  the trained regime (neighbouring layers trained to share) is the one that matters.

## 4. Triton-on-Metal end-to-end run

`docs/triton-on-apple.md` verifies the `triton-msl` plugin, its entry point, and the
unofficial Triton wheel for cp314/macOS 15+. The GPU run is deferred: it needs a 3.14
venv plus a 142 MB Triton wheel and a 127 MB torch wheel (see the doc for the exact
commands).
