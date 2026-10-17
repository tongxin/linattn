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

## 3. A real CSA2 layer (cross-layer KV reuse + FP4 latent)

Currently the repo has the *compressed-indexing* primitive (`DSAAttention` with
`compress_ratio > 1`, and `_csa_pages` on MLX) and a cost-model constant for the
reported 890 B/token (`kv/format.py::csa2`). What is missing to claim CSA2:

* **cross-layer KV reuse**: a layer group where one member computes and stores the
  compressed KV, and the others read it. Needs a layer-loop harness (the current
  `(q, k, v)` contract has no notion of "layer N reads layer M's cache") and a
  schedule (which layers are "full", as in IndexCache);
* **FP4 storage**: a per-group-scale quantiser/dequantiser for the latent (E2M1 with
  an absmax scale is the standard choice), plus a numerical test of the quality
  impact -- the cost model currently only assumes the byte count;
* **validation** against the per-layer baseline: bytes/token, attention output
  divergence, and the cost breakdown recomputed with the real numbers rather than the
  published constant.

## 4. Triton-on-Metal end-to-end run

`docs/triton-on-apple.md` verifies the `triton-msl` plugin, its entry point, and the
unofficial Triton wheel for cp314/macOS 15+. The GPU run is deferred: it needs a 3.14
venv plus a 142 MB Triton wheel and a 127 MB torch wheel (see the doc for the exact
commands).
