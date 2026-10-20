# Kernels and hardware-software co-design

This document covers the execution half of the problem: which kernels exist, what
they buy, and where the accounting gets hidden. It is also the guide to the
accelerated code in this repository.

---

## 1. Three regimes, three kernels

Attention cost is not one thing. The kernel that matters depends on the phase:

| phase | bottleneck | what the kernel must do | where the wins are |
| --- | --- | --- | --- |
| **prefill** (long prompt/tool output) | compute at short context, bandwidth at long context | keep Tensor Cores busy, one read of K/V per tile | block-sparse compute (MInference, FlexPrefill, XAttention), compressed groups (NSA), quantised GEMM |
| **decode** (long-horizon reasoning, agents) | memory bandwidth, always | read fewer KV bytes per query; keep the state resident | page selection (Quest), token selection (DSA), recurrent state (KDA) |
| **selection** (both) | now often the dominant term | cheap query-aware scoring without materialising `O(T²)` | indexer kernels, cross-layer reuse, predicted indexing, compressed indexing |

The third row is the 2026 story. Selection used to be a detail; at million-token
context it is the cost.

### 1.1 The arithmetic, in numbers

For decode with head dim `d = 128`, reading one KV entry costs `2·d = 256`
numbers = 512 bytes in fp16, and buys `4·d = 512` FLOPs (two matmuls). Arithmetic
intensity ≈ 1 FLOP/byte. An H100 SXM reaches ~3.3 TB/s of HBM bandwidth against
~67 TFLOP/s of fp16 tensor-core throughput, i.e. a ridge point near **20
FLOP/byte**. Decode attention therefore runs at ~5% of peak regardless of how
good the kernel is: **the only lever is reading fewer bytes.**

That single inequality explains:

* why every decode-side method in this survey reports "activated tokens per
  query" rather than FLOPs;
* why quantising the KV cache (fp16 → fp4 = 4x fewer bytes) is worth more than
  most algorithmic sparsity on decode;
* why the C kernel in `kernels/c/decode_attn.c` shows the same ordering as a GPU:
  the bytes dominate on a laptop CPU too, which is why running it is instructive.

The analytic table's FLOPs formula and its matmul-only exclusions are documented in
[analytic-flops.md](analytic-flops.md). In particular, the current dense base
estimate counts the full `T × T` matrix; causal-triangle estimates are about half.

## 2. What the kernels actually do

### 2.1 Block-sparse execution (`kernels/triton/block_sparse.py`)

The observation that unifies Quest, MInference, XAttention, MoBA, NSA's selection
branch, DSA/CSA and SeerAttention-R: **they differ only in which blocks they put
in the list; the attention over those blocks is the same kernel.** The kernel here
is a decode-shaped online-softmax loop over selected KV blocks (one program per
query token, query head and batch; the KV head is derived from the group so that
GQA heads share a block list).

This separation is what makes the "selector" research composable: a better
selector is a drop-in, and the execution cost is a function of the number of
selected blocks, not of the selector's identity.

### 2.2 Selection kernels

**Quest** (`kernels/triton/quest_pages.py`) needs two primitives: an incremental
per-page min/max update (touches `O(d)` numbers per appended token) and a page
scoring pass. The metadata is `2·d` numbers per page *per KV head*, so at page size
16 the metadata is `d/8` numbers per token -- negligible against the `2·d` per
token it replaces. The kernel here writes the full page-score vector and leaves
top-k to the caller; fusing the top-k is the remaining step, and
`fused_indexer.py` is the template for it.

**DSA/CSA indexer** (`kernels/triton/fused_indexer.py`) implements the two-pass
threshold selection that avoids materialising the `[B, T_q, T_k]` score tensor:
pass 1 writes one maximum per key block per query, the threshold is the k-th
largest block maximum, pass 2 emits every pair above it (a superset of the true
top-k). The trade is explicit: two reads of the keys instead of one, in exchange
for never allocating the score matrix. The failure mode this avoids is not
hypothetical -- the public CSA implementations allocate 256 GB for the score
tensor at 65k tokens with V4-Flash indexer dimensions and OOM on an H200.

The natural next steps, both visible in the 2026 literature, are (a) *cross-layer
reuse* of the selected indices (IndexCache: Full layers run their own indexer,
Shared layers reuse the nearest Full layer's top-k) and (b) *prediction* instead
of scoring (SPIN: predict important blocks from history and verify cheaply).
Both change the kernel interface from "score everything" to "propose a small
candidate set, verify it".

### 2.3 The 2026 kernel frontier

Three newer designs are worth separating from the generic block-sparse path:

* **FFD** (ICML 2026) fuses the selector and the decode kernel, using low-bit
  content-aware scans and a top-delta filter so block metadata never round-trips
  through HBM. This is the clearest hardware-software co-design result in the
  current selection line; it is cited but not implemented here because the
  contribution is the fused CUDA path.
* **PISA** (2026) makes the *routing* `O(N log N)` with a pyramid Top-K over
  pooled keys. The reference in `variants/frontier.py` implements the hierarchy;
  a real kernel needs to fuse hierarchical selection with the attention loop,
  which the repository does not claim.
* **SAGA** (NeurIPS 2026) observes that sparse decoding can shift the bottleneck
  from value reads to QK scoring and changes the key/value head allocation. Its
  kernel consequence is asymmetric GQA metadata and a selector whose head
  allocation is not the usual `n_heads / n_kv_heads` partition.

The lesson common to all three: selection is now a **memory-layout** problem as
much as a modelling problem. A selector that produces a beautiful block list but
forces a synchronisation or a metadata read per layer can lose to a simpler one
that stays fused.

### 2.4 Chunked delta-rule scan (`kernels/triton/delta_chunk.py`)

The compressed-state family needs one kernel that turns the sequential recurrence
`S_t = Diag(α_t)(I - β_t k_t k_tᵀ)S_{t-1} + β_t k_t v_tᵀ` into something
parallelisable across the sequence. The chunkwise form used here (derived in
`variants/recurrent.py` and validated against the recurrence in
`tests/test_recurrent.py`) needs, per chunk:

1. elementwise decay normalisation,
2. three `C×C` matrix products,
3. **one triangular solve**,
4. two more `C×C` products for the outputs.

Everything except the solve is `tl.dot`-shaped; the solve is the only
sequential-in-chunk part, which is why the algorithm is a *chunked scan* rather
than a pure matmul. Two implementation notes that matter for anyone reproducing
this:

* **Numerics.** The matrices contain `exp(gc_{i-1} - gc_j)`, which grows with the
  decay accumulated inside the chunk. Production kernels bound the log-decay
  ("lower-bounded decay" in KDA) and accumulate in fp32; this repository exposes
  `decay_clamp` and runs the equivalence tests in float64 so that algorithmic
  correctness is not confounded by conditioning. If a chunked implementation
  diverges from the recurrence, suspect this term first.
* **Asymmetric erase factors.** With channel-wise decay, the erase operation's
  left factor (`a_i = k_i/G_{i-1}`), right factor (`b_i = k_i·G_{i-1}`) and write
  factor (`c_i = k_i/G_i`) are all different. Gated DeltaNet-2's abstract makes
  the same point from the modelling side ("decoupling erase and write"); the
  kernel consequence is that a naive WY transform assuming one shared factor will
  be wrong. Our derivation keeps the three factors separate.

**Status of the kernels in this repo.** The Triton kernels are written against
stable Triton APIs and are exercised through the parity harness
(`tests/test_kernels.py`) that runs the torch fallback everywhere and the Triton
path when CUDA is available. They have *not* been executed on this machine (no
NVIDIA GPU); the C kernel has. The kernel-vs-reference parity tests are marked
`gpu` and skipped automatically when no GPU is present, so the claim "the kernels
agree with the reference" is testable but not claimed here for CUDA.

For a Mac GPU the options are the MLX backend ([mlx.md](mlx.md)) or the third-party
`triton-msl` compiler ([triton-on-apple.md](triton-on-apple.md));
`python -m linattn.cli backends` reports whether either is installed.

## 3. The compiler path (the part most teams should use)

Hand-written kernels are the wrong first move for most teams. The right move is
to express the selection as a **block mask** and let the compiler generate the
kernel:

* `kernels/flex_mask.py` converts a variant's token-level selection into a
  `torch.nn.attention.flex_attention.BlockMask`, and *verifies* that the block
  reduction is a superset of the selection (FlexAttention cannot express
  sub-block sparsity, so a selection that keeps one token per block still costs a
  whole block read).
* That verification is worth running before benchmarking anything: the block-level
  coverage number tells you immediately how much of your selection precision the
  kernel interface throws away, which is often the difference between a 4x and a
  1.5x practical speedup.

This is also what makes the co-design literature intelligible: NSA's insistence on
*block-aligned* selection, LongCat LSA's streaming-aware layout conversion, and
MInference's 1x64 blocks for the vertical-slash pattern are all responses to the
gap between what a selector wants and what the execution layer can do cheaply.

## 4. Offload, heterogeneity and quantisation (2026 additions)

* **KV offload to CPU/SSD** becomes attractive once attention reads only a few
  percent of the cache: the offload papers (SSD-based sparse KV storage, CPU-GPU
  load balancing of the cache, processing-near-memory decode) all assume sparsity
  as the enabler. The new bottleneck they report is *finding* the entries to move.
* **Quantisation interacts with selection.** FP4 KV (V4.1-Flash) reduces the bytes
  per entry 4x, which is comparable to a 4x sparsity win and does not require a
  selector. Low-bit recurrent state (LeapQuant) does the same for the compressed
  family. Expect selection and quantisation to be co-tuned rather than chosen.
* **Precision failures are model-quality failures.** The bf16-gradient instability
  reported for FlashAttention-3 late in training (arXiv:2609.34272) is the
  clearest recent example; on the sparse side, the analogous risk is a selection
  that is numerically fine in fp32 and unstable in the training precision.

## 5. How to reproduce the measurements in this repository

```bash
python -m linattn.cli backends                     # what this machine can run
python bench/bench_decode.py --seq 100000 --budget 2048   # bytes-read scaling, C kernel vs torch
python bench/run_suite.py --seq 1024 --budget 128         # error/recall per variant
python -m pytest tests -q                                 # correctness gates
```

The benchmark scripts write JSON plus a markdown table into `reports/`, and they
print the backend report at the top so that a result is always attributable to a
specific machine and backend.
