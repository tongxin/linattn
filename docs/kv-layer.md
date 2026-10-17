# The KV storage and data-management layer

Attention research asks *how many* KV entries a kernel reads. Production asks a
larger question: **which bytes are in which tier, how did they get there, and what
does a miss cost.** By 2026 the second question is often the bigger one, which is
why this repository treats the KV layer as a first-class part of the survey rather
than as a serving detail.

---

## 1. Why the layer became first-class

Four observations, each from a shipped system rather than a benchmark:

1. **Footprint is the binding constraint on long context.** DeepSeek-V4.1-Flash
   reports 890 bytes/token of global KV (CSA2 + cross-layer reuse + FP4), about a
   quarter of its predecessor. Kimi Linear reports a 75% KV reduction from its 3:1
   KDA hybrid. These are headline numbers in model reports now, alongside accuracy.
2. **Agentic workloads are input-heavy and re-read.** DeepSeek frames V4.1-Flash
   around "increasingly input-heavy" long-horizon agents; the cost is prefill and
   KV storage, not decode FLOPs.
3. **Recompute after idle gaps dominates some sessions.** In a public analysis of
   393 real agent sessions, requests arriving after the provider's 5-minute TTL
   accounted for 17.5% of recompute, and three "smarter than LRU" policies all lost
   to plain radix-leaf LRU. A cache policy change is therefore a *cost* change.
4. **The substrate is already paged.** PagedAttention/RadixAttention style block
   management is what every serving stack assumes, and sparse attention's block
   selection was designed to fit it. Algorithm and allocator are co-designed.

## 2. The building blocks

| block | what it decides | representative work |
| --- | --- | --- |
| **paged allocator** | block size, copy-on-write sharing, fragmentation | PagedAttention (arXiv:2309.06180) |
| **prefix / radix cache** | which prefixes are shared across requests, and how they are matched | SGLang RadixAttention (arXiv:2312.07104) |
| **admission + eviction** | the hit rate under a capacity budget: LRU, TTL-LRU, FIFO, S3-FIFO, Belady | the agentic-cache analysis; LRU-vs-paper debate |
| **tiering / offload** | what lives in HBM vs DRAM vs SSD, and the promotion path | InplaceKVCache (arXiv:2609.14507), SSD-backed sparse KV (arXiv:2609.36938), PIM decode (arXiv:2608.03555) |
| **compression / quantisation** | bytes per token: latent (MLA), groups (CSA), fp8/fp4 | DeepSeek-V4.1-Flash (arXiv:2609.19969), LeapQuant (arXiv:2609.38166) |
| **cross-layer reuse** | whether every layer needs its own index/selection | IndexCache (arXiv:2603.12201), LongCat LSA (arXiv:2608.01662) |
| **state checkpoints** | how hybrids snapshot a recurrent state for prefix reuse | DASC (arXiv:2608.30386), ReplaySSM (arXiv:2610.13778) |
| **memory-guided eviction** | using the recurrent state to decide what attention keeps | GateSight (arXiv:2610.14754), DeltaS (arXiv:2609.27470) |
| **transfer / disaggregation** | moving KV between prefill and decode workers | Star Attention (arXiv:2411.17116), ESS (arXiv:2512.10576) |
| **sparse-first engine** | one loop over heterogeneous cache representations | SparseEngine (arXiv:2609.39068), HySparse2 (arXiv:2609.26368) |

Three interactions are easy to miss:

* **Compression and sparsity trade against each other.** FP4 KV cuts bytes 4x and
  needs no selector; block sparsity cuts *reads* and also saves compute. The model in
  this repo reports both so the choice is visible.
* **A prefix cache turns compute into bandwidth.** A hit avoids prefill (quadratic)
  and pays a read (linear). That is why hit rate is not a "cache statistic" but a
  first-order cost parameter.
* **Hybrids change what a cache *is*.** A recurrent layer's cache is a fixed state,
  so prefix reuse needs state *checkpoints*, not blocks -- and the state's size is
  independent of context length, which changes the tiering calculus completely.

## 3. The instrument in this repository

```
src/linattn/kv/
  format.py   bytes/token for dense fp16, fp8, MLA, CSA2, and fixed-size state
  cache.py    block prefix cache: LRU / TTL-LRU / FIFO / random + Belady oracle, trace simulator
  trace.py    synthetic agentic sessions (shared prompt, tool output, idle gaps)
  cost.py     roofline breakdown: attention, MLP, KV write, promotion, recompute, lookup
```

The cost model reports a **breakdown and names the dominant term**, because that is
the actionable output:

```
attn        max(attention FLOPs / peak, KV bytes read / bandwidth)
mlp         non-attention transformer work (MLP + projections)
kv_write    newly computed KV written to the hot tier
promote     cached prefix pulled from the tier it lives in
recompute   (1 - hit_rate) x prefill work  <-- the miss penalty
lookup      prefix-cache block lookups
```

`hit_rate` is an explicit input (the "artificial feature parameter"): sweep it, or
measure it from a trace with the simulator and feed the result back in.

```bash
python -m linattn.cli kv-cost --prompt 131072 --hit-rate 0.8 --format mla --cached-tier dram
python experiments/kv_economics.py          # -> reports/kv_economics.md
```

## 4. What the instrument says (CPU, analytic model)

From `reports/kv_economics.md`, 61 layers, 32 KV heads x 128, H100-class bandwidths,
90% prefix-cache hit rate, 4k output:

| format | context | KV GiB | attention ms | MLP ms | total ms | dominant |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense fp16 | 32k | 8.58 | 678 | 915 | 1,683 | MLP |
| dense fp16 | 128k | 31.45 | 9,214 | 3,354 | 12,993 | attention |
| dense fp16 | 1M | 244.95 | 559,325 | 26,117 | 593,698 | attention |
| MLA | 1M | 68.89 | 559,325 | 26,117 | 593,642 | attention |
| CSA2 | 1M | 0.87 | 559,325 | 26,117 | 593,620 | attention |
| KDA state | 1M | **0.03** | **272** | 26,117 | 29,044 | MLP |

Four conclusions, each of which changes where effort should go:

1. **Compression does not reduce attention FLOPs.** MLA and CSA2 cut footprint by
   3.6x and 280x versus dense, and leave `attention_ms` *identical* -- they are not a
   compute fix. If a workload is compute-bound, only sparsity or a linear-time
   algorithm helps.
2. **At 8k-32k the MLP dominates**, even with dense attention. Optimising attention
   there is optimising the smaller term.
3. **Sparsity flips the dominant term, but only down to the MLP floor.** With MLA at
   128k, 10% KV reads move attention from 9.2 s to 0.9 s and the MLP (3.4 s) becomes
   the wall; going to 2% buys little. That is the "sparsity has a knee" result, visible
   in one table.
4. **The recurrent state removes the quadratic term entirely** (272 ms vs 559 s at
   1M) and leaves a 0.03 GiB footprint -- which is exactly why the frontier hybrids
   pair it with a few full-attention layers rather than replacing attention outright.

For the cache layer itself, the trace (40 agentic sessions, 12 turns, mean 60 s idle
gaps, 16-token blocks) gives:

| policy | capacity | hit rate | recompute tokens |
| --- | ---: | ---: | ---: |
| LRU | 256 blocks | **0.886** | 41,632 |
| TTL-LRU (300 s) | 256 | 0.883 | 42,784 |
| FIFO | 256 | 0.836 | 59,904 |
| random | 256 | 0.290 | 259,168 |
| Belady oracle | 256 | 0.886 | -- |

LRU is at the Belady bound and random collapses, reproducing the published
"LRU is hard to beat" result; the TTL costs 0.3 percentage points of hit rate and
~2.8% more recompute, which is the same mechanism the 17.5% figure measures at
larger scale.

## 5. Extending the instrument

* **Add a format**: one `KVFormat(...)` entry in `kv/format.py` (bytes/token, or a
  fixed per-sequence state); it is then available to the cost model and the CLI.
* **Add a policy**: implement eviction in `BlockCache._evict_one` and add the name to
  `CacheConfig`; the simulator and reports pick it up.
* **Use a real trace**: any iterable of objects with `blocks` (prefix-inclusive block
  ids) and `at` (arrival seconds) works with `kv.simulate`.
* **Couple to a kernel**: the `kv_read_fraction` parameter is where a sparse
  attention variant's measured read budget plugs in, so a kernel result becomes a
  cost result without changing the model.

## 6. What this model deliberately does not do

It is analytic, not a simulator of a specific engine: no continuous-batching queueing,
no P/D disaggregation latency, no SSD queue depth, no compression artefacts on
quality. Those matter for capacity planning and are better measured. What it *is* for
is the question the survey keeps returning to -- **which term dominates, and did my
change move it** -- and that question it answers exactly.

## 7. CSA2: what it is, and what this repository actually implements

**CSA2 = Compressed Sparse Attention 2**, the attention/KV design in
DeepSeek-V4.1-Flash (arXiv:2609.19969). It is the composition of three mechanisms,
not a single algorithm:

| component | what it does | in this repo? |
| --- | --- | --- |
| **compressed indexing** | score *pooled* key groups instead of every token, then expand the selected groups back to tokens -- this is what makes the indexer cheap enough for every layer | **yes, as a primitive**: `DSAAttention(compress_ratio>1)` (torch) and `_csa_pages` (MLX) |
| **cross-layer KV reuse** | one cached KV copy (or one index set) per *group* of layers instead of per layer; the surveyed papers are IndexCache (arXiv:2603.12201) and LongCat LSA (arXiv:2608.01662) | **no** -- neither the reuse schedule nor a shared-KV layer exists; the cost model can only *assume* a smaller footprint |
| **FP4 KV** | store the latent at 4 bits per value with per-group scales, roughly halving bytes versus fp8 | **no** -- there is no quantiser; the `csa2` preset just hard-codes the reported 890 B/token |
| **reported outcome** | 890 bytes/token of global KV at 1M context, ~1/4 of V4-Flash | a **cost-model constant** (`format.py::PRESETS["csa2"]`), used in `reports/kv_economics.md` |

So: the *indexing* idea is implemented and tested; the *storage* ideas (cross-layer
sharing, FP4) are only modelled. Do not read the `csa2` row in the economics table as
"we implemented CSA2" -- it is the published footprint plugged into the cost model so
that the storage-vs-compute trade-off can be compared. `docs/roadmap.md` lists what a
real CSA2 layer would need.
