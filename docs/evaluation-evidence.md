# Evaluation evidence: what these methods actually score, and where the evidence is thin

This document collects the *reported* numbers for the variants in
[`variants/`](variants/), organised by the workload the survey cares about
(general long-context, coding, long-horizon agentic, reasoning, deep research).
Every number here is quoted from a primary source, with the configuration it was
measured under, because the configurations differ enough that the numbers are not
directly comparable with each other.

> **How to read these tables.** Sparse-attention results are extremely sensitive
> to (a) whether the model was *trained* with the pattern, (b) the token budget
> ("activated tokens per query"), (c) whether the first layers are kept dense, and
> (d) the benchmark's mass distribution. A "+0.05 on RULER" and a "-0.02 on AIME"
> can both be true of the same method. Numbers below include their setting for
> this reason.

---

## 1. General long-context

| method | training | setting | reported result |
| --- | --- | --- | --- |
| **NSA** (arXiv:2502.11089) | native (pretrained with the pattern) | 27B total / 3B active, 32k sequences, 2560 activated tokens per query | LongBench **0.469** vs full attention **0.437** vs exact top-k at the same budget **0.423**; multi-hop QA HPQ **+0.087** and 2Wiki **+0.051** over full attention; code (LCC) **+0.069**; passage retrieval (PassR-en) **+0.075** |
| **MInference 1.0** (arXiv:2407.02490) | training-free | LLaMA-3-8B-1M, A100, 1M-token prompts | up to **10x** prefill speedup (30 min → 3 min); 128k attention is up to **96.8%** sparse; top-4k columns recover **96.8%** of attention mass; accuracy maintained across InfiniteBench/RULER/PG-19/NIAH on 5 long-context models |
| **XAttention** (arXiv:2503.16428) | training-free | stride 4, threshold 0.9-0.95, per-head thresholds from offline DP | up to **13.5x** attention speedup; RULER and LongBench parity with full attention; video (VideoMME) and video generation (VBench: PSNR 23.5, SSIM 0.822, LPIPS 0.155 at >50% sparsity) |
| **MoBA** (arXiv:2502.13189) | continued pretraining (100B tokens with the pattern active) | Llama-8B-1M-MoBA, block size 4096, top-k = 12 | RULER **0.7818** vs full-attention Llama-8B-1M-Full **0.7849** at up to ~95% sparsity; needle-in-a-haystack up to 1M context |
| **Quest** (arXiv:2406.10774) | training-free | page size 16, 4096-token budget, later layers only (first 2 layers dense) | claims near-lossless long-context quality with a large reduction in KV movement; the paper's own guardrails are the interesting part: early layers are dense because their sparsity is low |
| **Kimi Linear (KDA)** (arXiv:2510.26692) | native, 1.4T tokens | 3B active / 48B total, 3:1 KDA:MLA hybrid | RULER-128k **84.3** vs full MLA **81.3**; MMLU-Pro (4k) **51.0** vs MLA **47.2**; KV cache **-75%**; TPOT at 1M **1.84 ms** vs MLA **11.48 ms** |
| **Random Attention** (arXiv:2609.03430) | training-free | 4 models, 6 reasoning tasks | matches the strongest scored baseline while delivering **32-43%** higher throughput in vLLM |

The pattern across the table: **methods trained with the pattern report quality
gains; methods applied post-hoc report parity.** That is not a coincidence, and it
is the single most useful thing to know when reading a new paper in this area.

## 2. Coding

Coding evidence is thin, and it splits by task shape:

* **Code understanding over a long file/repo** rewards sparse attention, because
  most of the context is genuinely irrelevant to the current question. NSA
  reports **+0.069 on LCC** at equal budget (above), and MInference-class methods
  report parity on long-code subsets of RULER.
* **Agentic software engineering** (SWE-bench-style) is a different shape: the
  model must combine a few *specific* retrieved lines with a lot of boilerplate
  and tool output, often across many turns. Here the published evidence is about
  *context pruning and cache management*, not attention sparsity:
  SWE-Pruner-style pruning (arXiv:2607.18213), HySparse2's two-level KV sharing
  for "short actions + long observations" (arXiv:2609.26368), and SparseEngine's
  heterogeneous cache handling (arXiv:2609.39068).
* **Cost, not accuracy, dominates practitioner discussion.** In the DeepSeek-V3.2
  release thread the recurring practical objection was whether the model can run
  at a usable context length on consumer hardware at all -- which is why the
  frontier moved toward *storage* per token (890 bytes/token in V4.1-Flash) rather
  than only attention FLOPs.

**Verdict for coding:** sparse prefill is safe and useful for repository-scale
context; sparse *decode* for long agentic traces is where the risk is, and the
mitigation that currently works is keeping exact attention on a minority of layers
(the hybrid pattern) plus a small always-on local window plus the sink tokens.

## 3. Long-horizon agentic workloads

These are input-heavy, tool-mediated, and re-read: the dominant costs are prefill
and KV storage/reuse, in that order.

| source | claim relevant to long-horizon |
| --- | --- |
| DeepSeek-V4.1-Flash (arXiv:2609.19969) | agentic input-heaviness motivates a causal encoder-decoder that activates **16B params on decode but only 8B on prefill**; global KV footprint **890 bytes/token** via CSA2 (cross-layer reuse) + FP4 KV, ~1/4 of V4-Flash |
| HySparse2 (arXiv:2609.26368) | long-horizon agents produce short actions but long observations; proposes two-level KV sharing for prefill and compaction |
| IndexCache (arXiv:2603.12201) | top-k selections from DSA are highly similar across adjacent layers; reusing indices across layers removes most indexer cost, training-free or training-aware |
| SPIN (arXiv:2610.09025) | predicted (not scored) indexing reaches 30-40% sparsity with **+14.9%** output throughput and **-13.2%** median inter-token latency in vLLM |
| LongCat LSA (arXiv:2608.01662) | streaming-aware + cross-layer + hierarchical indexing scales from 69B-A3B to 560B-A27B "on par with full attention" |
| Agentic KV-cache study (public analysis, 393 real Claude Code sessions) | requests arriving after a gap longer than the provider TTL account for **17.5%** of recompute; three "smarter than LRU" policies all lost to plain radix-leaf LRU. Relevant because it bounds what any attention optimisation can buy if cache *reuse* is what dominates |

The V4.1-Flash report also gives whole-model evidence that the compressed design
did not simply collapse capability. Selected V4.1-Flash vs V4-Flash values from
the report are:

| benchmark | V4.1-Flash | V4-Flash |
| --- | ---: | ---: |
| Codeforces rating | 3471 | 3289 |
| MathArena Apex pass@1 | 65.6 | 58.6 |
| Terminal-Bench 2.1 | 90.6 | 82.7 |
| Terminal-Bench 3.0 | 30.0 | 7.6 |
| DeepSWE v1.1 resolved | 74.2 | 54.4 |
| HLE pass@1 | 36.8 | 37.8 |

These are **model-level, not CSA2-isolated** results: the release also changes
CED, data, post-training, parameter count and deployment. The report says FP4
main KV caused “only marginal performance degradation” and retains FP8 for the
sensitive SWA cache, but it does not publish a layerwise dense-vs-CSA2 attention
ablation. See [csa2-approximation.md](csa2-approximation.md).

**Verdict for long-horizon agents:** the interesting wins in 2026 are in
selection overhead (IndexCache, SPIN) and KV footprint (FP4, cross-layer reuse),
not in attention FLOPs. A 10x attention-kernel speedup barely moves an agent
workload whose cost is 80% prefill and cache management.

## 4. Reasoning and test-time scaling

Two directly opposed pieces of evidence, both credible:

* **Selection scores barely matter for reasoning.** Random Attention
  (arXiv:2609.03430) evicts uniformly at random (keeping prompt + recent window)
  and matches the strongest tuned selector on six reasoning tasks across four
  models, with 32-43% higher throughput. Its controlled experiments attribute the
  entire gap between selectors to *whether they happened to keep the prompt*.
* **Native sparse training improves reasoning.** NSA-R, distilled from R1 on
  32k reasoning traces, beats its matched full-attention R model on AIME:

  | generation limit | Full Attention-R | NSA-R |
  | --- | --- | --- |
  | 8k | 0.046 | **0.121** |
  | 16k | 0.092 | **0.146** |

  Both models improve with a longer generation budget, and NSA's advantage
  persists -- i.e. sparse attention did not truncate the benefit of thinking
  longer.

The reconciliation: *eviction* of reasoning traces is risky and largely
unnecessary (the prompt is what matters), while *native sparse attention* changes
what the model learns to attend to and can help. Also worth noting on the
efficiency side: the same DeepSeek release discussion flags that its deepest
reasoning variant emits ~50% more tokens than its sibling (3.5x Gemini 3 on one
Codeforces-style comparison), so reasoning gains and token-efficiency gains
trade off against each other.

## 5. Deep research / multi-hop

Multi-hop QA is the discriminator between families:

| family | expected behaviour on multi-hop | evidence |
| --- | --- | --- |
| hard top-k token selection (DSA-style, tuned k) | worst case: evidence spread over several moderate-mass passages, budget spent on one of them | implied by construction; DSA mitigates with a large k (2048) and a compressed branch in CSA |
| block selection with a coarse branch (NSA) | best reported: coarse branch provides global coverage, fine branch provides precision | HPQ **+0.087**, 2Wiki **+0.051** over full attention |
| eviction (H2O/SnapKV) | risky: evicted evidence is unrecoverable, and long research sessions revisit early material | mitigated in practice by never evicting the prompt (Random Attention's control experiment) |
| compressed state (KDA/GDN hybrids) | weakest in theory (finite state cannot do exact recall), rescued by the full-attention layers | MQAR-style probes show the recall gap; `experiments/hybrid_recall.py` reproduces the shape of it at small scale |

## 6. Kernel-level and serving-level numbers

Attention-only speedups are the most commonly quoted and the least useful number:

| method | attention speedup | end-to-end evidence |
| --- | --- | --- |
| NSA | 9.0x forward / 6.0x backward at 64k (Triton, A100) | none published in the paper |
| MInference 1.0 | up to 10x prefill at 1M (30 min → 3 min) | end-to-end prefill latency (this *is* nearly end-to-end for 1M prefill) |
| XAttention | up to 13.5x attention | RULER/LongBench/VBench accuracy, not latency |
| MoBA | ~(1 - sparsity) attention time at 1M | not reported end-to-end |
| DSA (V3.2) | release reported latency reduction for long-context inference | third-party commentary confirms cheaper inference |
| SPIN | indexer overhead removal | **+14.9% output throughput, -13.2% median ITL** (vLLM, end-to-end) |
| StreamIndex | avoids OOM at 65k (naive CSA materialises 256 GB with V4-Flash dims) | layer-level |

The honest planning number for a production system is SPIN-class: single-digit to
low-double-digit percent end-to-end throughput for an indexer optimisation, plus
whatever the KV footprint reduction buys in batch size (which is often the larger
term: V4.1-Flash's 4x smaller KV cache multiplies served batch size, and Kimi
Linear's 75% reduction combined with a 1M-context TPOT of 1.84 ms vs 11.48 ms
changes what is servable at all).

## 7. Where the public evidence is inadequate

1. **No agentic-sparsity protocol.** Nothing isolates "how much sparsity before
   an agent's task success rate drops". SWE-bench's variance swamps the effect at
   accessible model sizes.
2. **Different budgets are compared as if equal.** "2560 activated tokens" (NSA),
   "4096 tokens" (Quest), "top-2048 tokens" (DSA) and "top-12 blocks of 4096"
   (MoBA) are wildly different budgets; cross-paper accuracy comparisons are
   mostly meaningless.
3. **Almost all evaluations use a single needle or a single hop.** Multi-hop and
   many-needle variants exist (RULER has some; 2Wiki/HPQ in NSA; "deep research"
   tasks in agentic suites) but are rarely used for sparse-attention ablations.
4. **Serving numbers are rarely end-to-end.** See the table above.
5. **Numerical claims are rarely reported.** The bf16-gradient result
   (arXiv:2609.34272) shows kernel numerics can break training; almost no sparse
   attention paper reports the precision-sensitivity of its selection or of its
   compressed state.

## 8. What this repository measures, and what it does not

Measured here (CPU, reproducible on any machine):

* error against dense attention and attention-mass recall for every implemented
  variant, at several budgets (`bench/run_suite.py`, `reports/`);
* the exactness of the chunked linear-attention algorithms against their
  sequential recurrence, in float64 and in gradients
  (`tests/test_recurrent.py`);
* whether a random-init learned selector is useless and how much a fitted one
  recovers (`variants/selection.py::DSAAttention.fit`,
  `SeerAttentionR.fit_gate`, `variants/native_sparse.py::NSAAttention.fit_gates`);
* associative-recall accuracy versus persistent state size for hybrids
  (`experiments/hybrid_recall.py`) and the *deterministic* capacity curve of a
  fixed-size delta-rule state (`experiments/state_capacity.py`: cosine recall of
  random associations falls from ~0.99 at 4 associations to ~0.12 at 1024 with an
  8192-float state);
* decode attention time versus bytes read, with an explicit IO-aware tiling
  comparison (`bench/bench_decode.py`, using the C kernel).

Not measured here: anything requiring a multi-GPU serving stack or a frontier
checkpoint. Those numbers are cited, above, with their sources.
