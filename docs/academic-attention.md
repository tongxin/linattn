# The 2025–2026 academic frontier in attention and KV management

The industry-facing survey records what ships. This page records what the research
community is arguing *now*. It deliberately leads with 2025–2026 work; the 2019–2021
canon is kept at the end as lineage, not as the centre of gravity.

Venue labels follow the paper's own arXiv comment or the proceedings. A paper with no
stable venue is marked **preprint**, even when it is important. Numbers below are the
authors' reported results unless the row says *reproduced here*. Those are different
kinds of evidence and should not be pooled.

## 1. What changed since the 2020 synthesis

Four shifts matter more than the proliferation of acronyms.

| shift | old question | 2025–2026 question | representative work |
| --- | --- | --- | --- |
| **selection objective** | which attention entries are large? | which entries best preserve the matrix product `A V`? | MASA, OVAL, DropKV, ReST-KV |
| **selection cost** | how do we score every query-block pair? | how do we avoid materialising a `T × T` routing matrix? | PISA, JSSA, FFD, cross-layer index reuse |
| **state structure** | one matrix is enough | multi-state, mixture, uncertainty-aware, higher-order | Q-Delta, KDN, SwiLA, DLA, GSA2, Triadic, RunningTensor |
| **KV management** | evict or keep a token | placement, precision, basis, reuse, retrieval and hit rate | OVAL, BeaconKV, RoLaQ, Latent Compaction, CacheBlend, vAttention |

The old approximation tradition asked whether softmax can be reproduced with a bound.
The current field asks a more operational question: **what information does the next
query need, what is the cheapest representation of it, and what does a miss cost?**
That is why attention papers now look like systems papers, and why a credible survey
has to include the storage layer.

## 2. Selected 2025–2026 work

### 2.1 Selection and sparse attention

| work | venue | mechanism | evidence reported | in this repo |
| --- | --- | --- | --- | --- |
| **SAGA / Atop-N** | NeurIPS 2026 | asymmetric key/value head counts when sparse attention shifts the bottleneck to QK scoring | >2× long-context decode speedup; near-GQA quality up to 1.5B | `saga` |
| **PISA** | preprint 2026 | pyramid Top-K over hierarchical key summaries; `O(N log N)` routing | comparable commonsense, better retrieval than baseline | `pisa` |
| **JSSA** | preprint 2026 | shared low-rank QK subspace for token ranking; gather-free kernel | up to 6.43× full-KV decode throughput at 128K on B200; 96–99% of DSA throughput at 700B scale | `jssa` |
| **MASA** | preprint 2026 | score sparse units by reduction in `||A V||_F` error | RULER average 87.27 vs 87.00 for MInference; FlexPrefill-Llama 89.94 vs 89.18 | `masa` (analysis oracle) |
| **OVAL** | preprint 2026 | output-aware page bases from centred key–value coupling | +3.7–3.9 avg@k over FreeKV across three R1 models; LongBench-v2 29.82 / 28.03 on two models | `oval` |
| **SFA** | ICLR 2026 | sparse feature codes while retaining all tokens | up to 2.5× speed, ~50% FLOPs and KV reduction at matched quality | `sfa` |
| **FFD** | ICML 2026 | fused selector + decode kernel; top-delta block filtering | 11.6× kernel speedup, 2.37× end-to-end at 256K | cited, not reimplemented |
| **ARM** | ICML 2026 | differentiable routed fixed-size memory with Gumbel selection | better commonsense/long-context trade-off than fixed caching | cited, not reimplemented |
| **Virtual Head Attention** | NeurIPS 2026 | Q premix + low-rank postmix, folds to GQA-2 at inference | 4× smaller KV than GQA-8; RULER 84.4 vs 73.0 in one matched experiment | cited, not reimplemented |

### 2.2 KV eviction and compression

The 2026 "Random Attention" result was read too quickly. It shows that *raw attention
mass* is a weak selection signal on reasoning traces once the prompt and recent window
are protected. It does not show that eviction as a mechanism is dead. The newest
eviction papers changed the score to output perturbation or future-query coverage:

| work | venue | mechanism | evidence reported | in this repo |
| --- | --- | --- | --- | --- |
| **DropKV** | NeurIPS 2026 | decoupled per-token residual-output perturbation; smallest-perturbation tokens evicted | 19.8× scoring-kernel speedup, up to 9.9% end-to-end prefill; best RULER/LongBench at 5%/10% in its comparison | `dropkv` |
| **ReST-KV** | ICLR 2026 | layer-wise output reconstruction, temporal EMA, spatial pooling | +2.58 LongBench, +15.2 RULER, 10.61× decode latency reduction at 128K | `restkv` |
| **BeaconKV** | ICML 2026 | keep representatives of past query clusters to anticipate thought-revisiting queries | up to 5.8× memory reduction, >4.3× throughput, near-full-cache accuracy | `beaconkv` |
| **Random Attention** | preprint 2026 | protect prompt + recent window; randomise the rest | matches strongest tuned selector on six reasoning tasks; 32–43% higher vLLM throughput | `random_eviction` |
| **RoLaQ** | preprint 2026 | joint rotation and range calibration for 2-bit MLA KV | 3.96× BF16 KV capacity, 2.51× decode throughput at 256K on one model | cost model only |
| **Latent Compaction** | preprint 2026 | optimise a document-specific compact KV cache at test time | 96.7% RULER at 32× compression vs 40.0% for KVZip+ | cited, not reimplemented |

The honest synthesis is narrower than either camp claims: **attention-mass eviction is
a weak default; output-aware eviction is an active research line.** DropKV, ReST-KV and
BeaconKV are not variations on H2O's score; they add value-aware perturbations, future
query representatives, or a reconstruction objective. They also pay more scoring cost,
which is why fused kernels are part of the papers rather than an afterthought.

### 2.3 Linear attention and compressed state

| work | venue | what is new relative to GDN/KDA | in this repo |
| --- | --- | --- | --- |
| **Gated DeltaNet (GDN)** | ICLR 2025 | scalar gated decay plus the delta-rule erase | `gdn` |
| **Kimi Delta Attention (KDA)** | 2025 technical report | channel-wise decay; 3:1 hybrid with full attention | `kda` |
| **Gated DeltaNet-2** | 2026 preprint | separate channel-wise erase and write gates | `gdn2` |
| **Q-Delta** | ICML 2026 | the query's own value prediction becomes a second correction signal | `qdelta` |
| **Kalman DeltaNet** | 2026 preprint | track uncertainty and use the Kalman gain as write strength | `kdn` |
| **SwiLA** | COLM 2026 | route a mixture of delta-rule memories; nonlinear in the state | `swila` |
| **DLA** | ICML 2026 | information-aware merging of multi-state memory | cited, not reimplemented |
| **GSA2** | 2026 preprint | key-side Oja correction plus value-side delta correction | cited, not reimplemented |
| **Triadic Linear Attention** | 2026 preprint | third-order recurrent state from a triadic outer product | cited, not reimplemented |
| **RunningTensor** | 2026 preprint | generalise linear attention to order-`o` tensor states | cited, not reimplemented |

The architectural direction is clear: the matrix-valued state is no longer treated as
the endpoint. The live questions are how to allocate state capacity, how to correct it
without destroying old associations, and whether a mixture or higher-order state buys
recall more efficiently than a wider matrix. The papers disagree on the answer, and
most evidence is at 0.75B–1.5B parameters, not frontier scale.

## 3. What the benchmarks actually support

### 3.1 Retrieval-shaped benchmarks

RULER, LongBench, needle-in-a-haystack and InfiniteBench dominate the papers. These
benchmarks reward finding a small number of high-value keys, which is precisely the
regime where top-k block selection is nearly optimal. The 2025–2026 results are
therefore strongest for:

- page/block retrieval and synthetic multi-needle tasks;
- long-document QA where evidence is localised;
- prefill-heavy codebase or document understanding;
- video and diffusion attention, where block structure is even stronger.

They are weaker evidence for:

- multi-hop questions whose evidence is dispersed;
- long chains of thought that repeatedly revisit an early plan;
- agentic coding, where tool observations and source files compete for cache;
- deep research, where the model must synthesize many moderately relevant sources.

### 3.2 Coding, long-horizon reasoning and deep research

The evidence is still indirect. SAGA, DropKV, BeaconKV and FFD report long-context
decoding or prefill results, not end-to-end SWE-bench or agent success. OVAL reports
long reasoning gains (MATH500, AIME24, GPQA), which is closer to the target workload,
but it evaluates a compressed retrieval cache rather than a natively sparse model.
Random Attention evaluates reasoning traces but deliberately removes the selector.

The consistent pattern across the evidence is:

1. **Prompt preservation matters more than the score.** Every strong eviction result
   protects the prompt, sinks, or a recent window.
2. **Native training closes gaps that inference-time sparsity leaves open.** NSA/DSA/KDA
   results are not comparable to plug-in sparsification of a dense model.
3. **The scoring path becomes the bottleneck.** MASA, DropKV and FFD all spend most of
   their engineering effort on making selection cheap, not on the sparse attention
   multiplication itself.
4. **Compression and retrieval compose.** OVAL's page bases, RoLaQ's 2-bit latent and
   CSA2's cross-layer reuse attack different axes; the best system will use more than
   one.

### 3.3 The frontier-lab gap

Open-weight model reports publish mechanisms and bytes/token. Closed frontier labs
mostly publish pricing and cache behaviour. The observable signals for closed models
(prompt-cache discounts, cache-read pricing, long-context surcharges) are consistent
with **dense attention + KV reuse + quantisation**, not with a published native sparse
attention architecture. That is an inference from pricing, not a disclosure.

The academic frontier is therefore ahead on mechanisms and behind on deployment
evidence. A paper can be important without having landed in a frontier model, but its
benchmark claims should not be treated as deployment claims.

## 4. The older canon still worth knowing

These are foundations rather than the current selection:

| work | venue | why it still matters |
| --- | --- | --- |
| Linformer | ICML 2020 | the low-rank hypothesis behind feature-axis compression and MLA |
| Performer | ICML 2021 | the clean statement of kernelised attention; its variance explains why exact selection won |
| BigBird | NeurIPS 2020 | window + random + global as a universal-approximation argument |
| Reformer LSH | ICLR 2020 | hashing as routing; the ancestor of learned indexers |
| FlashAttention | NeurIPS 2022 | IO-awareness is the co-design thesis the whole kernel line inherits |
| H2O / SnapKV / TOVA | NeurIPS 2023/2024, ICLR 2024 | the eviction lineage, now baselines for output-aware methods |
| HyperAttention | NeurIPS 2023 | block-sparse plus low-rank for near-linear long-context attention |

The older approximation papers are still useful as controls. They are no longer a good
description of where the field is investing.

## 5. Systems and hardware: KV as a memory hierarchy

The attention kernel is only one term in the serving cost. The systems literature
reached this earlier than the modeling literature:

| work | venue | contribution |
| --- | --- | --- |
| Orca | OSDI 2022 | iteration-level scheduling and continuous batching |
| PagedAttention / vLLM | SOSP 2023 | paged KV with copy-on-write sharing |
| DistServe / Splitwise / Sarathi-Serve | OSDI / ISCA 2024 | disaggregate prefill and decode; chunked prefill |
| CachedAttention / AttentionStore | ATC 2024 | multi-turn and hierarchical KV reuse |
| Strata | SOSP 2024 | a real cache hierarchy for foundation-model KV |
| vAttention | ASPLOS 2025 | dynamic KV memory without paging, via virtual memory |
| CacheBlend | EuroSys 2025 | fuse cached knowledge for RAG instead of recomputing |
| Mooncake | ACM ToS 2025 | KV-cache-centric disaggregated serving |
| FFD | ICML 2026 | fuse selection and decode so metadata never becomes the bottleneck |
| PISA | 2026 preprint | make block routing itself log-linear |

Hardware work anticipated the bandwidth argument by years. SpAtten (HPCA 2021) already
framed attention as bandwidth-bound and pruned tokens in a cascaded pipeline; ELSA and
the sparse-attention accelerator line made the same bet. The modern kernel literature
re-derived the point with FlashAttention, FA3 asynchrony and FA4's FP4/TMA path.

## 6. What this repository reproduces

There are now 20 academic/analysis attention variants plus the random-eviction control,
and seven state mixers. The honour system is: state what is faithful, what is
approximated, and what refused to work.

| variant | source | status |
| --- | --- | --- |
| `linformer` | ICML 2020 | exact at identity projection; non-causal by design |
| `performer` | ICML 2021 | positive random features; slow convergence measured |
| `bigbird` | NeurIPS 2020 | exact at full window |
| `reformer_lsh` | ICLR 2020 | exact with one bucket; error grows with buckets |
| `free_solo` | 2026 | window + keypoint cells; learned pruning approximated |
| `namoh` | 2026 | head-owned token subsets; exact with one head |
| `hla_mixing` | 2026 | chunk-wise dynamic gate; paper's routing approximated |
| `mass_oracle` | COBS 2026 | exact attention-mass oracle |
| `saga` | NeurIPS 2026 | exact top-N inference semantics; key/value head asymmetry modelled |
| `pisa` | 2026 | hierarchical routing; mean-pooled nodes, exact selected-token attention |
| `jssa` | 2026 | key-covariance subspace; learned score-preservation objective approximated |
| `masa` | 2026 | exact output-error score as an analysis oracle |
| `oval` | 2026 | output-aware page bases; `G=I` |
| `sfa` | ICLR 2026 | top-magnitude feature code; learned codebook not reproduced |
| `dropkv` | NeurIPS 2026 | exact per-token perturbation score; fused Triton scorer not reproduced |
| `restkv` | ICLR 2026 | reconstruction score + EMA/pooling; layer-wise solve simplified |
| `beaconkv` | ICML 2026 | farthest-point beacon representatives; learned clustering approximated |
| `h2o`, `snapkv`, `tova` | 2023–2024 | eviction lineage kept as baselines |
| `qdelta`, `kdn`, `swila` | 2026 | reference recurrences; chunkwise kernels cited, not reproduced |

### Measured findings

- **Degenerate cases are exact.** SAGA, PISA, JSSA, MASA and OVAL all reduce to dense
  attention at an unrestricted budget; SFA does when every feature is kept. This is
  tested in `tests/test_frontier.py`.
- **Output-aware selection is not free.** MASA needs the dense profile to score
  blocks. Its value is as an upper bound: it tells you how much of a selector's gap is
  the score versus the sparse kernel.
- **Eviction is not one mechanism.** DropKV's output-perturbation selection differs
  from H2O's accumulated mass, and the test suite checks that they do not collapse to
  the same output.
- **Performer converges slowly.** Relative error went from ~0.51 at 16 features to
  ~0.46 at 1024 on random inputs. Unbiased does not mean low variance.
- **LSH is routing, not quality.** Error grows monotonically with bucket count; one
  bucket is exact because it is dense attention.
- **Nystromformer refused to work.** The pseudo-inverse of the landmark matrix blew
  outputs up 4–10× at small landmark counts; it is not shipped.
- **Hierarchical routing is the right complexity argument, not automatically the right
  wall-clock result.** PISA's Python reference has no fused kernel; treat its cost
  table as an algorithmic bound, not a GPU measurement.
- **The laptop-scale synthetic comparison is deliberately unflattering.** At
  `T=512`, a 128-entry budget and planted retrieval structure
  ([`reports/frontier_2026.md`](../reports/frontier_2026.md)), PISA had the lowest
  error (0.038), followed by OVAL (0.108) and JSSA (0.123); MASA's oracle was 0.181.
  SAGA's two-key-head allocation was much worse (0.841), and an untrained 16/64
  feature code for SFA was 0.908. The eviction variants all clustered around the
  random control (0.65–0.68), with SnapKV best in this particular run (0.576).
  These are not reproductions of the papers' training or benchmark settings. They
  are a warning that **architecture changes which need training do not become
  training-free just because a reference wrapper runs**, and that synthetic
  retrieval structure is not a substitute for a real coding or reasoning trace.

## 7. Open problems worth caring about

1. **A benchmark for dispersed evidence.** RULER does not measure the multi-hop,
   codebase-wide or long-research regime where top-k selection is weakest.
2. **Output-aware scores with cheap estimators.** MASA and OVAL point at the right
   objective; neither gives a selector that is both cheap and universally accurate.
3. **State capacity allocation.** Q-Delta, KDN, SwiLA, DLA and the higher-order state
   papers disagree about whether the next win is correction, routing, or more state.
4. **KV lifecycle economics.** Hit rate, TTL, placement and recomputation are now
   first-class variables. A kernel speedup that ignores them can lose end-to-end.
5. **Low-precision error geometry.** 2-bit MLA, FP4 KV and recurrent-state
   quantisation need evaluation on reasoning traces, not only perplexity.
6. **Training/inference co-design.** The strongest sparse results are trained natively.
   Training-free selection is a deployment convenience, not a free lunch.

## 8. How to read the two halves together

The industry half optimises **system cost at a fixed quality bar**: bytes/token, hit
rate, throughput. The academic half optimises **mechanisms and guarantees**: routing
complexity, approximation error, expressivity, state capacity. The productive question
is where they disagree.

Today the disagreement is sharpest around selection. Production assumes a learned
indexer can be accurate enough; COBS/MASA say the objective should be output error,
not mass; Random Attention says the score can be irrelevant once the prompt and recent
window are protected; DropKV/BeaconKV say the *right* score is not irrelevant. These
claims are testable against each other, and this repository is set up to make that
comparison rather than to declare a winner from one benchmark.

The KV layer adds the missing half of the cost model. See [kv-layer.md](kv-layer.md)
for the format, cache-policy and hit-rate instruments that turn an attention speedup
into an end-to-end claim.
