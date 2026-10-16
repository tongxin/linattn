# Sparse, compressed-state and co-designed attention: a 2026 survey and gap analysis

*Scope: literature and releases up to mid-October 2026. Companion documents:
[evaluation evidence](evaluation-evidence.md), [the frontier gap](frontier-gap.md),
[kernels and co-design](kernels.md), and one page per variant in
[`variants/`](variants/). All quantitative claims about this repository's own
measurements come from `reports/`; everything else is cited.*

---

## 1. Executive summary

**The field has consolidated around four distinct mechanisms, and the last 18
months have been about making them *trainable* and *kernel-shaped* rather than
about inventing new sparsity patterns.**

1. **Token-level eviction is being abandoned as a quality lever.** Every
   score-based eviction method (H2O, SnapKV, PyramidKV, R-KV, Ada-KV) shares one
   paradigm: estimate how much each cached token will matter, keep the top ones.
   The 2026 result that matters here is uncomfortable and reproducible: on
   reasoning workloads, replacing the score with *uniform random selection* while
   keeping the prompt and the recent window matches the strongest tuned baseline
   across four models and six reasoning benchmarks, and is 32-43% faster in vLLM
   because it computes no scores at all (Random Attention, arXiv:2609.03430).
   The corollary: what those methods protect -- prompt tokens and recent tokens --
   is what carries the quality, not the scoring heuristic.

2. **Block/page selection won because it is kernel-friendly and query-aware.**
   Quest (page min/max bounds), MInference (A-shape / vertical-slash /
   block-sparse heads), XAttention (strided antidiagonal sums), FlexPrefill
   (cumulative-attention budgets) and MoBA (top-k block routing) all keep the KV
   cache intact and choose which *blocks* to read. They are the family that
   survives contact with production kernels, because selection can be expressed as
   a block mask and executed by one generic block-sparse kernel.

3. **The production design has converged on a learned indexer sitting in front of
   a sparse core.** DeepSeek's DSA (*lightning indexer*: `I(t,s) = Σ_j w_{t,j} ·
   ReLU(q_{t,j}·k_s)`, then top-k tokens, then MLA over the selected set) is now
   the reference design, and 2026's work is almost entirely about *the indexer's
   own cost*: cross-layer reuse (IndexCache), hierarchical/streaming indexing
   (LongCat LSA), memory-bounded top-k (StreamIndex), predictive indexing (SPIN),
   token-level vs compressed indexing (CSA2 in DeepSeek-V4.1-Flash). The
   bottleneck moved from attention to selection.

4. **Compressed state is no longer a quality compromise -- it is a hybrid
   architecture with a 3:1 or 4:1 pattern.** Kimi Linear's KDA (channel-wise decay
   + delta rule, chunkwise via a specialised diagonal-plus-low-rank transition)
   claims, with a fair matched-token comparison, to *outperform* full MLA while
   cutting KV cache by up to 75% and hitting 6x higher decode throughput at 1M
   context; 2026's Gated DeltaNet-2 and Kimi K3 (2.8T MoE, 1M context) build
   directly on it. The retrieval weakness of finite state is handled by keeping a
   few full-attention layers, not by fixing the state.

**Five claims that the 2026 evidence does *not* support:**

5. *"Sparse attention is training-free plug-and-play at frontier quality."*
   Training-free selection is genuinely good on retrieval-flavoured long-context
   benchmarks (RULER, LongBench, needle-in-a-haystack) and genuinely fragile on
   tasks that need dispersed evidence: multi-hop QA, codebase-wide understanding,
   and long chains of thought. The 2025-2026 pattern is consistent -- methods that
   only sparsify *inference* on a dense-trained model lose the most exactly where
   the reasoning tokens matter, and methods trained natively (NSA, DSA, KDA)
   close or reverse the gap.

6. *"Block-sparse selection is a solved approximation problem."* The COBS analysis
   (arXiv:2607.09052) reduces block selection to ranking blocks by *attention
   mass* and shows that if selection were exact, block-sparse would match dense.
   It is not exact, and the proxies differ by a lot: pooled-key scores (NSA),
   channel bounds (Quest), antidiagonal sums (XAttention), mean-pooled inner
   products (MoBA). Which proxy is best is still an open, empirically contested
   question.

7. *"Cheaper attention automatically means cheaper serving."* Indexer overhead is
   now the dominant term at long context: the naive CSA pipeline materialises a
   `[B, S, H_I, T]` FP32 score tensor that reaches 256 GB at 65k tokens with
   V4-Flash indexer dimensions, and OOMs on an H200 (StreamIndex,
   arXiv:2605.02568). Also, on agentic workloads the KV-cache *reuse* policy
   (prefix caching, TTLs, LRU vs. its critics) frequently dominates any attention
   kernel win -- a point argued at length, and largely unrefuted, in the "LRU is
   harder to beat than the KV-cache papers suggest" analysis.

8. *"Long context is the same problem as long-horizon agents."* The workloads that
   drive 2026 deployments are *input-heavy and tool-mediated*: DeepSeek's own
   framing of V4.1-Flash is that "long-horizon agents have made model workloads
   increasingly input-heavy", which is why prefill cost and KV *storage* (890
   bytes/token, FP4 KV, cross-layer reuse) get as much design attention as decode
   speed.

9. *"Closed frontier models publish enough to tell."* OpenAI, Anthropic and Google
   publish neither attention architecture nor cache footprints. The observable
   proxies (cache pricing per token, explicit context caching, prompt-cache
   discounts, per-token cache-read pricing 10x below prefill) are consistent with
   dense attention + KV reuse + quantisation rather than with natively sparse
   attention, but this is inference from pricing, not documentation.
   See [frontier-gap.md](frontier-gap.md).

---

## 2. Method and sources

This survey is assembled from four kinds of evidence, and it says which is which
everywhere:

| source | what it is good for | examples used here |
| --- | --- | --- |
| peer-reviewed / technical-report papers | mechanisms, ablations, benchmark tables | NSA (ACL'25), Quest (ICML'24), MInference (NeurIPS'24), XAttention (ICML'25), MoBA, FlexPrefill, Kimi Linear, Gated DeltaNet-2 |
| frontier model reports and model cards | what actually ships, with what numbers | DeepSeek-V3.2 / V3.2-Exp, DeepSeek-V4.1-Flash, Kimi K3, MiniMax-01 |
| systems papers that audit the deployable pipeline | overhead accounting, OOM walls, serving behaviour | StreamIndex, LongCat LSA, IndexCache, SPIN, SparseEngine, parallel-attention offload papers |
| public discussion (Hacker News threads, GitHub issues, release notes, blog posts) | what practitioners report that papers omit: kernel maturity, benchmark-vs-vibe gap, cache economics | the DeepSeek-v3.2 thread (982 points), Kimi Linear (306), the DeltaNet-family walkthrough (297), the agentic-KV-cache/LRU study (108), FlashAttention-4 reverse engineering |

Where a claim rests on a single source, it is marked *(single source)*. Where the
community disagrees with a paper's framing, both are given.

**An honest caveat about benchmarks.** Almost every sparse-attention paper reports
RULER, LongBench and needle-in-a-haystack. These are *retrieval-shaped*: a small
number of high-mass keys determine the answer, which is exactly the regime where
top-k selection is nearly optimal. Coding agents, long chains of thought and deep
research stress a different regime -- many moderately-attended tokens whose
*combination* matters -- and that regime is under-measured. This is the single
most important methodological point in this survey, and it is why section 5 spends
as much space on what is *not* measured as on what is.

---

## 3. Taxonomy

Four mechanisms, one per row. Every variant implemented in this repository is a
point in this table.

| family | what is stored | what is read per query | training needed? | representative methods |
| --- | --- | --- | --- | --- |
| **A. structural / local** | full KV, windowed reads | `window + sinks` | no | StreamingLLM, sliding-window layers, sink tokens |
| **B. training-free selection** | full KV (+ small metadata) | top-k pages/blocks/tokens by a proxy score | no | H2O, SnapKV, PyramidKV, Quest, MInference, FlexPrefill, XAttention, MoBA |
| **C. learned / native sparse** | full KV, learned selector | top-k by a learned indexer or gate | yes (or a fine-tune) | NSA, DSA (DeepSeek V3.2), CSA/CSA2 (V4.x), SeerAttention-R, SAS |
| **D. compressed state** | fixed-size state (or latent) | everything, but through a state of size `d_k x d_v` | yes | MLA, Mamba-2/SSD, Gated DeltaNet, KDA/Kimi Linear, hybrids (3:1) |

Two cross-cutting axes matter as much as the mechanism:

* **Query-aware vs. query-agnostic.** Quest/MInference/XAttention/FlexPrefill/MoBA/DSA
  look at the current query before deciding; eviction methods (H2O/SnapKV) decide
  once and hope the decision survives. The 2026 literature is unambiguous that
  query-awareness buys quality per unit budget; it also costs a scoring pass, and
  that scoring pass is now the bottleneck.
* **Compression along the sequence axis vs. the feature axis.** Eviction and
  selection compress the sequence axis (drop or skip tokens); MLA and the
  linear-attention family compress the feature axis (store fewer numbers per
  token). They compose: V4.1-Flash is *both* (CSA2 selection + FP4 latent + 890
  bytes/token).

### 3.1 The cost model that explains the whole field

Decode is memory bound. For one query, one head, head dim `d`, reading `n` KV
entries costs `2·n·d` numbers and buys `4·n·d` FLOPs -- arithmetic intensity ~2,
far below the ~100-200 FLOP/byte ridge point of modern accelerators. Therefore:

* **Fewer bytes read per query is the only thing that speeds up decode.** This is
  why every method in families B and C reports "tokens activated per query"
  (NSA's own accounting target is ~2560 tokens at 32k), and why the repository's
  `read_budget()` metric is measured in KV entries per query rather than FLOPs.
* **Prefill is different**: it is compute bound at short context and increasingly
  bandwidth bound as context grows, which is why prefill-oriented methods
  (MInference, FlexPrefill, XAttention) optimise *compute* sparsity, while decode
  methods (Quest, DSA, MoBA, SeerAttention-R) optimise *read* sparsity.
* **A fixed-size state changes the asymptotics, not just the constant.** KDA/GDN
  decode cost is `O(d_k·d_v)` per token per head *regardless of context*, which is
  why the hybrid family wins at 1M tokens on throughput even when its per-token
  quality is slightly worse.

`docs/kernels.md` shows these three regimes measured, including a CPU
demonstration (the same numbers appear at a smaller scale on a laptop, which is
why the C kernel in this repo is worth running).

---

## 4. The mechanisms in detail

### 4.1 Family A -- structural sparsity

The cheapest and most deployed: every layer reads the last `W` tokens plus a few
leading "sink" tokens. It is the local branch inside NSA, the "streaming head"
behaviour of DuoAttention, and the reason MoBA can be described as a
generalisation of both sliding-window and sink attention. Its failure mode is
well understood: anything that must be retrieved from far away is simply gone.
What is *not* widely appreciated is how much of a trained model's long-context
behaviour this covers: MInference's head analysis finds many heads are A-shaped
(initial tokens + local window) and stable, while a minority do the retrieval.
Implementation: [`variants/local.py`](../src/linattn/variants/local.py).

### 4.2 Family B -- training-free selection

**Eviction (H2O, SnapKV, PyramidKV, R-KV, Ada-KV).** Score tokens, drop the rest,
permanently. They reduce *memory first* and speed second, and they pay an
irreversible cost: once evicted, a token can never be recovered, which is
catastrophic for exactly the workloads where a later query needs an earlier
detail. The 2026 critique (Random Attention) also undercuts the scoring signal
itself. Our implementation includes `random_eviction` as a *control* so that any
downstream comparison can check whether a scored selector actually beats the
random baseline: [`variants/eviction.py`](../src/linattn/variants/eviction.py).

**Quest.** Page granularity (16 tokens in the paper), query-aware, with a bound
rather than an estimate: `score(q, page) = Σ_d max(q_d·m_d, q_d·M_d)` where `m, M`
are the channel-wise min/max of the page's keys. Cheap metadata (`2·d` numbers per
page), no training, and a clean correctness story (it is an upper bound on the
page's best logit). Two caveats the paper itself flags and this repo reproduces:
early layers are kept dense because their sparsity is low, and under GQA the
per-head page selections must be unioned, inflating the effective budget by up to
the group ratio -- the exact problem Page-EntroKV formalises for eviction.

**MInference 1.0.** The most "structural" of the selectors: per-head pattern
search over A-shape, vertical-slash and block-sparse, with online estimation of
the important vertical lines (columns) and slash lines (diagonals) from the last
64 query rows, and mean-pooled block scores for block-sparse heads. Its published
claims are strong and honest about their setting: up to 10x prefill speedup at 1M
tokens on an A100 (30 minutes to 3 minutes) with accuracy maintained, and the
observation that top-4k columns recall 96.8% of attention mass at 128k.

**XAttention.** Scores each block by summing only the *strided antidiagonals* of
its attention submatrix (stride 4 in the paper, i.e. 1/4 of the entries), then
softmaxes the block scores and keeps the minimal set of blocks whose cumulative
probability exceeds a threshold (0.9-0.95, per-head thresholds found offline by
dynamic programming). The insight is genuinely useful: a vertical line, a slash
line and a dense blob all intersect antidiagonals, so the antidiagonal sum is a
better proxy per unit compute than pooling. It reports up to 13.5x attention
speedup with RULER/LongBench parity, plus video (VideoMME) and video-generation
(VBench) results -- the latter is a reminder that the strongest evidence for
block-sparse methods currently comes from vision, where attention maps are
empirically more block-structured.

**FlexPrefill.** Two stages: estimate the pattern from a representative query
block, decide between a "structured" (one global estimate) and "query-aware"
(per-block estimate) regime using the Jensen-Shannon distance between the
estimated and implied distributions, then select indices by *cumulative attention*
until a threshold (0.9 query-aware, 0.95 structured) is reached. Its selling
point over MInference is that the budget adapts to the input rather than being a
fixed top-k.

**MoBA.** Top-k routing over KV blocks with the gate `q · mean(K_block)`, always
routing to the current block with a causal mask. Its conceptual value is the
"less structure" argument: sliding window and attention sinks are special cases
of the gate, so a learned gate can express them and more. Our implementation uses
the released inference-time semantics (mean-pooled gate, top-k), which is what
most deployments of MoBA actually use.

### 4.3 Family C -- native / learned sparse

**NSA** (ACL 2025) is the reference architecture: three parallel branches
(compressed, selected, sliding window) combined by learned gates, with selection
*reusing the compression branch's scores* so that the selection itself is block
aligned and cheap. The reported evidence is worth quoting in full because it is
the strongest quality-side evidence in the field: LongBench average 0.469 vs
0.437 for full attention and 0.423 for exact top-k at equal budget, with
double-digit relative gains on multi-hop QA (HPQ +0.087, 2Wiki +0.051), code
understanding (LCC +0.069) and passage retrieval (PassR-en +0.075); and on AIME
after R1-distillation, NSA-R scores 0.121 / 0.146 (8k / 16k generation limit) vs
0.046 / 0.092 for the matched full-attention model. Kernel-wise: up to 9.0x
forward and 6.0x backward speedup at 64k with its Triton implementation.

**DSA / CSA (DeepSeek V3.2 - V4.1).** Replace NSA's structural branches with a
learned *lightning indexer* and top-k token selection (2048 tokens in the V3.2
release), then run MLA over the selected tokens. The 2026 evolution is entirely
about indexer cost and KV footprint: V4.1-Flash combines CSA2 (cross-layer KV
reuse + compressed indexing) with FP4 KV caching to reach "890 bytes per token of
global KV cache, roughly 1/4 of DeepSeek-V4-Flash", with a causal
encoder-decoder that activates 16B parameters during decode and only 8B during
prefill.

**SeerAttention-R.** Makes the block gate *learned* (pool q/k inside blocks, then
a small projection and sigmoid) and adapts it for long reasoning decodes. It is
the clearest example that the learned-selector family needs a training stage: a
randomly initialised gate is meaningless, which our `fit_gate` demonstrates by
showing both that the parameterisation *can* fit the true block mass and that it
does not do so out of the box.

**SAS** (arXiv:2609.13141) names the fundamental problem with learned selectors:
distilling dense attention distributions into a top-k selector aligns the ranking
with *attention weights* rather than with the *effect on the prediction under a
fixed budget*. It proposes end-to-end optimisation of the ranking instead. If this
line works, it implies many published selector-training recipes are mis-specified.

### 4.4 Family D -- compressed state

**MLA** is the compressed-feature baseline: cache one latent per token, expand to
per-head K/V inside the kernel. It is not sparsity, but it is the substrate on
which DSA operates and the baseline against which Kimi Linear is measured.

**Mamba-2 / SSD** introduced the chunkwise semiseparable matrix form that makes
linear attention trainable at scale (intra-chunk quadratic, inter-chunk via a
state).

**Gated DeltaNet** adds a delta-rule erase (`S ← α(I - βkkᵀ)S + βkvᵀ`) and scans it
chunkwise with a WY/UT transform. **KDA** makes the decay channel-wise and
parameterises the transition as a specialised diagonal-plus-low-rank matrix,
which is what makes a bespoke chunkwise kernel possible; Kimi Linear reports that
it *beats* full MLA at 1.4T training tokens while cutting the KV cache by 75%.

**Gated DeltaNet-2** (arXiv:2605.22791) identifies the remaining structural
limitation: the scalar gate ties together "how much old content to erase on the
key side" and "how much new content to commit on the value side". Splitting them
into a channel-wise erase gate and a channel-wise write gate recovers KDA as a
special case and simplifies to GDN when the decay collapses. Our chunked
implementation makes the same asymmetry explicit (`a_i` erase-left, `b_i`
erase-right, `c_i` write), which is why the derivation in
`variants/recurrent.py` is worth reading even if you only care about kernels.

**Hybrids are the deployment answer.** MiniMax-01 (lightning attention), Qwen3-Next
(Gated DeltaNet + gated attention), Kimi Linear (3:1 KDA:MLA) and Kimi K3
(KDA + attention residuals) all keep a minority of full-attention layers. The
reason is measurable and is reproduced in this repo's MQAR experiment: a
fixed-size state cannot do exact associative recall, and the recall gap grows with
the number of key-value pairs *faster* than it shrinks with state size, so a
small number of exact-attention layers is the cheapest way to buy back retrieval.

---

## 5. What the evidence says about the workloads that matter

Full detail and per-benchmark tables: [evaluation-evidence.md](evaluation-evidence.md).

### 5.1 Coding

Coding is the workload where the *combination* of many moderately-attended tokens
matters most, and it is the workload with the least sparse-attention evidence.
The honest summary:

* **Prefill-heavy codebase understanding** is where block-sparse methods do best;
  NSA reports +0.069 on LCC at equal budget, and MInference/FlexPrefill report
  RULER/InfiniteBench parity at 10x prefill speedup. Repository-scale prompts are
  long, and much of the content is genuinely irrelevant.
* **Agentic SWE tasks are retrieval-shaped but with a long tail.** The relevant
  evidence is not attention papers but serving papers: SWE-Pruner-style *context*
  pruning, HySparse2's two-level KV sharing for "short actions + long tool
  observations", and the observation that agentic sessions alternate inference and
  tool use, which changes which parts of the context are hot.
* **The benchmark-vs-practice gap is real.** In the DeepSeek-V3.2 discussion the
  most-upvoted technical concern was not quality but throughput: "even if you have
  4x RTX 5090, can it run with a reasonable context window at more than crawling
  speed?" -- i.e. for local practitioners the binding constraint is KV *footprint*
  (hence FP4 KV, 890 bytes/token, cross-layer reuse), not attention FLOPs.

### 5.2 Long-horizon agents

This is the workload that changed the field's objective function. Long-horizon
sessions are input-heavy, re-read the same prefix many times, and interleave tool
outputs; the costs are prefill, KV storage, and cache *reuse*, in that order.
Consequences visible in the 2026 literature:

* DeepSeek's V4.1-Flash report is explicitly motivated by agentic input
  heaviness, and its headline number is a *storage* number (890 bytes/token),
  with a prefill-side parameter reduction (8B active during prefill vs 16B during
  decode).
* HySparse2, SparseEngine and the SSD-offload serving papers treat "heterogeneous
  cache representations" as the central systems problem -- mixing evicted KV,
  compressed states and full attention in one serving loop.
* Reverse-prefill/parallel-attention work (Star Attention-style) and prefix-cache
  policy work target the same cost, which is why a survey of sparse attention
  that ignores *cache reuse* would be misleading: a 2x attention kernel win is
  smaller than the difference between a good and a bad prefix-cache policy.

### 5.3 Reasoning and test-time scaling

Reasoning is the regime where the KV cache grows *because of the model's own
output*, and where the naive intuition "long CoT means mostly-irrelevant tokens"
collapses:

* Random Attention's result (selection signal ≈ worthless; keep prompt+recent)
  was measured on six reasoning tasks across four models -- the strongest
  available evidence that score-based eviction is not what carries reasoning.
* SeerAttention-R and LISA are efforts to make *decode-phase* sparsity work for
  reasoning specifically, i.e. they take the opposite view: with a properly
  adapted selector, sparse decode is viable for long reasoning.
* The training-time evidence cuts the other way and is stronger: NSA-R beats its
  matched dense model on AIME at both 8k and 16k generation limits. Native sparse
  training does not merely preserve reasoning -- it can improve it.
* Efficiency caveat: the same DeepSeek release discussion notes that
  Speciale-style deep-reasoning variants emit 50% more tokens (3.5x Gemini 3 on
  one Codeforces-style comparison), so token-count gains can eat efficiency gains.

### 5.4 Deep research / multi-hop

Multi-hop retrieval is the canonical failure mode of top-k selection, because the
evidence for an answer is spread across several moderately-attended passages and
no single key has dominant mass. The one large-scale quality win reported here is
NSA's multi-hop QA results (HPQ +0.087, 2Wiki +0.051 over full attention at equal
budget), achieved with native training and a compressed branch that provides
coarse global coverage. Training-free selectors with a hard top-k budget do worse
on this shape of task by construction, and the field's mitigation is exactly what
NSA does -- keep a coarse "everything, but low-resolution" branch in addition to
the fine-grained selection.

### 5.5 The measurement gap, stated plainly

Three specific holes in the public evidence:

1. **No standard agentic-sparsity benchmark.** RULER and LongBench measure
   retrieval; SWE-bench measures end-to-end outcome with enormous variance; there
   is no accepted protocol that isolates "how much attention sparsity can an agent
   tolerate before task success degrades".
2. **Selectors are evaluated against full attention on the *same* weights.** That
   is the right control for training-free methods, but it systematically
   under-reports native sparse methods, which are trained with the pattern in the
   loop (NSA's own framing: "training-aware algorithm design").
3. **Serving-level numbers are rarely end-to-end.** A 13.5x attention speedup
   (XAttention) is not a 13.5x end-to-end speedup; at long context the indexer,
   the cache layout, the top-k reduction and the prefill of the tool output often
   dominate. The papers that report end-to-end numbers (SPIN: +14.9% output
   throughput, -13.2% median inter-token latency at 30-40% sparsity; LongCat LSA
   scaling 69B-A3B to 560B-A27B) report much more modest figures, and those are
   the numbers to plan with.

---

## 6. Hardware-software co-design: where the wins actually come from

Detailed treatment in [kernels.md](kernels.md); summary:

* **Selection is now the bottleneck, not attention.** Indexer scoring is `O(T²)`
  even when attention is `O(Tk)`. Every 2026 systems paper in this survey attacks
  that: reuse indices across layers (IndexCache), stream the top-k without
  materialising scores (StreamIndex), predict instead of score (SPIN), index
  hierarchically (LongCat LSA), or compress before indexing (CSA2).
* **Block alignment is a hard constraint, not a preference.** Contiguous block
  reads are what let sparse attention reuse Tensor Cores and coalesced HBM
  access; NSA's design notes say this explicitly, and LongCat LSA's
  "streaming-aware indexing" exists purely to convert scattered KV entries into
  hardware-aligned contiguous layouts.
* **Precision is becoming part of the algorithm.** FP4 KV in V4.1-Flash; FP4
  tensor-core attention kernels (MpFA's QK4V8, hardware-aware FP4 FlashAttention
  4); low-bit recurrent-state quantisation for GDN/KDA (LeapQuant). A cautionary
  result from the same period: bf16 FlashAttention-3 gradients can blow up late
  in training (arXiv:2609.34272) -- kernel-level numerics are now a model-quality
  concern, not just a systems concern.
* **The compiler layer is real.** FlexAttention's block-mask interface is how most
  teams should ship a selector (choose a mask, let the compiler generate the
  kernel); TileLang/TLX-style DSLs appear in the Blackwell attention-kernel
  literature; PIT is what MInference used for its dynamic sparse indexes. This
  repository implements the flex-mask path (``kernels/flex_mask.py``) so that the
  "algorithm -> mask -> kernel" pipeline is exercised rather than described.
* **Offload and heterogeneity are first-class.** SSDs holding sparse KV, CPU-GPU
  load balancing of the cache, processing-near-memory decode, and NPU/vector-CPU
  attention kernels all appear in 2026. The unifying observation: once attention
  reads only a few percent of the cache, *moving* the cache stops being the
  bottleneck and *finding* the right entries becomes it.

---

## 7. The gap between this research and frontier-lab practice

Full analysis: [frontier-gap.md](frontier-gap.md). The short version:

| lab / model | attention design | evidence |
| --- | --- | --- |
| DeepSeek V3.2 / V3.2-Exp | MLA + **DSA**: lightning indexer, top-k=2048 tokens | technical report, kernel in the release |
| DeepSeek V4 / V4.1-Flash | **CSA2** (compressed + sparse + cross-layer KV reuse) + FP4 KV, 890 B/token, causal encoder-decoder | technical report (arXiv:2609.19969) |
| Kimi Linear / K3 | **KDA** (channel-wise decay + delta rule, DPLR chunkwise) in a 3:1 hybrid with full attention; K3 adds attention residuals at 2.8T | arXiv:2510.26692, 2607.24653 |
| MiniMax-01 | lightning (linear) attention at scale | arXiv:2501.08313 |
| Qwen3-Next | Gated DeltaNet + gated attention hybrid | model card |
| GLM / Yi / Llama-4 lineage | mostly dense (+ RoPE scaling, iRoPE-style chunking) | model cards |
| OpenAI / Anthropic / Google | undisclosed; pricing consistent with dense + cache reuse + quantisation | inference, not documentation |

Two structural conclusions:

1. **The gap is training integration, not cleverness.** Every architecture that
   reports *quality* parity or better (NSA, DSA, KDA) trains the sparsity into the
   model. Every method that only sparsifies inference has to fight the model's
   learned dense attention distribution and loses the most on the hardest tasks.
   Labs that can afford ~100B-token ablations ship native sparsity; labs that
   cannot, ship kernels, quantisation and cache reuse on dense attention.
2. **The Chinese open-weight labs are the ones publishing mechanisms; the US
   frontier labs are the ones publishing pricing.** For a practitioner, the
   actionable asymmetry is that the strongest *published* recipes for long-context
   efficiency are open-weight (DeepSeek's DSA/CSA, Moonshot's KDA), and that
   closed models' cache-read pricing (typically ~10x cheaper than a fresh prefill
   token) is the strongest public signal about their internal cache strategy --
   it tells you they invest in *reuse*, which any architecture can exploit.

---

## 8. Open problems (ranked by how much they would change practice)

1. **End-to-end budget accounting for selectors.** The field reports attention
   speedups; serving needs total-cost curves including the indexer, the top-k, the
   layout transformation and the cache policy. StreamIndex-style accounting should
   be the norm, not the exception.
2. **Principled selection under a budget for multi-hop evidence.** COBS reduces
   selection to attention-mass ranking; nobody has a cheap estimator that is good
   at *dispersed* mass. This is the crux for deep research and codebase-wide
   reasoning.
3. **Training recipes for learned selectors.** SAS shows the standard
   distill-the-attention-distribution objective is misaligned. Meanwhile DSA/NSA
   work, but are expensive to iterate on. A cheap, correct objective would unlock
   the family for teams that cannot retrain a frontier model.
4. **Numerics of compressed state.** Chunked delta-rule algorithms are
   ill-conditioned when the decay range inside a chunk is large (the `exp(gc_i -
   gc_j)` factors); production kernels bound the decay and accumulate in FP32.
   The interaction of decay bounds, quantised state (LeapQuant) and long-context
   quality is barely explored, and it may be the real limit on state compression.
5. **Cache policy as an attention problem.** The agentic-KV-cache analysis argues
   the mainstream eviction literature mis-specifies its own objective when the
   deployment has TTLs and prefix reuse. Formalising sparse attention and cache
   admission/eviction jointly looks like the highest-leverage unclaimed ground.
6. **Hybrid layer placement.** Everyone uses 3:1 or 4:1 with the attention layers
   uniformly spread. Whether placement should follow head roles (retrieval heads
   in attention layers), depth (early layers dense, per Quest's observation) or
   the workload is open, and cheap to study at small scale -- which is exactly
   what `experiments/hybrid_recall.py` does here.

---

## 9. Using this repository

| what you want | where to look |
| --- | --- |
| the mechanisms, in code, with citations | `src/linattn/variants/*.py` (each class carries its own citation) |
| one page per variant: mechanism, usage, cost, caveats | `docs/variants/*.md` |
| how selection is expressed for a compiler-generated kernel | `src/linattn/kernels/flex_mask.py` |
| hand-written kernels and their fallbacks | `src/linattn/kernels/triton/`, `src/linattn/kernels/c/` |
| the validation story (what is checked, and how strictly) | `tests/`, and section 3 of `README.md` |
| measured numbers produced on this machine | `reports/` |

The repository deliberately separates **semantics** (a variant's selection rule,
implemented with explicit masks so it can be audited) from **execution** (the
kernel that reads the selected entries). Every variant is validated twice: against
dense attention at an unrestricted budget, and against its own chunked/streaming
implementation for cache equivalence. If a variant cannot pass those two, its
benchmark number is not interpretable.

---

## 10. References

See [`citations.bib`](citations.bib) for the machine-readable list (with URLs) and
[`research/corpus.jsonl`](research/corpus.jsonl) for the raw survey corpus that
this document was built from, including the discussion threads.

Primary sources cited above, in order of appearance: Random Attention
(arXiv:2609.03430); Quest (arXiv:2406.10774); MInference 1.0 (arXiv:2407.02490);
XAttention (arXiv:2503.16428); FlexPrefill (arXiv:2502.20766); MoBA
(arXiv:2502.13189); NSA (arXiv:2502.11089); COBS (arXiv:2607.09052); StreamIndex
(arXiv:2605.02568); LongCat Sparse Attention (arXiv:2608.01662); IndexCache
(arXiv:2603.12201); SPIN (arXiv:2610.09025); DeepSeek-V4.1-Flash
(arXiv:2609.19969); Kimi Linear (arXiv:2510.26692); Kimi K3 (arXiv:2607.24653);
Gated DeltaNet-2 (arXiv:2605.22791); Gated DeltaNet (arXiv:2412.06464); Mamba-2
(arXiv:2405.21060); MLA/DeepSeek-V2 (arXiv:2405.04434); SeerAttention-R
(arXiv:2506.08889); SAS (arXiv:2609.13141); MiniMax-01 (arXiv:2501.08313);
H2O (arXiv:2306.14048); SnapKV (arXiv:2404.14469); PyramidKV (arXiv:2406.02069);
StreamingLLM (arXiv:2309.17453); DuoAttention (arXiv:2410.10819); LServe
(arXiv:2502.14866); Star Attention (arXiv:2411.17116); ShadowKV (arXiv:2410.21465);
FAQ/Page-EntroKV (arXiv:2610.03135); LeapQuant (arXiv:2609.38166); MpFA
(arXiv:2609.33135); Hardware-Aware FP4 FlashAttention-4 (arXiv:2609.04105);
BF16 attention gradients (arXiv:2609.34272); HISA (arXiv:2603.28458); PIVOT
(arXiv:2607.24593); LiteTopK (arXiv:2607.11976); LISA (arXiv:2607.19358);
FreeSolo (arXiv:2610.17191); NAMOH (arXiv:2609.38832); HySparse2
(arXiv:2609.26368); SparseEngine (arXiv:2609.39068); DeltaS (arXiv:2609.27470);
GateSight (arXiv:2610.14754); HLA (arXiv:2610.05842); Optimizing NSA
(arXiv:2511.00819).
