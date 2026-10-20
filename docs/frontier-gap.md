# The gap between published attention research and frontier-lab practice

The question this document answers: *is the sparse/compressed attention that the
literature reports what frontier labs actually run?*

The answer has three parts: (1) for the open-weight frontier labs, the gap has
essentially closed -- they *are* the papers; (2) for the closed labs, nobody
outside knows, and the observable proxies suggest a different (more
conservative) strategy; (3) for everyone else, the binding constraint is
training integration, not algorithm availability.

---

## 1. What ships, as of October 2026

| model / lab | attention design | compressed state? | sparse selection? | evidence |
| --- | --- | --- | --- | --- |
| **DeepSeek V3.2 / V3.2-Exp** | MLA core + **DSA** lightning indexer, top-k = 2048 tokens per query | yes (MLA latent KV) | yes, learned indexer | technical report released with the weights; the *exp* variant (Sept 2025) shipped the mechanism first |
| **DeepSeek V4 / V4.1-Flash** | **CSA2**: compressed + sparse attention with **cross-layer KV reuse**, FP4 KV cache, causal encoder-decoder | yes (latent + compressed groups) | yes, compressed-then-indexed | arXiv:2609.19969: 890 bytes/token global KV, ~1/4 of V4-Flash, 1M context, 552B backbone, 16B active on decode / 8B on prefill |
| **Kimi Linear** | **KDA** (channel-wise decay + delta rule, specialised DPLR chunkwise kernel) in a **3:1 hybrid** with MLA | yes (KDA state) | no (linear + full attention, not selection) | arXiv:2510.26692: beats full MLA at 1.4T tokens, KV -75%, up to 6x decode throughput at 1M; open KDA kernel + vLLM integration |
| **Kimi K3** | KDA + **attention residuals**, 2.8T MoE (104B active), 1M context | yes | no | arXiv:2607.24653: ~2.5x scaling efficiency over K2, "frontier-level long-horizon coding"; explicitly credits "algorithm-system co-design for KDA" |
| **MiniMax-01** | lightning (linear) attention at scale | yes | no | arXiv:2501.08313 |
| **Qwen3-Next** | Gated DeltaNet + gated attention hybrid | yes | no | model card / release |
| **Mamba-/hybrid lineage (Mamba-3, HLA, Jamba-style)** | SSM/linear layers, some hybrid | yes | no | HN thread on Mamba-3 (300 points, Mar 2026); HLA (arXiv:2610.05842) |
| **Llama-4 / GLM / Yi lineage** | dense attention (+ RoPE scaling, chunked iRoPE-style patterns) | no | partial (chunked/alternating patterns in Llama-4's iRoPE) | model cards |
| **OpenAI / Anthropic / Google** | undisclosed | undisclosed | undisclosed | see section 3 |

The V4.1-Flash entry proves that CSA2 ships at frontier scale; it does not prove
that the surrogate attention is close to dense softmax layer by layer. See
[csa2-approximation.md](csa2-approximation.md) for the decomposition and the
official model-level evidence.

### The pattern

1. **Everyone at the open-weight frontier compresses state.** Not one of the 2026
   open-weight frontier models is pure dense attention. The choice is between
   latent compression (MLA), recurrent compression (KDA/GDN), or both.
2. **Sparse *selection* ships only where a lab can retrain with the indexer in the
   loop.** That means DeepSeek (DSA/CSA) essentially alone. MoBA comes from
   Moonshot, and *Kimi Linear chose a hybrid rather than selection*. That choice
   is informative: the same lab that invented MoBA shipped KDA.
3. **Compression along the feature axis (MLA) plus a hybrid layer pattern is the
   consensus safe choice.** It is architecture-visible (KV footprint numbers are
   publishable), it does not require a learned selector to be well-calibrated, and
   it composes with quantisation.
4. **Selection is being applied where it is cheap and provably safe:** a local
   window (always), the prompt/sink tokens (always), and compressed groups for
   global coverage (NSA/CSA). The aggressive "top-k tokens and nothing else"
   configuration is what the reasoning literature has been attacking.

## 2. Why the research-to-production gap exists at all

Four concrete reasons, in rough order of how much they matter:

**2.1 Training integration is the real cost.** Every method that reports *quality
parity or better* trains the pattern into the model: NSA (pretrained sparse), DSA
(indexer trained with the model), KDA (delta rule from scratch), MoBA (100B tokens
of continual pretraining with the pattern active). Training-free methods are
*forced* into a harder regime: they must approximate the attention distribution of
a model that was optimised under a different objective, and their errors are
largest where dense attention is most non-local -- which is where reasoning and
multi-hop tasks live. The literature rarely states this as the reason, but the
pattern in the numbers is unmistakable (see
[evaluation-evidence.md](evaluation-evidence.md)).

**2.2 Kernel maturity, not kernel novelty.** MInference, Quest and XAttention are
all implemented on Triton/PIT-derived dynamic-sparse kernels; NSA ships its own
Triton kernels; DSA ships in the V3.2 release. In contrast, a paper with a
reference implementation and no vendor integration (FlexAttention mask support,
vLLM/SGLang backend, or a fused kernel in the release) is, in practice, not
deployable -- which is why this repository implements both the semantics and the
execution paths and prints a backend-availability report
(`linattn.kernels.describe_backends()`).

**2.3 Serving systems re-optimise the problem.** Production stacks fold
attention efficiency into a bigger picture: paged KV, prefix caching, chunked
prefill, quantised cache, speculative decoding, and (increasingly) offload to
CPU/SSD. A method that reduces attention FLOPs by 4x but breaks paged-cache layout
or forces a non-contiguous KV gather can be *slower* end-to-end. LongCat LSA's
"streaming-aware indexing" is precisely an acknowledgment of this: it exists to
turn scattered accesses into hardware-aligned contiguous access.

**2.4 Evaluation mismatch.** RULER/LongBench needles reward top-k selection; the
workloads that pay the bills (agents, codebase reasoning, deep research) reward
combining dispersed evidence. A method tuned on the former can be genuinely worse
on the latter while looking better in papers.

## 3. What can be inferred about the closed frontier models

No published architecture. The observable signals:

* **Prompt-cache pricing.** All three major closed providers price *cache reads*
  roughly an order of magnitude below fresh prefill tokens, and cache *writes*
  slightly above. That is the economics of reuse (prefix caching over a growing
  KV), not of sparse attention. If attention were natively sparse in the DSA
  sense, the KV footprint advantage would show up as cheaper long-context
  *storage*, which is not how their pricing is structured.
* **Context windows and pricing vs. context length.** Linear-ish price scaling in
  context length is consistent with dense attention plus efficient kernels plus
  cache reuse; sublinear scaling would suggest sparsity or compression.
* **No public kernel/library releases** for sparse attention, in contrast to
  DeepSeek/Moonshot/Qwen shipping kernels with weights.

**Conclusion (inference, clearly labelled):** the closed labs most likely run
dense-or-locally-windowed attention with MLA-like or quantised KV, aggressive
prefix caching, and architecture choices (MoE routing, model depth) that are
orthogonal to attention sparsity. If that is right, then the *published* frontier
of attention sparsity is ahead of what the closed frontier runs -- the reverse of
the usual gap direction, and a rare opportunity for open-weight practitioners to
use better techniques than the closed APIs expose.

## 4. Where the gap is closing fast

* **Indexer efficiency.** The 2026 systems papers (IndexCache, SPIN, LongCat LSA,
  StreamIndex, HISA, PIVOT, LiteTopK, MISA) are exactly the work needed to make
  DSA-style selection deployable by teams that are not DeepSeek. They are
  accelerator-agnostic in spirit: reuse, predict, hierarchise, stream.
* **Hybrid recipes.** Kimi Linear's released kernel + vLLM integration means the
  3:1 KDA:MLA pattern is now a drop-in option; Gated DeltaNet-2 (arXiv:2605.22791)
  improves the same primitive with separated erase/write gates.
* **Compressed KV + quantisation.** FP4 KV in V4.1-Flash, KVarN-style native
  KV-quantisation backends, low-bit recurrent state (LeapQuant) -- the memory
  story is advancing faster than the compute story, which matches where agentic
  workloads actually hurt.

## 5. What a practitioner should do

1. **If you cannot retrain:** use dense attention on a minority of layers (or all
   of them if the budget allows), add a generous local window and sinks, and spend
   your effort on KV *quantisation* and *cache reuse*. Use block-sparse selection
   only for prefill over long, low-relevance contexts, where the evidence is
   strongest and the risk is lowest. If you must select during decode, select at
   page granularity with a query-aware bound (Quest) rather than evicting.
2. **If you can fine-tune:** adapt a learned gate/indexer with a bounded budget
   (SeerAttention-R style), verify it recovers attention mass on *your* workload,
   and keep prompt + recent + coarse-global branches unconditional.
3. **If you can pretrain:** the 2026 evidence says train the pattern in (NSA/DSA
   for selection, KDA/GDN for state), keep a coarse branch, and keep a few exact
   layers. This is where the quality wins are real rather than recovered.
