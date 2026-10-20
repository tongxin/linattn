# CSA2: how close is it to dense softmax?

The cost table shows a large reduction in KV footprint and prefill FLOPs for
CSA2. Those numbers are an algorithmic budget, not a quality result. This page
separates the three sources of approximation in the repository's CSA2 reference:

1. **selection** of compressed entries;
2. **pooling** several token KV pairs into one entry;
3. **quantisation** of the stored entries.

The repository result is a controlled synthetic decomposition. The official
DeepSeek-V4.1-Flash report provides model-level benchmark evidence, but does not
publish a dense-versus-CSA2 layerwise attention-error ablation. Those two levels
of evidence must not be conflated.

The disclosed layer schedule, dimensions and precision formats are compared with
the repository's surrogate in [csa2-official-config.md](csa2-official-config.md).

## 1. The exact and approximate computations

For one query and one head, dense causal attention is

```text
p_j = exp(q · k_j / sqrt(d)) / Z
O   = Σ_j p_j v_j
```

The CSA2 reference keeps the recent SWA window `W` exactly and replaces older
tokens with pooled groups. For a group `g` of `m` tokens:

```text
μ_k,g = (1/m) Σ_{j∈g} k_j
μ_v,g = (1/m) Σ_{j∈g} v_j
```

It then runs one softmax over:

```text
W  ∪  {selected compressed group entries}
```

The implementation does **not** use the exact group-sum numerator
`Σ_{j∈g} exp(q·k_j/√d) v_j`; it treats the group as one pooled entry with
`μ_k,g` and `μ_v,g`. This is the main modelling approximation, before any
top-k or FP4 error is considered.

The order of approximation matters:

```text
dense tokens
  → choose a support S of raw-window tokens + compressed groups
  → replace each group by (μ_k, μ_v)
  → optionally quantise the stored K/V
  → normalise softmax over the resulting entries
```

With a raw window covering the whole sequence, CSA2 is exactly dense attention.
With `m=1`, every group is a token; with every group selected, it is also dense
attention. Both cases are tested.

## 2. Theoretical error decomposition

There is no useful global approximation guarantee as strong as “CSA2 is within
ε of dense softmax” for arbitrary K/V. The error is data-dependent. A tractable
analysis uses separate bounds for the three stages.

### 2.1 Selection error

Let `δ` be the dense attention mass outside the selected support:

```text
δ = 1 - Σ_{j∈S} p_j
```

If the selected support is renormalised and values are bounded by
`V_max = max_j ||v_j||₂`, then

```text
||O_dense - O_selected||₂
  ≤ 2 δ V_max
```

This explains why `mass_capture = 1 - δ` is a useful diagnostic, but not a
sufficient quality metric: two selectors can capture the same mass and choose
different value directions.

### 2.2 Pooling error

Let `μ_k,g`, `μ_v,g` be the group means and define

```text
ε_k,g = max_{j∈g} ||k_j - μ_k,g||₂
ε_v,g = max_{j∈g} ||v_j - μ_v,g||₂
```

Define the reference group mass `α_g = Σ_{j∈g} p_j` and the pooled-entry mass
`β_g` after the surrogate softmax. A pathwise bound of the form

```text
||O - O_pooled||₂
  ≤ Σ_g |α_g - β_g| ||μ_v,g||₂
    + Σ_g α_g ε_v,g
```

shows the two failure modes: substantial within-group key variation changes the
attention mass allocated to a group, while value variation changes the result
even when the mass is correct. The bound is deliberately conservative and is not
the implementation's exact error; it identifies the missing terms.

### 2.3 Quantisation error

For an FP4 reconstruction error `δk_j`, the logit perturbation is
`δa_j = q·δk_j/√d`. If `ε_a = max_j |δa_j|`, a softmax perturbation can be
bounded by a term proportional to `exp(ε_a) - 1`; a value reconstruction error
`ε_v` contributes directly to the output error. The codec here uses:

- FP4: E2M1 values, per-group scales;
- FP8: e4m3 with per-group scales;
- FP16: storage accounting only, no quantisation.

The official DeepSeek report uses QAT for FP4 main KV, an E4M3 scale per 16
channels, and retains FP8 for the SWA cache because it is more sensitive. Those
details matter: a post-hoc FP4 conversion of an FP16 cache is not the training
configuration described by the report.

### 2.4 Cross-layer sharing error

Reuse modes read another layer's main KV. If layer `l` uses representative `r`,
the additional error is driven by `||K_l-K_r||`, `||V_l-V_r||`, and whether the
selected indices still target the same positions. The repository sweeps a
controlled correlation parameter `σ` in
[`csa2_model.py`](../experiments/csa2_model.py). It observed zero sharing
divergence for identical layers, 0.068 at `σ=0.05`, 0.267 at `σ=0.2`, and
approximately 1.0 for independent layers. That is a model of the architectural
requirement, not a measurement from a released checkpoint.

## 3. Empirical decomposition in this repository

Run:

```bash
make csa2-analysis
# or
python experiments/csa2_approximation.py --seq 512 --batch 1
```

The experiment uses `H=4`, `H_kv=2`, `d=64`, `T=512`, a trained-model-free
indexer, and two input distributions: random K/V and planted retrieval spikes.
The numbers below are from the structured input at `T=512`; the complete JSON
and random-input results are in
[`reports/csa2_approximation.md`](../reports/csa2_approximation.md).

| configuration | relative L2 error | mass capture | bytes/token | dense / CSA2 FLOPs |
| --- | ---: | ---: | ---: | ---: |
| full raw window; no compressed groups | 0.000 | 1.000 | 513 | 1.00× |
| quantisation only, FP16 | 0.000 | 1.000 | 513 | 0.89× |
| quantisation only, FP8 | 0.059 | 1.000 | 273 | 0.89× |
| quantisation only, FP4 | 0.237 | 1.000 | 145 | 0.89× |
| selection only, 128-token window + top-8 tokens | 0.285 | 0.927 | 640 | 2.56× reduction |
| pooling only, `m=16`, window=1 | 1.057 | 0.918 | 33 | 13.8× reduction |
| combined, `m=32`, FP16, window=128, top-8 groups | 0.778 | 0.948 | 144 | 3.7× reduction |
| combined, `m=32`, FP4, window=128, top-8 groups | 0.778 | 0.947 | 132.5 | 3.7× reduction |

Three findings matter:

1. **Mass capture is not enough.** The combined `m=32`, top-8 configuration
   captures 94.8% of dense attention mass but still has 0.778 relative output
   error. The missing mass and the within-group value averaging both contribute.
2. **FP4 is not the dominant error source in this untrained reference.** FP4
   changes error by only about 0.0001 once pooling and selection are active. At
   `m=1`, FP4 alone gives 0.237 relative error, so precision still matters when
   the compressed entries are otherwise faithful.
3. **Pooling is severe on random or heterogeneous groups.** With `m=16` and no
   selection loss, error is already 1.057 on the structured input. The reference
   mean-pooling compressor is not trained to preserve attention outputs. The
   production CSA2 compressor and QAT are trained; this surrogate is not.

The random-input rows are equally important: combined error is 0.536, selection
only at the same window/top-k is 0.470, and pooling only at `m=16` is 1.588.
That is a warning against using a single synthetic distribution as evidence of
quality parity.

## 4. What the official model report establishes

The DeepSeek-V4.1-Flash report claims:

- global KV cache footprint of **890 bytes/token**, about one quarter of
  DeepSeek-V4-Flash;
- persistent KV footprint reduced to about one eighth with SWA Bounded Replay;
- FP4 **main** KV during training with “only marginal performance degradation”;
- FP8 retained for the more sensitive SWA KV;
- CSA2 Full, Reindex and Reuse modes, with cross-layer main-KV/indexer-K sharing
  and a hierarchical sparse indexer;
- a model-level comparison in which V4.1-Flash is broadly stronger than
  V4-Flash, including Codeforces 3471 vs 3289, MathArena Apex 65.6 vs 58.6,
  Terminal-Bench 2.1 90.6 vs 82.7, Terminal-Bench 3.0 30.0 vs 7.6, and DeepSWE
  v1.1 74.2 vs 54.4.

The report also gives counterexamples to a blanket claim: V4.1-Flash is lower
than V4-Flash on HLE in its table (36.8 vs 37.8), and the authors explicitly
warn that “potential selection errors in CSA2 ... may still cause capability
degradation in untested boundary cases.”

Most importantly, these are **whole-model** results. V4.1-Flash also changes the
causal encoder-decoder, parameter count, data recipe, post-training, Engram,
DSpark and deployment stack. The report does not isolate CSA2's dense-softmax
approximation error, does not provide a layerwise dense-vs-sparse attention
ablation, and does not publish a matched dense model with the same parameters
and training data.

## 5. Public discussion: useful signals and limits

The public discussion is dominated by deployment economics rather than
layerwise approximation:

- The September 2026 Hacker News release thread (`49639090`, ~1k points and
  hundreds of comments) focused on the jump from 284B to 552B parameters, whether
  the model is still practical locally, the 890 B/token cache claim, and
  benchmark skepticism. One commenter explicitly separated benchmark scores from
  real-world use; another noted the cache-price implications.
- The October 2026 `1017` thread (`50129045`, also ~1k points) continued the
  discussion around cache economics, deployment cost and benchmark trust.
- A practitioner-oriented architecture write-up was also submitted as
  `49735410`, but it is an explainer, not an independent quality measurement.

The useful conclusion from the discussion is not “CSA2 matches dense attention”.
It is: **the cache economics are large enough that a small quality regression can
still be a deployment win, and the benchmark claims need independent
coding/agent evaluation.** No public thread provides the layerwise dense-versus-
CSA2 output error needed to settle the approximation question.

## 6. What to test with a released checkpoint

The next decisive experiment should use the actual V4.1-Flash checkpoint or an
open CSA2-style model, not this synthetic indexer:

1. Run a dense teacher and the CSA2 layer on identical pre-RoPE K/V and compare
   attention outputs, softmax mass capture, top-k overlap, and downstream hidden
   states.
2. Ablate one factor at a time: full SWA window; all compressed groups; FP16
   versus FP8 versus FP4; Full versus Reindex versus Reuse modes.
3. Evaluate long-horizon coding repairs, multi-hop research, and evidence-sparse
   reasoning separately from RULER-style needle retrieval.
4. Report cache bytes, prefill FLOPs, end-to-end TTFT, and task success together.
   A 3.7× FLOP reduction with a small quality loss is a different result from a
   3.7× reduction with a retrieval collapse.

Until that exists, the defensible repository claim is: **CSA2 is the strongest
published cost-side design in this survey, but this repository has not
established quality parity with dense softmax.**
