# Quantifying how closely efficient attention approximates softmax attention

**Status: a complete first validation cycle on synthetic and matched tiny-model
evidence, not a frontier-LLM result.**

## Abstract

Sparse selection, low-rank projection, random features, quantization, and
compressed recurrent state all make attention cheaper, but they do not
approximate the same object. Benchmark accuracy is therefore an unreliable
standalone measure of approximation quality. This report develops and tests a
three-stage audit:

1. certify the local attention operator against dense softmax at the same q/k/v;
2. measure hidden-state drift after the remaining model layers execute;
3. perform a matched layer intervention to measure predictive-distribution and
   task consequences.

The theory review finds no general subquadratic approximation guarantee for
arbitrary K/V. It does yield useful conditional results: exact decompositions
for sparse support, probabilistic bounds for sampling and streaming methods,
and variance/concentration analyses for kernel approximations. We prove and test
two elementary but load-bearing local bounds: support error depends on both
dropped attention mass and value geometry, and quantized-key error is controlled
by the maximum logit perturbation.

The empirical cycle covers 60 one-layer interventions on each of two task
families across three trained checkpoints. Final hidden-state drift ranks
downstream loss better than local attention error in both tasks
(within-checkpoint mean Spearman `rho=0.974/0.934` versus `0.927/0.898`). Local
error remains useful, but it is not sufficient: interventions with nearly equal
local error can differ by roughly 5-10x in task loss. The first-order task
projection is competitive within a checkpoint but is misleading when loss deltas
from checkpoints with different scales are pooled.

## 1. Research question and estimand

For one query and head, define dense causal attention by

```text
z_j = q · k_j / sqrt(d)
p_j = exp(z_j) / Σ_i exp(z_i)
O   = Σ_j p_j v_j.
```

An efficient method produces `F_hat`. The question is not merely whether
`F_hat(q,k,v)` is close to `F(q,k,v)`. It is:

> Which object does the variant approximate, under what assumptions, and which
> local and propagated errors are sufficient to predict its consequence in a
> trained model and on a stated workload?

The audit keeps three distinctions:

- **different support:** sparse selection evaluates the same softmax on a
  restricted token set;
- **different kernel or precision:** random features, low-rank
  projections, and quantization perturb the attention distribution itself;
- **different state:** recurrent and compressed-state methods replace
  token-addressable history with a fixed representation.

These mechanisms require different theories. A benchmark score mixes all of
them with data, training, and serving effects.

## 2. What the literature establishes

The detailed claim-by-claim map is
[approximation-theory-map.md](approximation-theory-map.md). The main conclusions
are:

### 2.1 No universal guarantee over arbitrary inputs

*Efficiently Approximating Attention Is Hard* (arXiv:2609.37261) rules out a
nontrivial uniform additive or relative guarantee for truly subquadratic
attention under standard complexity assumptions, even with polynomial
preprocessing of K/V. This does not say efficient attention cannot work. It says
that every useful guarantee must condition on input structure, a sampling
model, a restricted operator, or a task distribution.

### 2.2 Sampling and streaming do support conditional certificates

Priority sampling over a MIPS oracle gives explicit retrieved-key requirements
(arXiv:2610.10135). Certified sparse attention uses perturbed retrieval and
confidence bounds per query (arXiv:2610.24991). Streaming coreset methods give
approximation-versus-space bounds (arXiv:2502.07861, arXiv:2606.07205).
*Express* introduces a general conversion from noncausal to causal approximation
with matching guarantees and reports `log^{3/2}(n)/s`-type error under its
Thinformer setting (arXiv:2606.10944).

These are not certificates for arbitrary top-k heuristics. Their assumptions,
estimator, and output object must be instantiated explicitly.

### 2.3 Sparse support has an exact interpretable decomposition

Let `S` be the selected keys, `a=Σ_{j∈S}p_j`, `mu_S` the retained weighted
average value, and `mu_D` the weighted average of omitted values. The selected
output is `O_S=mu_S`, while

```text
O - O_S = (1-a) (mu_D - mu_S).
```

Therefore

```text
||O - O_S||_2 <= (1-a) diam(V) <= 2(1-a) V_max.
```

Dropped attention mass matters, but value-centroid separation and value diameter
matter equally. High mass recall can still fail when the omitted values occupy a
different useful subspace. COBS and CEDAR reach nearby conclusions from block
mass and coarse-key/value dispersion respectively (arXiv:2607.09052,
arXiv:2609.07237); SSA also shows dropped mass controls a sparse-output error
(arXiv:2511.20102).

### 2.4 Attention-distribution error does not identify output error

If `TV(p,p_hat)=tau` and the value diameter is `diam(V)`, then

```text
||O - O_hat|| <= tau * diam(V).
```

The reverse implication is false. A minimal control in this repository gives
the same `TV=0.4` while output error changes from zero, when the two keys have
identical values, to `0.566`, when their values are orthogonal. Distribution
metrics are diagnostics, not substitutes for value-weighted error.

### 2.5 Quantized KV has a local perturbation certificate

For quantized keys, let

```text
eta = max_j |z_hat_j - z_j|.
```

Every posterior likelihood ratio lies between `exp(-2 eta)` and
`exp(2 eta)`, giving

```text
TV(softmax(z), softmax(z_hat)) <= tanh(eta).
```

Value quantization can be separated from this key-induced term:

```text
O(q,k_hat,v_hat)-O(q,k,v)
  = [O(q,k_hat,v)-O(q,k,v)]
  + [O(q,k_hat,v_hat)-O(q,k_hat,v)].
```

The first term is bounded by `2 V_max tanh(eta)`; the second by
`max_j ||v_hat_j-v_j||`, since the attention weights sum to one. This is a local
bound, not a claim about FP4-trained checkpoints or end-to-end quantization.

### 2.6 Kernel, low-rank, and recurrent methods have different guarantees

- Performer/FAVOR+ gives an unbiased positive-feature estimator with variance
  and concentration results; unbiasedness does not imply a small per-query
  error (ICML 2021).
- Linformer relies on an approximate sequence-axis low-rank hypothesis and a
  Johnson-Lindenstrauss-style projection argument (ICML 2020).
- Nyström and polynomial sketches bound conditional matrix/kernel
  reconstructions, not arbitrary sparse-support outcomes.
- Fixed-state recurrences cannot preserve arbitrary token-addressable KV
  information. Recent controlled studies show interference, write/read
  geometry, and training curriculum determine associative recall
  (arXiv:2609.33093, arXiv:2609.16183).

### 2.7 Local error does not compose for free

For a block map `h_{l+1}=B_l(h_l)`, a local intervention with error vector
`r_l=B_hat_l(h_l)-B_l(h_l)` obeys

```text
delta_{l+1} = B_l(h_l+delta_l)-B_l(h_l)+r_l.
```

If `B_l` is `L_l`-Lipschitz, then

```text
||delta_{l+1}|| <= L_l ||delta_l|| + ||r_l||.
```

Unrolling this inequality yields a product-of-Lipschitz bound and can be
extremely loose. The practical task-aligned first-order quantity is
`g^T r_l`, where `g` is the gradient of model loss with respect to the dense
attention output. That quantity includes the downstream Jacobian, but it loses
accuracy for finite perturbations because of curvature and cancellation.

## 3. Audit protocol

### 3.1 Required measurements

| level | measurement | question |
| --- | --- | --- |
| support | retained mass and support intersection | what evidence was omitted? |
| posterior | total variation, JS, Hellinger | how did the probability vector change? |
| operator | normalized value-weighted error and cosine distance | did the change alter the useful output? |
| state | next-layer and final hidden-state drift | did the local error amplify or cancel? |
| prediction/task | logit KL, task loss, accuracy, cost | did model behavior change? |

For an exact support approximation, report the omitted vector
`(1-a)(mu_D-mu_S)` as well as its norm bound. For a kernel estimator, report
feature count, variance/seed dispersion, and operator error; mass recall is not
defined unless the method exposes an explicit support. For quantization, report
key-induced and value-induced error separately.

### 3.2 Required controls

- use the same checkpoint, inputs, positions, and all non-attention weights;
- replace one layer at a time before testing all-layer replacement;
- match selected entries, bytes, and feature dimensions where possible;
- keep dense FP16/BF16 as the reference for precision ablations;
- evaluate both retrieval-shaped and dispersed-evidence inputs;
- keep failed or non-converged seeds in the result.

### 3.3 Falsifiable hypotheses

**H1. Exact local identities.** Sparse omission, attention TV, and quantization
decompositions should match their formulas to numerical precision.

**H2. Metric non-identifiability.** Equal attention-distribution error should
not always imply equal value-weighted error.

**H3. Propagation.** On a matched trained checkpoint, final hidden-state drift
should rank task loss better than local attention error alone.

**H4. Pooling control.** Correlations must be computed within most comparable
units; pooling loss deltas across differently calibrated checkpoints can reverse
the apparent value of a first-order gradient metric.

## 4. Evidence

All experiments run on CPU with PyTorch. Machine-readable outputs are listed at
the end of this report.

### 4.1 Local certificates hold

The selector and quantization checks use three seeds and three input families:
random, retrieval-shaped, and dispersed evidence. Across every tested selector,
the exact sparse decomposition closes within `1.4e-15`, implementation parity is
within `2.7e-15`, and the `2(1-a)V_max` bound is never violated. Its realized
ratio to the bound is `0.131-0.261`, so it is sound but usually loose.

For KV round-trips, all sampled rows satisfy the `tanh(eta)` TV bound. FP16 is
exact; FP8 has smaller attention distribution and output error than FP4:

| input | precision | mean TV | mean output relative error |
| --- | --- | ---: | ---: |
| random | FP8 | 0.0089 | 0.0295 |
| random | FP4 | 0.0375 | 0.1220 |
| dispersed | FP8 | 0.0159 | 0.0366 |
| dispersed | FP4 | 0.0669 | 0.1583 |

This validates the local error decomposition. It does not show that FP4 is
acceptable after quantization-aware training.

### 4.2 Matched trained-checkpoint intervention

The audit trains two-layer causal transformers on two synthetic tasks:

- four-key single-answer retrieval;
- two-evidence dispersed summation.

Three independently initialized checkpoints per task reach held-out accuracies
of `0.979-1.000` for retrieval and `0.994-1.000` for dispersed summation. At each
of two layers, ten interventions are evaluated while all weights and other
layers remain dense and fixed. This gives 60 interventions per task.

The interventions include Quest at one and two blocks, MoBA at one and two
blocks, MInference block-sparse, an exact-mass oracle, sliding window, Performer
with 32 features, and FP8/FP4 KV quantization.

Rank correlations with task-loss change are:

| task | local attention error | final hidden drift | first-order task projection |
| --- | ---: | ---: | ---: |
| retrieval, pooled | 0.914 | **0.966** | 0.582 |
| retrieval, mean within checkpoint | 0.927 ± 0.029 | **0.974 ± 0.006** | 0.853 ± 0.072 |
| dispersed, pooled | 0.869 | 0.923 | 0.723 |
| dispersed, mean within checkpoint | 0.898 ± 0.005 | 0.934 ± 0.014 | **0.960 ± 0.016** |

Final hidden drift has the higher within-checkpoint mean in both task families.
It beats local error in five of six individual checkpoints and ties within
`0.002` on the remaining retrieval checkpoint. This is descriptive for this model
family, not a universal ranking theorem.

### 4.3 Similar local error can hide different task damage

The averages contain counterexamples that rule out treating local error as
sufficient:

| task | intervention | local error | final hidden drift | loss increase |
| --- | --- | ---: | ---: | ---: |
| retrieval | Quest top-2 | 0.142 | 0.134 | 0.149 |
| retrieval | MInference block | 0.173 | 0.437 | 1.432 |
| dispersed | MoBA top-2 | 0.081 | 0.090 | 0.331 |
| dispersed | MInference block | 0.086 | 0.340 | 1.570 |

In the dispersed case, local error differs by 5%; downstream hidden drift and
loss differ by roughly 4-5x. The local metric omits the direction of the error
relative to the checkpoint's later computation.

### 4.4 First-order projection is not a replacement

The first-order projection is an instructive failure of evaluation design. Its
pooled correlations (`0.582/0.723`) look much worse than its within-checkpoint
means (`0.853/0.960`). The cause is that absolute loss deltas have different
scales across checkpoints: pooling ranks mix checkpoint calibration with
intervention effect. We therefore rank interventions within checkpoint before
aggregating.

Within a checkpoint, the first-order projection remains weaker than hidden drift
for retrieval but stronger for dispersed evidence. Curvature and cancellation
still limit it as a finite-error guarantee, but the pooled number alone would
have produced the wrong conclusion.

## 5. Interpretation

The evidence supports a conditional conclusion:

> For an efficient attention approximation, local operator error is necessary
> evidence but not a sufficient predictor of task damage. A defensible audit
> measures the approximand, exact support/precision decomposition, propagated
> hidden-state drift, and task consequences under matched interventions.

The practical decision rules are:

- reject any claim of “lossless sparse attention” that reports only retained
  mass or benchmark accuracy;
- for sparse selection, require dropped mass plus value-geometry diagnostics;
- for kernel/low-rank methods, require feature/rank budgets and seed variance;
- for quantization, ablate key and value errors and include outlier-sensitive
  inputs;
- for deployed systems, compare against a dense matched teacher or explicit
  quality target, and report the task distribution;
- do not use a single local score to rank methods across task families without
  validating that score on the target checkpoint.

## 6. Limitations and open problems

- The trained checkpoints are tiny and synthetic. They establish the protocol
  and falsify stronger claims about metric sufficiency; they do not estimate
  behavior on natural language, code, or frontier models.
- The task labels are single-token outputs. Multi-token reasoning, tool use, and
  long-horizon error accumulation remain untested.
- The experiments compare a fixed set of mechanisms and budgets. They do not
  establish the best sparse method under native training.
- The quantization certificate is deterministic only for the observed local
  logit perturbation. It does not cover training, activation quantization, or
  cache-management errors.
- The literature map distinguishes preprints from foundational/peer-reviewed
  results, but several 2026 entries have not yet received independent replication.
- A natural-language or code-trained small checkpoint, multi-token tasks, and
  native sparse training are required before applying this framework to a
  frontier architecture.

## 7. Conclusion

There is no single scalar “approximation error” for efficient attention. The
defensible quantities are:

1. an exact or probabilistic statement about the attention operator;
2. a value-weighted local error and, where available, a certificate;
3. propagated hidden-state drift under the same checkpoint;
4. a causal task intervention with matched weights and budgets.

The first cycle confirms the exact sparse and quantization identities, shows
that distribution metrics cannot replace value-weighted metrics, and provides
matched trained-model evidence that downstream state drift adds information
beyond local attention error. The next scientific step is not another
approximation leaderboard; it is to rerun this protocol on a small
natural-language or code-trained checkpoint with multi-token outputs and native
sparse training.

## Reproduction

```bash
make theory-checks       # local identities, bounds, feature and precision controls
make approx-validation   # cross-mechanism metric matrix
make trained-audit       # 60 matched interventions per synthetic task
make check               # invariant and documentation tests
```

Artifacts:

- [theory_checks.md](../../reports/theory_checks.md)
- [approximation_validation.md](../../reports/approximation_validation.md)
- [approximation_audit.md](../../reports/approximation_audit.md)
- [approximation_audit_correlations.md](../../reports/approximation_audit_correlations.md)
- [approximation_audit_seed_correlations.md](../../reports/approximation_audit_seed_correlations.md)
- [approximation_audit_training.md](../../reports/approximation_audit_training.md)
- [approximation_audit_raw.md](../../reports/approximation_audit_raw.md)
