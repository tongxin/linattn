# Approximating full softmax attention: a measurement framework and first validation cycle

## Abstract

Efficient attention methods are usually compared through benchmark scores, yet
they approximate different mathematical objects. A sparse selector changes an
attention support, a random-feature method approximates a kernel, a low-rank
method projects the sequence axis, and a quantized cache perturbs stored keys and
values. This report asks what it means to say that one of these methods is close
to full softmax attention. We propose a four-level measurement hierarchy:
attention support/distribution, value-weighted output, propagated hidden state,
and model/task behavior. We map recent theoretical results to those levels,
separate impossibility results from structured guarantees, and run three
controlled validation studies on the repository's variants. The results support
four conclusions: (1) no universal cheap-attention guarantee is possible for
arbitrary inputs; (2) support mass alone does not determine output error; (3)
kernel and precision approximations show the expected error-reduction trends,
but the convergence can be slow and seed-dependent; (4) local attention error can
attenuate or amplify through the residual stream. The report ends with decision
rules for applying the framework to any deployed efficient-attention system.

## 1. Problem statement

Let

```text
a_j = q · k_j / sqrt(d)
p = softmax(a)
O = Σ_j p_j v_j
```

be the full attention distribution and value-weighted output for one query and
head. An efficient variant defines an approximation `F_hat`. The central
question is:

> Under what assumptions does `F_hat` approximate `F_ref`, at which level of the
> computation graph, and how does that local approximation propagate to model
> behavior?

This wording is deliberately more precise than “sparse attention is lossless.”
An approximation can have small attention-distribution error but large output
error, small output error but large hidden-state drift, or no local error but a
different downstream task distribution. The report treats these as distinct
claims requiring distinct evidence.

## 2. What the literature establishes

The modern theory splits into five non-interchangeable families.

### 2.1 Uniform approximation is impossible

*Efficiently Approximating Attention Is Hard* (arXiv:2609.37261) shows that under
standard complexity assumptions no truly subquadratic method gives a nontrivial
uniform additive or relative guarantee over arbitrary inputs, even with
polynomial KV preprocessing. This rules out a universal theorem saying that a
general sparse or compressed attention is “within ε” of dense attention. It does
not rule out useful guarantees for structured inputs, sampling schemes, or
restricted function classes.

### 2.2 Sampling and streaming can be certified under models

Priority sampling/MIPS results describe how many keys are needed for an
ε-output estimate (arXiv:2610.10135). Certified sparse attention uses perturbed
retrieval and Horvitz–Thompson estimates with simultaneous confidence bounds
(arXiv:2610.24991). Streaming coreset and discrepancy methods bound approximation
with explicit space costs (arXiv:2502.07861, arXiv:2606.07205).

These are genuine guarantees, but they rely on a sampling model or on a
well-defined attention primitive. They do not certify a deployed heuristic on
arbitrary language data merely because it uses similar scoring.

### 2.3 Dropped attention mass gives a useful local bound

SSA states that sparse-attention approximation error scales with dropped mass
(arXiv:2511.20102). COBS reduces exact block selection to attention-mass ranking
(arXiv:2607.09052); CEDAR adds a refinement bound based on within-chunk key/value
dispersion (arXiv:2609.07237). The repository validates the simple consequence:
if the selected support retains all but mass `δ`, and values have norm at most
`V_max`, then

```text
||O_ref - O_selected||₂ ≤ 2 δ V_max.
```

This is a bound on the selected value-weighted output, not on the final model and
not a statement that mass alone determines quality.

### 2.4 Kernel and low-rank methods bound different surrogates

Linformer's low-rank projection, Performer's positive random features,
Nyström landmark reconstruction and polynomial sketches each approximate a
different mathematical object. Performer is unbiased in expectation; that does
not give a small per-query error, and the variance is visible in controlled runs.
A low-rank projection can be accurate when attention is approximately low-rank
but is not a sparse-support bound. These results must be attached to the metric
they actually constrain.

### 2.5 Compressed state has an expressivity and interference problem

Linear-attention and state-space models replace token-addressable KV with a fixed
state. Recent controlled work shows that recall depends on write/read geometry,
interference and training curriculum, not merely state size
(arXiv:2609.33093, arXiv:2609.16183). A state method can have perfect storage
efficiency and still fail exact associative recall.

The complete claim map is in [approximation-theory-map.md](approximation-theory-map.md).

## 3. Measurement framework

### 3.1 Metric hierarchy

| level | primary metrics | question |
| --- | --- | --- |
| support/distribution | mass recall, TV, JS, Hellinger, top-k agreement | which probability mass and keys were retained? |
| value-weighted output | normalized L2, cosine distance | did retained/changed mass preserve the attention output? |
| hidden state | next-layer and final relative L2, cosine drift | did the local error survive the residual stack? |
| model/task | predictive divergence, task success, token/tool cost | did the model's behavior change? |

The definitions and invariants are in [approximation-metrics.md](approximation-metrics.md).

### 3.2 Falsifiable hypotheses

The study tests four hypotheses:

**H1. Sparse-support bound.** For mask-based selectors, the selected output obeys
the dropped-mass/value-norm bound.

**H2. Non-implication.** Similar mass recall can coexist with different
value-weighted output errors; support metrics are not a substitute for output
metrics.

**H3. Estimator/precision trends.** Increasing Performer feature count reduces
mean error, and FP4 produces larger KV perturbation than FP8, with FP16 exact.

**H4. Propagation.** The final hidden-state error is not a fixed scalar multiple
of local attention error; injection layer and input structure change amplification.

## 4. Experimental design

All experiments are CPU-only and synthetic. They validate measurement
instruments, not frontier-model quality.

- **Inputs:** random K/V, single-spike retrieval, and dispersed multi-block
  retrieval.
- **Metric matrix:** Quest, MInference block-sparse, MoBA, XAttention, Performer,
  and Linformer; T=128, H=4, H_kv=2, d=32, three seeds.
- **Theory checks:** selection bound across four selector families and three input
  families; Performer feature counts 4–256; group-wise KV precision FP16/FP8/FP4.
- **Propagation:** six-layer random residual stack; one Performer layer replaces
  dense attention at a controlled injection layer; three seeds.

Reproduce:

```bash
make approx-validation
make theory-checks
make error-propagation
```

## 5. Results

### 5.1 The selection bound holds, but it is loose

Across random, retrieval and dispersed inputs, every mask-based selector
satisfied the per-query bound `||O_ref-O_selected|| ≤ 2δV_max`. The mean realized
bound ratio ranged from 0.13 to 0.28: the inequality is valid but not tight
enough to explain the quality ordering. See
[`reports/theory_checks.md`](../../reports/theory_checks.md).

This is useful: a selector cannot be accepted merely because its dropped mass is
small, but a large dropped mass does guarantee a large worst-case risk.

### 5.2 Metric levels are not interchangeable

Three-seed means on the metric matrix:

| input | variant | output rel-L2 | support fraction | JS divergence |
| --- | --- | ---: | ---: | ---: |
| random | Quest | 0.195 ± 0.009 | 0.417 | 0.037 |
| random | MoBA | 0.248 ± 0.007 | 0.368 | 0.054 |
| random | MInference | 0.299 ± 0.007 | 0.370 | 0.066 |
| random | XAttention | 0.802 ± 0.083 | 0.242 | 0.193 |
| random | Performer | 0.620 ± 0.013 | n/a | n/a |
| random | Linformer | 1.395 ± 0.059 | n/a | n/a |
| structured | Quest | 0.068 ± 0.060 | 0.353 | 0.005 |
| structured | MoBA | 0.529 ± 0.019 | 0.371 | 0.068 |
| structured | MInference | 0.548 ± 0.043 | 0.374 | 0.075 |
| structured | XAttention | 0.984 ± 0.050 | 0.114 | 0.263 |

The ordering is not the point. The point is that support, distribution and output
metrics diverge. MoBA and MInference retain nearly identical support fractions but
have different output errors; XAttention retains much less support and also has a
different distributional error. A mask recall alone would not identify these
differences.

See [`reports/approximation_validation.md`](../../reports/approximation_validation.md).

### 5.3 Kernel and precision trends are real but slow

Performer error decreased as feature count increased, but the convergence was
slow and not monotone on every seed:

| input | 4 features | 256 features |
| --- | ---: | ---: |
| random | 0.775 ± 0.047 | 0.586 ± 0.010 |
| dispersed | 0.983 ± 0.027 | 0.871 ± 0.012 |

FP16 was exact in the KV codec, FP8 was better than FP4, and both degraded with
more heterogeneous values:

| input | FP16 | FP8 | FP4 |
| --- | ---: | ---: | ---: |
| random | 0 | 0.030 | 0.122 |
| dispersed | 0 | 0.037 | 0.158 |

This validates the estimator/precision direction but shows why “unbiased” or
“low precision” is not a complete quality claim.

### 5.4 Propagation changes the ranking

In the residual-stack experiment, random-input local errors of roughly 0.36–0.54
produced final errors of 0.48–0.55, with amplification as high as 1.41×. On
structured inputs the same injection was often attenuated, with amplification
below 1. The local-error ranking therefore does not determine the final-error
ranking.

See [`reports/error_propagation.md`](../../reports/error_propagation.md).

## 6. What this changes

### For research claims

Do not report “attention approximation error” without naming the level. The
minimum defensible claim names the input distribution, reference operator,
metric, support/read budget and whether training was matched.

### For method design

A sparse selector should report both support behavior and value-weighted output
error. A kernel method should report variance and feature count, not only an
unbiasedness statement. A precision method should report the error of the
attention output under the actual cache format. A compressed-state method should
report associative-recall/interference diagnostics.

### For deployed systems

The framework is general and should be instantiated explicitly for each deployment.
For any specific system, state the reference, the approximation mechanism, the
precision policy, the read budget, the workload family and the observation level.
CSA2 is one deferred application in the [roadmap](../roadmap.md), not a required
part of the general framework.

## 7. Limitations

- Synthetic inputs are not natural-language distributions.
- The residual stack is untrained and is not a transformer checkpoint.
- The sparse variants are selected at small budgets and do not include native
  sparse training.
- The metric matrix uses one layer; it does not establish model-level behavior.
- The theory map includes preprints whose assumptions need independent review.
- No conclusion here is a frontier-model quality guarantee.

## 8. Conclusion

The correct research question is not “which efficient attention has the lowest
benchmark error?” It is:

> Which object is being approximated, under which assumptions, and does the local
> bound survive propagation to hidden states and task behavior?

This cycle establishes that question operationally. It supplies a tested metric
hierarchy, verified claims from the literature, multi-seed controlled evidence,
and a local-to-final propagation probe. The next scientific step is a matched
small checkpoint with the same general protocol, not another synthetic
leaderboard.
