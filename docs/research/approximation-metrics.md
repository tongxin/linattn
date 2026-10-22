# Approximation metrics for full softmax attention

An efficient attention variant is not one approximation. It can differ from full
attention at the attention operator, the attention distribution, the value-weighted
output, the hidden state, the model distribution, or the final task behavior. A
metric at one level is not automatically evidence about another.

Let `q` be queries, `k,v` keys and values, `d` the head dimension, `T` the context
length, and

```text
a_j = q · k_j / sqrt(d)
p   = softmax(a)
O   = Σ_j p_j v_j
```

The reference is `F_ref(q,k,v) = (p,O)`. A variant defines an approximation
`F_hat`, which may expose a masked distribution `p_hat`, an approximate output
`O_hat`, or only a transformed hidden state.

## 1. Objects and metrics

| object | metric | formula / definition | what it detects | what it misses |
| --- | --- | --- | --- | --- |
| attention distribution | total variation | `TV(p,p_hat)=½‖p-p_hat‖₁` | total probability moved to different keys | which high-value keys were lost |
| attention distribution | Jensen–Shannon | `JS = ½ KL(p‖m) + ½ KL(p_hat‖m)`, `m=(p+p_hat)/2` | symmetric distribution shift, bounded in `[0,log2]` | output impact of value geometry |
| attention distribution | Hellinger | `H=√(½ Σ(√p-√p_hat)²)` | square-root distortion, bounded | value-weighted answer quality |
| sparse support | mass recall | `Σ_{j∈S} p_j / Σ_j p_j` | retained attention mass | whether retained values are the useful ones |
| attention operator | normalized Frobenius | `‖F_ref-F_hat‖_F / max(‖F_ref‖_F,ε)` | value-weighted output error over a query batch | task-level consequence |
| attention operator | cosine distance | `1 - cos(vec(F_ref),vec(F_hat))` | direction drift | magnitude-only errors |
| hidden state | relative L2 / cosine | `‖h_ref-h_hat‖ / max(‖h_ref‖,ε)` | residual-stream drift | whether the model has a usable alternative representation |
| model logits | KL / TV / top-k agreement | distributional distance on softmax logits | predictive distribution change | calibration or task correctness |
| task | task loss / success | benchmark-specific | decision consequence | attribution to one attention mechanism |

All probability metrics require valid non-negative rows that sum to one. A variant
that returns a sparse mask can be evaluated by first obtaining its normalized
attention weights, or by reporting mass recall separately if normalization is not
available.

## 2. Local versus propagated error

A local bound is a statement about one layer:

```text
‖O_ref - O_hat‖ ≤ ε
```

It does not imply that the final model changes by `ε`. Residual networks can
amplify, attenuate, or rotate an error. For a residual stack

```text
h_{l+1} = h_l + f_l(h_l)
```

the first-order propagation is

```text
δh_{l+1} ≈ (I + J_l) δh_l + δf_l
```

so the relevant scalar is not only the local output error but the projected
amplification of that error along the downstream computation. A useful protocol
therefore reports:

1. distribution and support error at the layer;
2. value-weighted output error at the layer;
3. hidden-state drift at the next layer and at the final layer;
4. model-distribution/task error with and without the sparsified layer.

The last comparison is an **intervention**, not a correlation.

## 3. What downstream benchmark scores can and cannot establish

A task score can establish that a complete model is useful under a particular
evaluation and training budget. It cannot by itself establish that a given
attention approximation is close to full softmax: data, post-training, token
budget, other architecture changes and harness choices are confounders.

Task behavior belongs in the hierarchy, but only after a layerwise/error-propagation
measurement has identified the mechanism being tested.

## 4. Invariants and cautions

- Identical inputs and identical operators must produce zero relative error and
  zero distributional distance.
- Total variation, Jensen–Shannon and Hellinger are symmetric; relative output
  error and cosine distance are not meaningful as signed preferences.
- Metrics should be computed after explicit normalization, not from an
  unnormalized score tensor.
- A mask-based selector is causal only if its retained support is causal; mass
  recall on a future-contaminated mask is invalid.
- Per-head metrics must not be silently averaged before checking that heads have
  comparable scale.
- A single random input is not evidence: report both random and structured inputs.
- A local bound should state whether it uses full `T×T` or causal `T(T+1)/2`
  accounting.

The executable definitions are in
[`../../experiments/approximation_metrics.py`](../../experiments/approximation_metrics.py)
and the invariants are tested in
[`../../tests/test_approximation_metrics.py`](../../tests/test_approximation_metrics.py).

## 5. Local error calculus

The hierarchy is useful only if adjacent levels have explicit bridges. Two
bridge inequalities are used throughout the study.

For a fixed reference value set with diameter
`diam(V)=max_{i,j}||v_i-v_j||`,

```text
||O - O_hat||_2 <= TV(p,p_hat) * diam(V).
```

The inequality is tight for adversarially chosen values. It is not reversible:
attention weights may move substantially between keys with identical values,
producing zero output error.

For hard support selection, write `a=Σ_{j∈S}p_j`,

```text
mu_S    = (Σ_{j∈S} p_j v_j) / a
mu_D    = (Σ_{j∉S} p_j v_j) / (1-a)
O       = a mu_S + (1-a) mu_D
O_S     = mu_S
O-O_S   = (1-a)(mu_D-mu_S).
```

The last line is an exact identity. It is stronger than the commonly reported
`2(1-a)V_max` bound because it exposes the direction and magnitude of the
omitted value centroid.

For quantized keys, let `eta=max_j|z_hat_j-z_j|`. The softmax likelihood ratio is
bounded by `exp(2 eta)`, yielding

```text
TV(softmax(z),softmax(z_hat)) <= tanh(eta).
```

Quantized values add a separate weighted perturbation. Reporting the key- and
value-induced output vectors separately prevents a low total error from hiding
two larger canceling errors.

[`../../experiments/approximation_certificates.py`](../../experiments/approximation_certificates.py)
implements and tests these checks:

- For hard sparse support, the omitted contribution is decomposed exactly as
  `(1-a)(μ_omitted-μ_selected)`, where `a` is retained mass. The conservative
  norm bound `2(1-a)V_max` is tested separately.
- For KV quantisation, key-induced and value-induced output errors are reported
  separately, with a softmax logit-perturbation bound `TV ≤ tanh(η)`.
- For distribution/value non-identifiability, two attention distributions with
  identical total variation are paired with equal and orthogonal values. The
  equal-value case has zero output error; the orthogonal-value case has positive
  output error.

These certificates validate the conceptual separation between distribution
error and value-weighted error. They do not compose automatically through a
trained transformer; that requires the propagation study.
