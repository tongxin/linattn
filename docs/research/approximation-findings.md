# Findings: what it means to approximate full softmax

This is the synthesis of subproblems 2–6. The result is a framework and a set of
evidence boundaries, not a claim that one sparse method is universally better.

## 1. There is no universal cheap-attention guarantee

The strongest result in the recent theoretical line is a negative one:
*Efficiently Approximating Attention Is Hard* (arXiv:2609.37261) rules out a
nontrivial uniform approximation guarantee for truly subquadratic attention under
standard complexity assumptions, even with polynomial KV preprocessing. That does
not make efficient attention useless. It means every useful theorem must assume
input structure, a sampling model, a restricted function class, or a task
distribution.

The positive results fall into distinct families:

- **sampling and retrieval:** priority sampling and MIPS give ε-error bounds with
  explicit dependence on the number of keys and the number of indices
  (arXiv:2610.10135); certified sparse attention gives probabilistic per-query
  certificates (arXiv:2610.24991);
- **streaming/coresets:** discrepancy and polynomial methods give
  ε-approximation with explicit space bounds (arXiv:2502.07861,
  arXiv:2606.07205);
- **dropped support:** sparse-attention error scales with retained or dropped
  attention mass (SSA, arXiv:2511.20102; COBS, arXiv:2607.09052);
- **kernel and low-rank approximation:** Linformer, Performer, Nyströmformer,
  PolySketchFormer and kNN attention each bound a different surrogate object;
- **state compression:** fixed-state recurrence has interference and recall limits
  that are not explained by state size alone (arXiv:2609.33093,
  arXiv:2609.16183).

The map is in [approximation-theory-map.md](approximation-theory-map.md).

## 2. Four levels are necessary

The practical metric hierarchy has four levels:

1. **support/distribution:** mass recall, TV, JS divergence, Hellinger distance;
2. **value-weighted output:** normalized Frobenius error and cosine distance;
3. **hidden state:** drift at the next layer and final layer;
4. **model/task:** predictive distribution, task success and task-specific cost.

A metric at one level can move independently of another. This is not a defect; it
is the central lesson. A selector can retain the same attention mass while
replacing the useful value directions. A one-layer output can be close while the
residual stream amplifies it. A task score can improve because of data or
post-training even when the attention approximation worsens.

## 3. Empirical validation is discriminating

The controlled metric experiment in
[`../../reports/approximation_validation.md`](../../reports/approximation_validation.md)
uses the same small input and read budget for several variants:

- Quest has the lowest structured-input output error in this run (0.016) and
  retains 35% support with very small JS divergence;
- MInference and MoBA are in the same broad regime;
- XAttention is much worse on this small synthetic input despite being a useful
  long-context method in its own setting;
- Performer and Linformer show output error without a reported causal sparse
  support, so they cannot be compared to selector metrics without a separate
  reference computation.

This is a validation of the metric instrument, not a ranking. The result shows
that support, distribution and output metrics are not interchangeable.

## 4. Propagation is real

In the six-layer residual experiment
[`../../reports/error_propagation.md`](../../reports/error_propagation.md), a
Performer layer error injected early in the random-input stack produced a final
error up to ~1.44× the local error. On structured inputs, the same local error
was attenuated downstream. Local error is therefore not monotone with final
error, and the direction of propagation depends on the model and input.

This supports the protocol requirement: report next-layer and final hidden-state
drift, not just attention-output error.

## 5. What remains open

- We do not yet have a theorem connecting a specific sparse-attention estimator
  to a final hidden-state error bound for trained transformers.
- The residual propagation study is synthetic and untrained.
- The quantisation results are mostly model-specific or approximate-local
  statements, not global softmax guarantees.
- The workload protocol has not been run on a real coding or deep-research agent.
- CSA2 remains outside the current result: its official configuration is much
  more complex than the single-layer surrogate, and no public layerwise
  dense-versus-CSA2 ablation exists.

## 6. Conclusion of this research cycle

The right question is not “which sparse attention has the smallest benchmark
loss?” It is:

> Which approximation theorem or sampling model applies, which object does it
> bound, and does that local bound survive propagation to the hidden state and
> task?

The repository now has the vocabulary, theory map, controlled metric tests,
propagation probe, workload protocol, and CSA2 revisit gate needed to ask that
question rigorously. That is a complete first research cycle; the next cycle
should replace the synthetic propagation stack with a matched small checkpoint.
