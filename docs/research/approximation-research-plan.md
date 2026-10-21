# Research plan: approximation to full softmax attention

This plan turns the broad question “how close is an efficient attention variant
to full softmax?” into six ordered subproblems. The rule is that a subproblem is
not allowed to borrow credibility from a downstream benchmark alone. Each stage
must expose its assumptions, its measurement object, and the kind of error it
does *not* capture.

CSA2 sensitivity is deliberately outside this plan until the framework is ready.
It remains a backlog item in [`../roadmap.md`](../roadmap.md).

## Subproblem 1 — Formalize the objects and metric hierarchy

**Question.** What exactly is being approximated, and which metric is meaningful
at each level of the computation graph?

**Concepts.** Full attention operator; variant operator; attention distribution;
value-weighted output; hidden state; model distribution; task behavior; local
error; propagated error.

**Deliverable.** A formal object/metric document plus executable metric functions
and invariants.

**Acceptance test.** Identical tensor inputs produce zero error where the metric
admits a zero; probability metrics are symmetric and bounded; changing only the
reference distribution produces a detectable nonzero error; all metrics are
finite for valid inputs.

**Dependency.** None.

**Stop condition.** The repository has a documented and tested first-level metric
vocabulary. Do not start a broad literature crawl in this subproblem.

## Subproblem 2 — Map existing approximation theories to the hierarchy

**Question.** Which theoretical results actually bound each level, under which
assumptions?

**Concepts.** Low-rank/operator approximation, random features and kernel error,
block selection and missing attention mass, quantization perturbation,
compressed-state/recall separations, and error propagation through residual
networks.

**Deliverable.** A claim-by-claim literature map with theorem type, assumptions,
guarantee level, and failure mode. Separate peer-reviewed results from preprints.

**Acceptance test.** Every theorem is attached to a metric in subproblem 1; no
paper is treated as a bound when it only reports a benchmark.

**Dependency.** Subproblem 1.

**Stop condition.** Ten to fifteen load-bearing references are mapped, not an
exhaustive bibliography.

## Subproblem 3 — Validate metrics on controlled variants

**Question.** Do the metrics distinguish mechanisms that are cheap but
approximate them in materially different ways?

**Concepts.** Exactness limits, monotonicity, metric saturation, distribution vs
output sensitivity, and causal-vs-full-matrix accounting.

**Deliverable.** A CPU experiment over dense, sparse-selection, low-rank,
fixed-state, and quantized variants with all metrics reported.

**Acceptance test.** Degenerate exact configurations produce zero/epsilon error;
known approximations produce monotone or explainable trends; metrics are not
reducible to one another on adversarial inputs.

**Dependency.** Subproblem 1; subproblem 2 informs the hypotheses.

**Stop condition.** A metric matrix is stable across two synthetic distributions
and one structured-retrieval distribution. Do not run frontier checkpoints yet.

## Subproblem 4 — Derive and test layer-to-task propagation

**Question.** When does a local attention error remain local, and when does it
amplify through the residual stream and subsequent layers?

**Concepts.** Lipschitz/Jacobian amplification, residual-stream norms, hidden-state
drift, output-sensitivity probes, and composition bounds.

**Deliverable.** A propagation experiment on a small trainable transformer or a
controlled residual stack, with local and downstream errors on the same checkpoint.

**Acceptance test.** The propagation model predicts the sign/order of hidden-state
and task-level degradation better than local output error alone.

**Dependency.** Subproblems 1 and 3.

**Stop condition.** A small-model result with explicit limitations; do not infer a
frontier-model guarantee from it.

## Subproblem 5 — Build a workload-sensitive evaluation protocol

**Question.** Which evaluations expose the failure modes that matter for coding,
long-horizon reasoning, and deep research?

**Concepts.** Dispersed evidence, relevant-token mass, multi-hop chains, tool
observations, context length, and task success vs token cost.

**Deliverable.** A protocol that reports layer metrics, propagation metrics, task
success, and deployment cost jointly.

**Acceptance test.** The protocol distinguishes retrieval-shaped tasks where
top-k selection is nearly optimal from evidence-sparse tasks where it is not.

**Dependency.** Subproblems 3 and 4.

**Stop condition.** A small set of controlled tasks and clear reporting rules;
no leaderboard claims.

## Subproblem 6 — Synthesize and gate the CSA2 revisit

**Question.** What evidence would justify applying the framework to CSA2 and other
deployed sparse designs?

**Concepts.** Runnable checkpoint, dense teacher, layerwise output/hidden-state
comparison, selection/mass diagnostics, precision ablation, and matched workloads.

**Deliverable.** A decision framework and a CSA2-specific experiment gate.

**Acceptance test.** The gate requires a dense or matched reference, disclosed
configuration, and a task suite that includes evidence-sparse workloads.

**Dependency.** All earlier subproblems; not part of the current execution run.

**Stop condition.** Publish the gate and return CSA2 to the backlog until the
required checkpoint-level evidence exists.

## Execution state

| subproblem | state |
| --- | --- |
| 1. Objects and metric hierarchy | implementation checkpoint complete; advisor review pending |
| 2. Theory map | not started; requires advisor approval |
| 3. Controlled metric validation | not started |
| 4. Propagation study | not started |
| 5. Workload protocol | not started |
| 6. CSA2 revisit gate | not started |
