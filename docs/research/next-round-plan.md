# Next-round plan: from measurement framework to trained-model evidence

The first cycle now includes matched tiny-checkpoint interventions. It supports
the measurement framework and shows that downstream hidden-state drift adds
information beyond local attention error, but the checkpoints are synthetic and
single-token. This next round tests whether the result survives natural-language
or code training, multi-token outputs, and multi-layer replacement.

## Objective

Estimate the layer-to-task approximation profile of at least two efficient
attention mechanisms on a small natural-language or code-trained checkpoint,
then compare the measured task degradation with local, hidden-state, and
task-projected predictions.

## Round 1 — extend to a matched natural-language checkpoint

The repository now has a matched tiny synthetic checkpoint audit. The next round
must extend it to a small natural-language or code-trained causal transformer
with:

- the same parameters and training data for dense and efficient variants;
- a sparse selector with a fixed read budget;
- a kernel/low-rank variant with a stated feature budget;
- an optional quantized-KV variant;
- a task with a retrieval-shaped and an evidence-sparse split.

Required artifacts:

- checkpoint hashes and training command;
- dense and efficient forward paths sharing all non-attention parameters;
- per-layer attention distributions where available;
- per-layer output/hidden-state traces;
- task accuracy, loss, token cost and actual read counts.

Exit criterion: the dense reference is trained to a stable baseline and the
efficient intervention runs against the same weights without leakage or budget
violations.

## Round 2 — test the hierarchy

For each intervention, measure:

1. support and mass recall;
2. attention/output distribution distance;
3. value-weighted output error;
4. next-layer and final hidden-state drift;
5. task delta and token cost.

Evaluate the four hypotheses from the current report across at least three seeds
and two task families.

Exit criterion: either a metric predicts task degradation across mechanisms, or
the negative result is reported. Do not declare success from a single task.

## Round 3 — verify the literature

For every theory-map entry used in the final report:

- verify title, authors, venue/status and theorem statement;
- record the exact assumptions and approximation object;
- distinguish theorem, empirical observation and conjecture;
- add a citation to `docs/citations.bib`.

Exit criterion: no unverified preprint is used as a guarantee, and no theorem is
generalized beyond its stated object.

## Round 4 — write the final report

The final report must contain:

- one problem statement;
- one theory section tied to the metric hierarchy;
- one methodology section with matched controls;
- a results section with confidence intervals and negative results;
- a limitations section that distinguishes synthetic from trained evidence;
- a clear conclusion and a list of unresolved questions.

Exit criterion: a reader can reproduce every table from repository artifacts and
identify exactly which claims are theoretical, empirical on synthetic inputs, or
empirical on a trained checkpoint.

## Out of scope

CSA2 and other frontier systems are examples for a later application round. They
are not the target of this general framework, and their absence does not block
the next round.
