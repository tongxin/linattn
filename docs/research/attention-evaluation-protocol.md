# Evaluation protocol for efficient-attention approximation

This protocol is the workload-sensitive part of the research program. It is
designed to expose the difference between retrieval-shaped and distributed-evidence
workloads; it is not a leaderboard.

## 1. Required comparisons

For each claim, report:

| dimension | required control |
| --- | --- |
| model | same checkpoint or explicitly matched training recipe |
| attention budget | same selected entries/pages/tokens and same context length |
| training state | native sparse training vs post-hoc/adaptation |
| precision | FP16 reference, then FP8/FP4 as separate ablations |
| layer intervention | one approximate layer vs all approximate layers |
| causal domain | same visible-token domain; never compare a noncausal projection with a causal dense reference |
| workload | retrieval, dispersed evidence, multi-hop/chain, coding, agentic |
| cost | layer FLOPs, KV bytes, read bytes, and end-to-end latency |

The dense reference must use the same model state, pre-attention features, and
causal domain. A different model, different training length, a changed tokenizer,
or a causal/noncausal mismatch is not a controlled approximation comparison.

## 2. Workload families

### Retrieval-shaped

One or a few keys carry most of the answer mass. Use single-needle and
many-needle retrieval, page-copy, and code-symbol lookup. These tasks are where
top-k/page selection is expected to look nearly lossless.

### Dispersed evidence

The answer depends on several moderate-mass passages, none individually dominant.
Use multi-document QA with evidence planted in different blocks, and require all
evidence to be cited or combined. This is the key contrast to needle retrieval.

### Compositional / long-horizon

Require repeated use of an early fact or plan after intervening tool observations
and turns. Measure task success, number of turns, total tokens and cache reads,
not only final answer accuracy.

### Coding / agentic

Use repository-level repair and tool-use tasks with:

- a small set of exact lines that must be retained;
- a large amount of boilerplate/distractors;
- multiple turns of tool output;
- a fail-to-pass test suite.

Report both task success and the number of tokens/tool calls spent.

## 3. Measurement stack

For every run, record:

1. attention support/mass recall for selector variants;
2. distributional distance (TV/JS/Hellinger) where a normalized distribution is available;
3. value-weighted output error;
4. next-layer and final hidden-state drift;
5. model predictive distribution / task outcome;
6. cache footprint and actual read budget;
7. prefill/decode latency under a fixed serving setup.

The first four are mechanism diagnostics; the last two are deployment outcomes;
task outcome is the decision consequence. No one column substitutes for the
others.

## 4. Expected discriminating pattern

The protocol should reproduce this qualitative pattern if the theory is useful:

- retrieval-shaped tasks tolerate aggressive support selection when prompt/recent
  mass is retained;
- dispersed-evidence tasks lose quality when selection removes moderate-mass
  evidence, even at similar attention-mass recall;
- native training can compensate for a fixed pattern, while post-hoc compression
  without training cannot;
- quantisation error may be small relative to selection/pooling error on some
  workloads, but can become material when values or key directions are outliers;
- a local output error that looks small can amplify downstream; final hidden-state
  drift is the bridge metric, not a replacement for task success.

## 5. Stopping and reporting rule

An experiment is complete when it either:

1. shows the measured ordering within each of at least three checkpoints and
   validates it across at least two workload families; or
2. clearly fails to discriminate the mechanisms and is reported as negative
   evidence.

Do not add another benchmark merely to rescue a weak result. Do not claim a
frontier-model approximation from a synthetic residual stack. Applying the
protocol to a specific deployed checkpoint is a separate roadmap decision and
requires a matched dense reference.
