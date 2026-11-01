# Research discussion log

*Living note, last updated 2026-11-01. This is a compact index of the ideas,
questions and side clarifications driving the work. It is not a transcript.*

## Working thesis

The project has two connected layers:

1. **Systems and algorithms:** understand sparse KV reads, compressed state,
   mixed precision, KV storage tiers and hardware/software co-design.
2. **Scientific measurement:** establish what it means for an efficient
   attention mechanism to approximate full softmax attention, without reducing
   the question to a single benchmark score.

The current research focus is the second layer. CSA2 and other deployed models
are applications of the framework, not the framework itself.

## Recurring themes

| theme | questions / interests | current artifacts |
| --- | --- | --- |
| Attention survey | Which sparse, compressed-state and kernel techniques matter, and which are actually deployed? | [`survey.md`](../survey.md), [`academic-attention.md`](../academic-attention.md) |
| KV storage layer | How do cache format, hit rate, tiering, eviction, reuse and quantisation affect end-to-end cost? | [`kv-layer.md`](../kv-layer.md), [`kv/`](../../src/linattn/kv/) |
| Academic recency | Which 2025–2026 works changed the argument rather than merely adding another benchmark? | [`academic-attention.md`](../academic-attention.md) |
| Approximation framework | What object is approximated: attention distribution, value-weighted output, hidden state or task behavior? | [`attention-approximation-report.md`](attention-approximation-report.md) |
| Theory | Which impossibility results, sparse-support identities, sampling certificates and compressed-state limits apply? | [`approximation-theory-map.md`](approximation-theory-map.md) |
| Metrics | Which metrics are meaningful at each level, and which are non-identifying? | [`approximation-metrics.md`](approximation-metrics.md) |
| Empirical validation | Do the metrics distinguish mechanisms, and does local error predict downstream loss? | [`theory_checks.md`](../../reports/theory_checks.md), [`approximation_validation.md`](../../reports/approximation_validation.md), [`approximation_audit.md`](../../reports/approximation_audit.md) |
| Workloads | How do retrieval, dispersed evidence, coding, long-horizon agents and deep research differ? | [`attention-evaluation-protocol.md`](attention-evaluation-protocol.md) |
| MLX / Apple Silicon | Can kernels and wrappers be validated locally on a Mac without CUDA Triton? | [`mlx.md`](../mlx.md), [`triton-on-apple.md`](../triton-on-apple.md) |
| Cost accounting | What does “FLOPs”, “KV reads” and “cache hit rate” mean, and which convention is used? | [`analytic-flops.md`](../analytic-flops.md), [`kv-layer.md`](../kv-layer.md) |

## Open questions

1. Does a metric hierarchy predict model degradation across mechanisms and task
   families, or only within one checkpoint?
2. How should sparse selection, pooled representation, KV precision and
   compressed state be separated experimentally?
3. When does a local attention error amplify through residual layers, and when
   does the model attenuate or compensate for it?
4. Which natural-language or code-trained checkpoint is small enough to run but
   representative enough to test multi-token tasks and native sparse training?
5. How should a deployed system be evaluated when no matched dense checkpoint or
   dense teacher is available?
6. Which theoretical results are peer-reviewed, which are conditional
   certificates, and which are only preprints or empirical observations?
7. What evaluation protocol exposes dispersed evidence rather than only
   RULER-style needle retrieval?

## CSA2 position

CSA2 is a strong deployment-oriented case study, not the current research gate.
The public V4.1-Flash/1017 configuration is substantially disclosed, but the
available evidence is whole-model and does not isolate a dense-versus-CSA2
layerwise approximation.

Keep CSA2 in the backlog as a future application. Before revisiting it, use the
general framework to define a matched checkpoint, layerwise diagnostics,
precision ablations, propagation measurements and evidence-sparse tasks.

See [`csa2-approximation.md`](../csa2-approximation.md),
[`csa2-official-config.md`](../csa2-official-config.md) and
[`roadmap.md`](../roadmap.md).

## Side-chat clarifications worth retaining

| question | resolution |
| --- | --- |
| What does `@T` mean in cost tables? | `T` is the sequence/context length passed as `--seq`; it is not throughput or generation time. |
| Are the FLOPs measured on hardware? | No. They are a matmul-only analytic prefill estimate. |
| Why did hand verification differ? | The base model counts a full `T×T` matrix; causal counting is about half, and projections/softmax are excluded. |
| What precision does `KV/state @T` use? | The table has an explicit storage-precision column: FP16 KV, latent FP16, FP8/FP4 compressed, or the actual format override. |
| Was there an actual `V4.1-Flash-1107`? | No official 1107 model was found in public metadata as of the review. The public official release is V4.1-Flash and its 1017 minor update. |
| Is CSA2 the research endpoint? | No. CSA2 is a deferred application; the research endpoint is a general approximation framework. |
| Why have multiple venvs? | The main environment is `.venv`; the Triton-on-Metal experiment uses a separate Python 3.14 environment because of ABI/wheel constraints. |

## Evaluation discipline

- Do not call an agent status update a scientific conclusion.
- Do not use downstream accuracy alone as evidence of a layerwise approximation.
- Do not compare different checkpoints, budgets or training states as if they were
  a controlled approximation experiment.
- Report local error, hidden-state propagation and task consequences separately.
- State whether the evidence is algebraic, conditional, empirical on synthetic
  data, or empirical on a trained checkpoint.

## Current status

The first validation cycle is complete:

- metric hierarchy and executable invariants;
- exact sparse-support and quantisation certificates;
- multi-seed synthetic mechanism checks;
- matched tiny-checkpoint retrieval/dispersed-evidence audit;
- residual-stack propagation probe;
- general workload protocol and next-round plan.

The report is **not final for frontier models**. The next round should use a
small natural-language or code-trained checkpoint, multi-token tasks and native
sparse training. See [`next-round-plan.md`](next-round-plan.md).
