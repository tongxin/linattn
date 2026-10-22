# Matched-checkpoint audit

This is the first trained-model evidence in the approximation framework. The
experiment trains a two-layer causal transformer on two synthetic tasks and then
replaces one attention layer at a time while keeping all weights, inputs and
other layers fixed.

This is a **matched intervention**, not a leaderboard. It answers whether a local
attention metric predicts the task consequence on this checkpoint.

## Protocol

- Model: two layers, `d_model=64`, four heads, vocabulary 32, sequence length 48.
- Tasks:
  - single-key retrieval: choose one key from six and report its associated value;
  - dispersed evidence: sum four evidence values spread across the context.
- Training: 900 optimization steps, two seeds.
- Evaluation: 256 held-out examples per checkpoint, every candidate attention
  intervention evaluated one layer at a time.
- Interventions: Quest top-1/top-2 blocks, MoBA top-1/top-2, MInference
  block-sparse, sliding window, Performer with 32 features, the exact-mass
  oracle, and FP8/FP4 KV precision.

## Training controls

| task | seed | train accuracy | eval accuracy | final train loss |
| --- | ---: | ---: | ---: | ---: |
| retrieval | 0 | 0.934 | 0.943 | 0.265 |
| retrieval | 1 | 0.631 | 0.645 | 1.063 |
| dispersed | 0 | 1.000 | 1.000 | 0.003 |
| dispersed | 1 | 0.988 | 0.996 | 0.091 |

The retrieval seeds have different fit quality, which is retained in the report
rather than hidden behind an average. The dispersed task is easier for this
small model.

## Metric-to-task prediction

| task | intervention count | local error vs loss | final hidden vs loss | projected loss vs loss |
| --- | ---: | ---: | ---: | ---: |
| retrieval | 40 | **0.900** | **0.948** | 0.829 |
| dispersed | 40 | **0.766** | **0.973** | 0.670 |

Values are Spearman rank correlations across the one-layer interventions. Final
hidden-state drift is the best predictor in both tasks. The first-order
task-projected loss effect is useful but weaker in this setup.

Representative outputs at equal intervention protocol:

| task | intervention | local attention error | final hidden error | task loss delta | accuracy delta |
| --- | --- | ---: | ---: | ---: | ---: |
| retrieval | Quest top-2 | 0.170 | 0.136 | 0.389 | -0.078 |
| retrieval | MoBA top-2 | 0.118 | 0.157 | 0.190 | -0.035 |
| retrieval | sliding window | 0.527 | 0.736 | 2.320 | -0.387 |
| dispersed | Quest top-2 | 0.082 | 0.068 | 0.029 | -0.008 |
| dispersed | MoBA top-2 | 0.070 | 0.095 | 0.158 | -0.043 |
| dispersed | sliding window | 0.308 | 1.107 | 4.951 | -0.789 |

Rows demonstrate the hierarchy's central point: local attention error and task
degradation are related but not interchangeable, and final hidden drift is often
the stronger bridge.

## What this establishes

- A matched trained checkpoint is now part of the evidence set.
- The metric hierarchy predicts task loss better when it includes final hidden
  drift than when it stops at local attention error.
- A small model can be evaluated reproducibly on both retrieval-shaped and
  dispersed-evidence tasks.
- The result remains synthetic and cannot be generalized to a frontier model.

Run:

```bash
python experiments/approximation_audit.py --steps 900 --seeds 2 --eval-examples 128
```

Machine-readable evidence:

- [`../../reports/approximation_audit.md`](../../reports/approximation_audit.md)
- [`../../reports/approximation_audit_correlations.md`](../../reports/approximation_audit_correlations.md)
- [`../../reports/approximation_audit_training.md`](../../reports/approximation_audit_training.md)
