# Matched-checkpoint audit

This study tests the central empirical question of the approximation framework:
does local attention error predict the loss caused by replacing dense attention
in an otherwise identical trained causal language model?

It is a **causal intervention**, not a benchmark correlation. Every intervention
uses the same parameters, token batch, position in the layer stack, and
post-attention computation as the dense reference.

## Protocol

- Model: two causal transformer layers, `d_model=64`, four heads,
  `head_dim=16`, vocabulary 32, sequence length 48.
- Task 1: retrieval from four key/value pairs in the context.
- Task 2: add two evidence values placed in separate regions of the context.
- Training: three seeded checkpoints per task, 2,200 AdamW steps per checkpoint.
- Evaluation: 256 held-out examples per checkpoint.
- Intervention: replace exactly one layer's attention while all weights and the
  other layer remain unchanged.
- Variants: Quest top-1/top-2, MoBA top-1/top-2, MInference block-sparse,
  exact-mass block oracle top-2, sliding window 8, Performer with 32 features,
  and groupwise FP8/FP4 KV round-trips.
- Total: 10 interventions x 2 layers x 3 checkpoints = 60 interventions per
  task.

The exact-mass oracle uses dense attention weights to choose two 8-token blocks.
It is an analysis upper bound for this block budget, not a deployable selector.

## Training controls

| task | checkpoint | train accuracy | held-out accuracy |
| --- | ---: | ---: | ---: |
| retrieval | 0 | 1.000 | 1.000 |
| retrieval | 1 | 1.000 | 1.000 |
| retrieval | 2 | 0.984 | 0.979 |
| dispersed | 0 | 1.000 | 1.000 |
| dispersed | 1 | 1.000 | 0.994 |
| dispersed | 2 | 1.000 | 1.000 |

The checkpoints are sufficiently fit to make prediction loss informative. They
are still small synthetic models, not evidence about natural-language models.

## Metric-to-task prediction

Spearman correlation across all interventions and checkpoints:

| task | local attention error | final hidden drift | first-order `g·delta` projection |
| --- | ---: | ---: | ---: |
| retrieval, pooled | 0.914 | **0.966** | 0.582 |
| retrieval, within-checkpoint mean | 0.927 ± 0.029 | **0.974 ± 0.006** | 0.853 ± 0.072 |
| dispersed, pooled | 0.869 | 0.923 | 0.723 |
| dispersed, within-checkpoint mean | 0.898 ± 0.005 | 0.934 ± 0.014 | **0.960 ± 0.016** |

Final hidden drift has the strongest within-checkpoint mean in both tasks. Local
attention error still has high rank correlation, so the correct conclusion is not
“local error is irrelevant.” It is that local error is insufficient and can tie
interventions with materially different downstream damage.

The pooled and within-checkpoint first-order correlations differ sharply. The
pooled number mixes differently calibrated checkpoints; the within-checkpoint
number is the defensible one for ranking interventions. This is a measurement
lesson in its own right.

## Rank reversals

| task | intervention | local error | final hidden drift | loss increase |
| --- | --- | ---: | ---: | ---: |
| retrieval | Quest top-2 | 0.142 | 0.134 | 0.149 |
| retrieval | MInference block | 0.173 | 0.437 | 1.432 |
| dispersed | MoBA top-2 | 0.081 | 0.090 | 0.331 |
| dispersed | MInference block | 0.086 | 0.340 | 1.570 |

In the dispersed case, MInference and MoBA top-2 have nearly the same local
error, but MInference causes about 4.8x the loss increase and 3.8x the final
hidden drift. A local scalar therefore misses the direction of the error relative
to the rest of the checkpoint.

The first-order projection `g^T delta_a` is useful as a directional diagnostic.
It is weaker than hidden drift for retrieval, stronger for dispersed evidence,
and misleading when loss deltas from different checkpoints are pooled without
normalisation. Finite errors are still affected by curvature and cancellation.

## What the study establishes

- The evaluation protocol can distinguish retrieval-shaped from
  dispersed-evidence failures under matched weights.
- Final hidden drift predicts task degradation better than local attention
  output error in both task families.
- Similar local errors can hide large task differences.
- FP8 and FP4 KV perturbations preserve the expected precision order in this
  checkpoint, but FP4 produces a measurable local and task effect.
- The result is evidence for the metric hierarchy and against single-scalar
  approximation claims.

## What it does not establish

- It does not estimate natural-language, coding, or frontier-model behavior.
- It does not test multi-token reasoning or long-horizon compounding.
- It does not test native sparse training or learned selectors.
- Correlations from two-layer models are not laws of transformer error
  propagation.

## Reproduce

```bash
make trained-audit
```

The command trains six small checkpoints on CPU. Generated evidence:

- [approximation_audit.md](../../reports/approximation_audit.md)
- [approximation_audit_correlations.md](../../reports/approximation_audit_correlations.md)
- [approximation_audit_seed_correlations.md](../../reports/approximation_audit_seed_correlations.md)
- [approximation_audit_training.md](../../reports/approximation_audit_training.md)
- [approximation_audit_raw.md](../../reports/approximation_audit_raw.md)
