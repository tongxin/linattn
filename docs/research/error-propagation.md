# Error propagation: local approximation is not model behavior

The metric hierarchy has a local object and a propagated object. This note makes
the distinction operational.

For a residual block

```text
h_{l+1} = h_l + f_l(h_l)
```

the first-order perturbation is

```text
δh_{l+1} = (I + J_l) δh_l + δf_l
```

where `δf_l` is the local approximation error introduced by the attention
variant. The residual path is not merely a pass-through: downstream Jacobians can
rotate, amplify, or cancel the error. A small value-weighted output error is
therefore not, by itself, a model-level guarantee.

## Controlled experiment

[`../../experiments/error_propagation.py`](../../experiments/error_propagation.py)
uses a six-layer fixed random residual stack. At exactly one layer, a Performer
attention output replaces dense softmax attention; subsequent layers use the
dense path. The experiment records local relative L2 error, final relative L2
error, and their ratio.

The result in [`../../reports/error_propagation.md`](../../reports/error_propagation.md)
shows why the hierarchy needs both levels:

| input | injection layer | local error | final error | amplification |
| --- | ---: | ---: | ---: | ---: |
| random | 0 | 0.436 | 0.552 | 1.27× |
| random | 2 | 0.410 | 0.572 | 1.39× |
| random | 3 | 0.511 | 0.545 | 1.07× |
| structured | 0 | 0.715 | 0.512 | 0.72× |
| structured | 2 | 0.546 | 0.512 | 0.94× |

The local error is not a monotone proxy for the final error. Random-input errors
can grow modestly through the residual stack; structured errors can shrink. This
is a warning against selecting a sparse method solely by its one-layer output
difference.

## Limitations

- The stack is fixed and randomly initialized, not a trained language model.
- Only one attention variant is injected; the interaction between several
  approximate layers is not tested.
- The result is useful for validating the measurement instrument, not for making
  frontier-model quality claims.
- A real model requires matched dense and approximate checkpoints, or a causal
  intervention that replaces one layer while holding all other parameters fixed.

The next protocol must report local support/output metrics, next-layer hidden
drift, final hidden drift, and task behavior together. A benchmark score alone
cannot identify whether an attention approximation, data, or post-training caused
the change.
