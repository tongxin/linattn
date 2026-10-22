# Matched-checkpoint audit: local attention error to task behavior

*Generated 2026-10-22T02:03:44Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

Two-layer trained causal transformers on synthetic four-key retrieval and dispersed two-evidence summation. One attention layer is replaced at a time while weights and other layers remain fixed. task_projected_abs is the magnitude of the first-order loss effect g.delta_a on the dense trajectory; it uses the downstream Jacobian but is not a finite-error guarantee. Synthetic tasks validate the audit instrument, not frontier-model quality.

| task | variant | n | local_attention_rel_l2 | final_hidden_rel_l2 | logit_kl | loss_delta | task_projected_abs | accuracy_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retrieval | kv-fp4 | 6 | 0.0916 | 0.07419 | 0.004206 | 0.006054 | 1.605e-05 | -0.003255 |
| retrieval | kv-fp8 | 6 | 0.02117 | 0.01794 | 0.0001101 | -6.27e-05 | 3.821e-06 | 0.000651 |
| retrieval | mass-oracle-2block | 6 | 0.051 | 0.02328 | 0.0001333 | 0.0003842 | 4.152e-06 | 0 |
| retrieval | minference-block | 6 | 0.1732 | 0.4369 | 1.437 | 1.432 | 0.0001527 | -0.1992 |
| retrieval | moba-1block | 6 | 0.3671 | 0.4522 | 1.571 | 1.572 | 0.00014 | -0.2331 |
| retrieval | moba-2block | 6 | 0.1388 | 0.1685 | 0.3471 | 0.3424 | 8.242e-05 | -0.05339 |
| retrieval | performer-32 | 6 | 0.5694 | 0.9136 | 4.598 | 4.608 | 0.0001534 | -0.6758 |
| retrieval | quest-1block | 6 | 0.7604 | 0.9586 | 5.173 | 5.172 | 0.0001609 | -0.5866 |
| retrieval | quest-2block | 6 | 0.1421 | 0.1335 | 0.1469 | 0.1491 | 3.003e-05 | -0.02995 |
| retrieval | sliding-window | 6 | 0.4889 | 0.8377 | 4.235 | 4.224 | 0.0002041 | -0.4941 |
| dispersed | kv-fp4 | 6 | 0.06873 | 0.08008 | 0.003038 | 0.001292 | 1.513e-05 | -0.001302 |
| dispersed | kv-fp8 | 6 | 0.01372 | 0.01902 | 6.101e-05 | -3.832e-05 | 2.313e-06 | 0 |
| dispersed | mass-oracle-2block | 6 | 0.05568 | 0.01456 | 0.0003488 | 0.0008733 | 3.555e-06 | -0.001302 |
| dispersed | minference-block | 6 | 0.08558 | 0.3397 | 1.565 | 1.57 | 4.055e-05 | -0.1875 |
| dispersed | moba-1block | 6 | 0.1625 | 0.2133 | 0.9337 | 0.9377 | 3.77e-05 | -0.1113 |
| dispersed | moba-2block | 6 | 0.08116 | 0.08957 | 0.3303 | 0.3305 | 2.257e-05 | -0.04232 |
| dispersed | performer-32 | 6 | 0.3444 | 0.838 | 4.107 | 4.124 | 0.0001065 | -0.5033 |
| dispersed | quest-1block | 6 | 0.4433 | 1.102 | 7.149 | 7.179 | 0.0001561 | -0.7578 |
| dispersed | quest-2block | 6 | 0.08585 | 0.04615 | 0.03451 | 0.03268 | 1.676e-05 | -0.007161 |
| dispersed | sliding-window | 6 | 0.3298 | 1.085 | 7.041 | 7.067 | 0.0001464 | -0.75 |
