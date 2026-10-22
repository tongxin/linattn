# Matched-checkpoint audit: local attention error to task behavior

*Generated 2026-10-22T01:38:38Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

Two-layer trained causal transformers on synthetic single-key retrieval and dispersed four-evidence summation. One attention layer is replaced at a time while weights and other layers remain fixed. task_projected_abs is the magnitude of the first-order loss effect g.delta_a on the dense trajectory; it uses the downstream Jacobian but is not a finite-error guarantee. Synthetic tasks validate the audit instrument, not frontier-model quality.

| task | variant | n | local_attention_rel_l2 | final_hidden_rel_l2 | logit_kl | loss_delta | task_projected_abs | accuracy_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retrieval | kv-fp4 | 6 | 0.08709 | 0.07164 | 0.01968 | 0.01342 | 0.0002265 | -0.009766 |
| retrieval | kv-fp8 | 6 | 0.01951 | 0.0168 | 0.0007626 | 0.0002725 | 4.77e-05 | -0.000651 |
| retrieval | mass-oracle-2block | 6 | 0.05148 | 0.02424 | 0.00332 | 0.003326 | 6.645e-05 | -0.002604 |
| retrieval | minference-block | 6 | 0.1404 | 0.4486 | 1.21 | 1.21 | 0.002143 | -0.1986 |
| retrieval | moba-1block | 6 | 0.3118 | 0.3962 | 1.153 | 1.112 | 0.001609 | -0.196 |
| retrieval | moba-2block | 6 | 0.1107 | 0.1747 | 0.3396 | 0.3116 | 0.0008233 | -0.0599 |
| retrieval | performer-32 | 6 | 0.5603 | 0.83 | 3.536 | 3.592 | 0.002941 | -0.6107 |
| retrieval | quest-1block | 6 | 0.8128 | 0.8834 | 3.808 | 3.768 | 0.003269 | -0.5202 |
| retrieval | quest-2block | 6 | 0.1377 | 0.1277 | 0.2255 | 0.2224 | 0.0004081 | -0.04948 |
| retrieval | sliding-window | 6 | 0.527 | 0.8109 | 3.276 | 3.211 | 0.00335 | -0.4557 |
| dispersed | kv-fp4 | 6 | 0.06758 | 0.08426 | 0.001357 | 0.00161 | 5.806e-06 | 0 |
| dispersed | kv-fp8 | 6 | 0.01398 | 0.02044 | 4.122e-05 | 0.0001192 | 1.41e-06 | 0 |
| dispersed | mass-oracle-2block | 6 | 0.05668 | 0.01939 | 0.0002917 | 0.0003532 | 1.957e-06 | 0 |
| dispersed | minference-block | 6 | 0.08186 | 0.3589 | 1.436 | 1.443 | 3.484e-05 | -0.2174 |
| dispersed | moba-1block | 6 | 0.1541 | 0.2081 | 0.6643 | 0.6686 | 1.924e-05 | -0.1055 |
| dispersed | moba-2block | 6 | 0.0768 | 0.09809 | 0.2901 | 0.2926 | 1.088e-05 | -0.04427 |
| dispersed | performer-32 | 6 | 0.3486 | 0.8435 | 3.28 | 3.294 | 9.317e-05 | -0.5078 |
| dispersed | quest-1block | 6 | 0.4368 | 1.085 | 6.046 | 6.07 | 0.0001064 | -0.7656 |
| dispersed | quest-2block | 6 | 0.09042 | 0.04805 | 0.02115 | 0.02178 | 5.249e-06 | -0.007161 |
| dispersed | sliding-window | 6 | 0.3375 | 1.066 | 5.972 | 5.995 | 9.786e-05 | -0.7578 |
