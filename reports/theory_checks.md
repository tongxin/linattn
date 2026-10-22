# Checks of attention error identities, bounds, feature convergence and KV precision

*Generated 2026-10-22T01:50:04Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

Synthetic CPU validation across multiple seeds. Selection rows check the exact identity O_ref-O_selected=(1-a)(mu_omitted-mu_selected), selector implementation parity, and the conservative bound ||error||<=2(1-a)Vmax. Precision rows split total error into key-induced and value-induced terms and check TV(softmax(z),softmax(z+delta))<=tanh(max|delta|). Feature rows test the expected variance reduction of positive random features. The bounds are local to one attention operation, not model-level guarantees.

| study | input | variant | setting | mean_rel_error | std_rel_error | mean_bound_ratio | bound_satisfied_rate | mean_attention_tv | max_attention_tv_bound | max_certificate_ratio | certificate_satisfied_rate | max_exact_decomposition_error | max_implementation_parity_error | mean_error_per_dropped_mass | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature_convergence | dispersed | performer | features=4 | 0.983 | 0.02721 |  |  |  |  |  |  |  |  |  | 5 |
| feature_convergence | dispersed | performer | features=8 | 0.9536 | 0.05167 |  |  |  |  |  |  |  |  |  | 5 |
| feature_convergence | dispersed | performer | features=16 | 0.9522 | 0.0484 |  |  |  |  |  |  |  |  |  | 5 |
| feature_convergence | dispersed | performer | features=32 | 0.9085 | 0.0233 |  |  |  |  |  |  |  |  |  | 5 |
| feature_convergence | dispersed | performer | features=64 | 0.9348 | 0.04923 |  |  |  |  |  |  |  |  |  | 5 |
| feature_convergence | dispersed | performer | features=128 | 0.8972 | 0.02826 |  |  |  |  |  |  |  |  |  | 5 |
| feature_convergence | dispersed | performer | features=256 | 0.8708 | 0.01183 |  |  |  |  |  |  |  |  |  | 5 |
| feature_convergence | random | performer | features=4 | 0.7746 | 0.04724 |  |  |  |  |  |  |  |  |  | 5 |
| feature_convergence | random | performer | features=8 | 0.6865 | 0.02119 |  |  |  |  |  |  |  |  |  | 5 |
| feature_convergence | random | performer | features=16 | 0.6766 | 0.02025 |  |  |  |  |  |  |  |  |  | 5 |
| feature_convergence | random | performer | features=32 | 0.6219 | 0.01214 |  |  |  |  |  |  |  |  |  | 5 |
| feature_convergence | random | performer | features=64 | 0.6157 | 0.0099 |  |  |  |  |  |  |  |  |  | 5 |
| feature_convergence | random | performer | features=128 | 0.5952 | 0.008725 |  |  |  |  |  |  |  |  |  | 5 |
| feature_convergence | random | performer | features=256 | 0.5863 | 0.01009 |  |  |  |  |  |  |  |  |  | 5 |
| precision | dispersed | dense_kv | fp4 | 0.1583 | 0.006244 |  |  | 0.06685 | 0.5104 | 0.04533 | 1 | 3.725e-08 |  |  | 3 |
| precision | dispersed | dense_kv | fp8 | 0.03661 | 0.0005564 |  |  | 0.01591 | 0.1377 | 0.04159 | 1 | 7.451e-09 |  |  | 3 |
| precision | dispersed | dense_kv | fp16 | 0 | 0 |  |  | 0 | 0 | 0 | 1 | 0 |  |  | 3 |
| precision | random | dense_kv | fp4 | 0.122 | 0.003088 |  |  | 0.03746 | 0.2719 | 0.02186 | 1 | 5.122e-09 |  |  | 3 |
| precision | random | dense_kv | fp8 | 0.02952 | 0.0001769 |  |  | 0.008916 | 0.06712 | 0.0218 | 1 | 1.863e-09 |  |  | 3 |
| precision | random | dense_kv | fp16 | 0 | 0 |  |  | 0 | 0 | 0 | 1 | 0 |  |  | 3 |
| selection_bound | dispersed | minference | all | 0.7042 | 0.03803 | 0.2383 | 1 |  |  |  |  | 1.332e-15 | 2.665e-15 | 1845 | 3 |
| selection_bound | dispersed | moba | all | 0.5543 | 0.01769 | 0.2359 | 1 |  |  |  |  | 1.332e-15 | 2.22e-15 | 1828 | 3 |
| selection_bound | dispersed | quest | all | 0.6233 | 0.008343 | 0.2394 | 1 |  |  |  |  | 9.992e-16 | 1.776e-15 | 1853 | 3 |
| selection_bound | dispersed | xattention | all | 0.9692 | 0.01719 | 0.2606 | 1 |  |  |  |  | 1.082e-15 | 1.776e-15 | 2018 | 3 |
| selection_bound | random | minference | all | 0.6284 | 0.02706 | 0.132 | 1 |  |  |  |  | 4.441e-16 | 8.882e-16 | 1022 | 3 |
| selection_bound | random | moba | all | 0.559 | 0.007296 | 0.1307 | 1 |  |  |  |  | 4.441e-16 | 1.776e-15 | 1012 | 3 |
| selection_bound | random | quest | all | 0.6003 | 0.01145 | 0.1316 | 1 |  |  |  |  | 5.551e-16 | 1.11e-15 | 1019 | 3 |
| selection_bound | random | xattention | all | 0.8016 | 0.08291 | 0.1538 | 1 |  |  |  |  | 4.441e-16 | 8.882e-16 | 1190 | 3 |
| selection_bound | retrieval | minference | all | 0.691 | 0.07343 | 0.1743 | 1 |  |  |  |  | 1.138e-15 | 1.332e-15 | 1348 | 3 |
| selection_bound | retrieval | moba | all | 0.5701 | 0.03462 | 0.1662 | 1 |  |  |  |  | 1.138e-15 | 2.665e-15 | 1286 | 3 |
| selection_bound | retrieval | quest | all | 0.6373 | 0.04171 | 0.1723 | 1 |  |  |  |  | 1.235e-15 | 1.776e-15 | 1334 | 3 |
| selection_bound | retrieval | xattention | all | 0.8289 | 0.02164 | 0.1996 | 1 |  |  |  |  | 1.332e-15 | 1.776e-15 | 1544 | 3 |
