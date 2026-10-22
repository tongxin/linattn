# Checks of sparse-support bounds, feature convergence and KV precision

*Generated 2026-10-22T01:24:00Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

Synthetic CPU validation across multiple seeds. Selection-bound rows test ||O_ref-O_selected|| <= 2·dropped_mass·Vmax. Feature rows test the expected variance reduction of positive random features; a non-monotone single seed is reported rather than hidden. Precision rows apply a group-wise KV codec before the exact dense attention operator.

| study | input | variant | setting | mean_rel_error | std_rel_error | mean_bound_ratio | bound_satisfied_rate | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature_convergence | dispersed | performer | features=4 | 0.983 | 0.02721 |  |  | 5 |
| feature_convergence | dispersed | performer | features=8 | 0.9536 | 0.05167 |  |  | 5 |
| feature_convergence | dispersed | performer | features=16 | 0.9522 | 0.0484 |  |  | 5 |
| feature_convergence | dispersed | performer | features=32 | 0.9085 | 0.0233 |  |  | 5 |
| feature_convergence | dispersed | performer | features=64 | 0.9348 | 0.04923 |  |  | 5 |
| feature_convergence | dispersed | performer | features=128 | 0.8972 | 0.02826 |  |  | 5 |
| feature_convergence | dispersed | performer | features=256 | 0.8708 | 0.01183 |  |  | 5 |
| feature_convergence | random | performer | features=4 | 0.7746 | 0.04724 |  |  | 5 |
| feature_convergence | random | performer | features=8 | 0.6865 | 0.02119 |  |  | 5 |
| feature_convergence | random | performer | features=16 | 0.6766 | 0.02025 |  |  | 5 |
| feature_convergence | random | performer | features=32 | 0.6219 | 0.01214 |  |  | 5 |
| feature_convergence | random | performer | features=64 | 0.6157 | 0.0099 |  |  | 5 |
| feature_convergence | random | performer | features=128 | 0.5952 | 0.008725 |  |  | 5 |
| feature_convergence | random | performer | features=256 | 0.5863 | 0.01009 |  |  | 5 |
| precision | dispersed | dense_kv | fp4 | 0.1583 | 0.006244 |  |  | 3 |
| precision | dispersed | dense_kv | fp8 | 0.03661 | 0.0005564 |  |  | 3 |
| precision | dispersed | dense_kv | fp16 | 0 | 0 |  |  | 3 |
| precision | random | dense_kv | fp4 | 0.122 | 0.003088 |  |  | 3 |
| precision | random | dense_kv | fp8 | 0.02952 | 0.0001769 |  |  | 3 |
| precision | random | dense_kv | fp16 | 0 | 0 |  |  | 3 |
| selection_bound | dispersed | minference | all | 0.7042 | 0.03803 | 0.2542 | 1 | 3 |
| selection_bound | dispersed | moba | all | 0.5543 | 0.01769 | 0.2532 | 1 | 3 |
| selection_bound | dispersed | quest | all | 0.6233 | 0.008343 | 0.2564 | 1 | 3 |
| selection_bound | dispersed | xattention | all | 0.9692 | 0.01719 | 0.2759 | 1 | 3 |
| selection_bound | random | minference | all | 0.6284 | 0.02706 | 0.1336 | 1 | 3 |
| selection_bound | random | moba | all | 0.559 | 0.007296 | 0.1318 | 1 | 3 |
| selection_bound | random | quest | all | 0.6003 | 0.01145 | 0.1338 | 1 | 3 |
| selection_bound | random | xattention | all | 0.8016 | 0.08291 | 0.1653 | 1 | 3 |
| selection_bound | retrieval | minference | all | 0.691 | 0.07343 | 0.1978 | 1 | 3 |
| selection_bound | retrieval | moba | all | 0.5701 | 0.03462 | 0.1865 | 1 | 3 |
| selection_bound | retrieval | quest | all | 0.6373 | 0.04171 | 0.1847 | 1 | 3 |
| selection_bound | retrieval | xattention | all | 0.8289 | 0.02164 | 0.2321 | 1 | 3 |
