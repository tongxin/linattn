# Metric hierarchy validation on representative attention variants

*Generated 2026-10-21T16:06:59Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

T=128, H=4, H_kv=2, d=32, four 16-token blocks, 32-token token budget. Mask-based selectors expose both support and a renormalized attention distribution; Performer/Linformer expose only an output, so distribution cells are blank. This is a metric-validation experiment, not a quality leaderboard. `structured` plants retrieval spikes; `random` is the control.

| variant | input | output_rel_l2 | output_cosine_distance | max_abs_error | reads_per_query | support_fraction | distribution_operator_rel_l2 | distribution_operator_cosine_distance | distribution_attention_tv | distribution_attention_js | distribution_attention_hellinger | distribution_topk_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quest | random | 0.2026 | 0.01969 | 0.4428 | 52.44 | 0.4097 | 0.2026 | 0.01969 | 0.1038 | 0.04066 | 0.1483 | 0.9097 |
| xattention | random | 0.6048 | 0.1427 | 1.289 | 34 | 0.2656 | 0.6048 | 0.1427 | 0.3744 | 0.1658 | 0.421 | 0.6375 |
| moba | random | 0.2281 | 0.02438 | 0.4599 | 47.12 | 0.3682 | 0.2281 | 0.02438 | 0.1349 | 0.05357 | 0.183 | 0.894 |
| minference | random | 0.272 | 0.03473 | 0.6662 | 48.19 | 0.3765 | 0.272 | 0.03473 | 0.151 | 0.06122 | 0.1943 | 0.8521 |
| performer | random | 0.6017 | 0.2013 | 1.756 | 32 |  |  |  |  |  |  |  |
| linformer | random | 1.335 | 0.9381 | 3.041 | 32 |  |  |  |  |  |  |  |
| quest | structured | 0.01608 | 0.0001274 | 0.2669 | 44.75 | 0.3496 | 0.01608 | 0.0001274 | 0.005967 | 0.002197 | 0.01359 | 0.9897 |
| xattention | structured | 0.8922 | 0.3914 | 4.921 | 16 | 0.125 | 0.8922 | 0.3914 | 0.3173 | 0.1842 | 0.3652 | 0.5486 |
| moba | structured | 0.5561 | 0.1577 | 4.471 | 47.88 | 0.374 | 0.5561 | 0.1577 | 0.08348 | 0.05486 | 0.09804 | 0.8777 |
| minference | structured | 0.6049 | 0.1849 | 4.916 | 46.75 | 0.3652 | 0.6049 | 0.1849 | 0.08746 | 0.05883 | 0.1045 | 0.866 |
| performer | structured | 0.9959 | 0.8005 | 3.83 | 32 |  |  |  |  |  |  |  |
| linformer | structured | 1.584 | 0.9714 | 5.333 | 32 |  |  |  |  |  |  |  |
