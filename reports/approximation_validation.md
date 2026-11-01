# Metric hierarchy validation on representative attention variants

*Generated 2026-11-01T13:24:30Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

T=128, H=4, H_kv=2, d=32, four 16-token blocks, 32-token token budget. Mask-based selectors expose both support and a renormalized attention distribution; Performer/Linformer expose only an output, so distribution cells are blank. Linformer is compared with a noncausal dense reference because its sequence projection mixes positions; other variants use the causal reference. This is a metric-validation experiment, not a quality leaderboard. `structured` plants retrieval spikes; `random` is the control.

| variant | input | reference_mode | n_seeds | output_rel_l2_mean | output_rel_l2_std | output_cosine_distance_mean | max_abs_error_mean | parallel_error_fraction_mean | orthogonal_error_fraction_mean | reads_per_query_mean | support_fraction_mean | attention_js_mean | error_cosine_to_reference_mean | error_cosine_to_estimate_mean | topk_agreement_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linformer | random | noncausal | 3 | 2.127 | 0.01656 | 0.847 | 1.663 | 0.3212 | 0.9444 | 32 | None | None | None | None | None |
| linformer | structured | noncausal | 3 | 1.314 | 0.008629 | 0.9128 | 4.915 | 0.6983 | 0.7156 | 32 | None | None | None | None | None |
| minference | random | causal | 3 | 0.2988 | 0.00712 | 0.04178 | 0.88 | 0.01486 | 0.9999 | 47.31 | 0.3696 | 0.06573 | -0.003445 | -0.2892 | 0.8437 |
| minference | structured | causal | 3 | 0.5484 | 0.04306 | 0.1522 | 4.946 | 0.2873 | 0.9577 | 47.83 | 0.3737 | 0.07544 | 0.2873 | -0.2628 | 0.8521 |
| moba | random | causal | 3 | 0.2482 | 0.006583 | 0.02872 | 0.6075 | 0.04949 | 0.9986 | 47.06 | 0.3677 | 0.05407 | -0.04949 | -0.2856 | 0.8906 |
| moba | structured | causal | 3 | 0.5288 | 0.01942 | 0.141 | 4.529 | 0.2779 | 0.9604 | 47.46 | 0.3708 | 0.06788 | 0.2779 | -0.2527 | 0.8687 |
| performer | random | causal | 3 | 0.6197 | 0.01299 | 0.2152 | 1.527 | 0.6358 | 0.7716 | 32 | None | None | None | None | None |
| performer | structured | causal | 3 | 1.001 | 0.00651 | 0.8145 | 3.534 | 0.929 | 0.3689 | 32 | None | None | None | None | None |
| quest | random | causal | 3 | 0.1948 | 0.008547 | 0.01826 | 0.5771 | 0.02992 | 0.9994 | 53.4 | 0.4172 | 0.03657 | -0.02992 | -0.2193 | 0.9213 |
| quest | structured | causal | 3 | 0.06798 | 0.05998 | 0.004093 | 1.421 | 0.07123 | 0.9974 | 45.15 | 0.3527 | 0.005317 | -0.03362 | -0.1013 | 0.9771 |
| xattention | random | causal | 3 | 0.8016 | 0.08291 | 0.2275 | 2.388 | 0.03346 | 0.9994 | 30.92 | 0.2415 | 0.1931 | 0.03346 | -0.6065 | 0.6051 |
| xattention | structured | causal | 3 | 0.9842 | 0.05032 | 0.4625 | 4.979 | 0.4441 | 0.8955 | 14.54 | 0.1136 | 0.2633 | 0.4441 | -0.5156 | 0.4889 |
