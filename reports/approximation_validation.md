# Metric hierarchy validation on representative attention variants

*Generated 2026-10-22T01:45:57Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

T=128, H=4, H_kv=2, d=32, four 16-token blocks, 32-token token budget. Mask-based selectors expose both support and a renormalized attention distribution; Performer/Linformer expose only an output, so distribution cells are blank. Linformer is compared with a noncausal dense reference because its sequence projection mixes positions; other variants use the causal reference. This is a metric-validation experiment, not a quality leaderboard. `structured` plants retrieval spikes; `random` is the control.

| variant | input | reference_mode | n_seeds | output_rel_l2_mean | output_rel_l2_std | output_cosine_distance_mean | max_abs_error_mean | reads_per_query_mean | support_fraction_mean | attention_js_mean | topk_agreement_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linformer | random | noncausal | 3 | 2.127 | 0.01656 | 0.847 | 1.663 | 32 | None | None | None |
| linformer | structured | noncausal | 3 | 1.314 | 0.008629 | 0.9128 | 4.915 | 32 | None | None | None |
| minference | random | causal | 3 | 0.2988 | 0.00712 | 0.04178 | 0.88 | 47.31 | 0.3696 | 0.06573 | 0.8437 |
| minference | structured | causal | 3 | 0.5484 | 0.04306 | 0.1522 | 4.946 | 47.83 | 0.3737 | 0.07544 | 0.8521 |
| moba | random | causal | 3 | 0.2482 | 0.006583 | 0.02872 | 0.6075 | 47.06 | 0.3677 | 0.05407 | 0.8906 |
| moba | structured | causal | 3 | 0.5288 | 0.01942 | 0.141 | 4.529 | 47.46 | 0.3708 | 0.06788 | 0.8687 |
| performer | random | causal | 3 | 0.6197 | 0.01299 | 0.2152 | 1.527 | 32 | None | None | None |
| performer | structured | causal | 3 | 1.001 | 0.00651 | 0.8145 | 3.534 | 32 | None | None | None |
| quest | random | causal | 3 | 0.1948 | 0.008547 | 0.01826 | 0.5771 | 53.4 | 0.4172 | 0.03657 | 0.9213 |
| quest | structured | causal | 3 | 0.06798 | 0.05998 | 0.004093 | 1.421 | 45.15 | 0.3527 | 0.005317 | 0.9771 |
| xattention | random | causal | 3 | 0.8016 | 0.08291 | 0.2275 | 2.388 | 30.92 | 0.2415 | 0.1931 | 0.6051 |
| xattention | structured | causal | 3 | 0.9842 | 0.05032 | 0.4625 | 4.979 | 14.54 | 0.1136 | 0.2633 | 0.4889 |
