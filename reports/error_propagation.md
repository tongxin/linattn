# Local attention error versus downstream residual-stream drift

*Generated 2026-10-22T01:22:40Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

Six-layer fixed random residual stack; a Quest attention output replaces dense attention at one layer, then dense dynamics continue. This isolates propagation, not trained-model quality. Amplification is final/local relative L2 error. Random and planted-retrieval inputs are controls.

| input | inject_layer | n_seeds | local_rel_l2 | local_rel_l2_std | final_rel_l2 | final_rel_l2_std | amplification | amplification_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0 | 3 | 0.4389 | 0.01021 | 0.5295 | 0.01634 | 1.207 | 0.04666 |
| random | 1 | 3 | 0.3627 | 0.01594 | 0.5122 | 0.04608 | 1.409 | 0.06723 |
| random | 2 | 3 | 0.4002 | 0.01207 | 0.4768 | 0.07952 | 1.187 | 0.1671 |
| random | 3 | 3 | 0.4904 | 0.01468 | 0.5152 | 0.02317 | 1.05 | 0.02434 |
| random | 4 | 3 | 0.5424 | 0.006753 | 0.5378 | 0.006475 | 0.9918 | 0.01532 |
| random | 5 | 3 | 0.5379 | 0.004676 | 0.5379 | 0.004676 | 1 | 0 |
| structured | 0 | 3 | 0.7067 | 0.006564 | 0.5257 | 0.009858 | 0.7441 | 0.02073 |
| structured | 1 | 3 | 0.5757 | 0.003737 | 0.5251 | 0.009309 | 0.9122 | 0.01909 |
| structured | 2 | 3 | 0.5443 | 0.006587 | 0.5251 | 0.009314 | 0.9648 | 0.02131 |
| structured | 3 | 3 | 0.5312 | 0.005766 | 0.5251 | 0.009315 | 0.9886 | 0.01868 |
| structured | 4 | 3 | 0.5298 | 0.00922 | 0.5251 | 0.009315 | 0.991 | 0.00709 |
| structured | 5 | 3 | 0.5251 | 0.009317 | 0.5251 | 0.009317 | 1 | 0 |
