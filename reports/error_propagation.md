# Local attention error versus downstream residual-stream drift

*Generated 2026-10-21T16:06:59Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

Six-layer fixed random residual stack; a Quest attention output replaces dense attention at one layer, then dense dynamics continue. This isolates propagation, not trained-model quality. Amplification is final/local relative L2 error. Random and planted-retrieval inputs are controls.

| input | inject_layer | local_rel_l2 | final_rel_l2 | amplification | local_cosine_distance | final_cosine_distance |
| --- | --- | --- | --- | --- | --- | --- |
| random | 0 | 0.4364 | 0.5523 | 1.266 | 0.09806 | 0.1664 |
| random | 1 | 0.372 | 0.5351 | 1.438 | 0.06717 | 0.1552 |
| random | 2 | 0.4103 | 0.5719 | 1.394 | 0.0817 | 0.1794 |
| random | 3 | 0.5111 | 0.5452 | 1.067 | 0.1153 | 0.1194 |
| random | 4 | 0.5516 | 0.5373 | 0.9741 | 0.0669 | 0.05138 |
| random | 5 | 0.5374 | 0.5374 | 1 | 0.05103 | 0.05103 |
| structured | 0 | 0.7154 | 0.5121 | 0.7158 | 0.3012 | 0.04272 |
| structured | 1 | 0.5783 | 0.5121 | 0.8855 | 0.1368 | 0.04271 |
| structured | 2 | 0.5462 | 0.512 | 0.9375 | 0.08838 | 0.0427 |
| structured | 3 | 0.5307 | 0.512 | 0.9648 | 0.06294 | 0.0427 |
| structured | 4 | 0.5188 | 0.512 | 0.987 | 0.04959 | 0.0427 |
| structured | 5 | 0.512 | 0.512 | 1 | 0.0427 | 0.0427 |
