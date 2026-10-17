# CSA2 end to end: footprint, sharing cost and the attention term it leaves alone

*Generated 2026-10-17T13:17:08Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

Footprint: reported 890 B/token (V4.1-Flash) vs a composition of a 576-value MLA latent at FP4 with 4-layer sharing, and the sharing group needed to approach the report. Sharing: output divergence between a stack with cross-layer reuse and the same stack without it, as the per-layer KV correlation (sigma) varies -- sigma=0 means layers agree exactly. It is an architectural property, not something reuse can create post hoc. Cost: `divergence` column holds attention ms for the cost rows; the point is that the storage win does not move the attention term.

| experiment | config | bytes_per_token | kib_per_token | sharing_group | bits | divergence | detail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| footprint | csa2_reported | 890 | 0.8691 | 0 | 4 | 0 | DeepSeek-V4.1-Flash reported global KV: 890 B/token across ~61 layers |
| footprint | csa2 | 5184 | 5.062 | 4 | 4 | 0 | 576 values/token, fp4, shared over 4 layers |
| footprint | csa2_reach890 | 972 | 0.9492 | 21 | 4 | 0 | 576 values/token, fp4, shared over 21 layers |
| sharing | sigma=0 | 9 | 0.008789 | 4 | 4 | 0 | 4-layer groups, fp4, pool x32 |
| sharing | sigma=0.05 | 9 | 0.008789 | 4 | 4 | 0.06773 | 4-layer groups, fp4, pool x32 |
| sharing | sigma=0.2 | 9 | 0.008789 | 4 | 4 | 0.2672 | 4-layer groups, fp4, pool x32 |
| sharing | sigma=1 | 9 | 0.008789 | 4 | 4 | 1.002 | 4-layer groups, fp4, pool x32 |
| cost | dense_fp16 @ 32768 | 2.499e+05 | 244 | 0 | 0 | 677.5 | mlp |
| cost | dense_fp16 @ 131072 | 2.499e+05 | 244 | 0 | 0 | 9214 | attn |
| cost | dense_fp16 @ 1048576 | 2.499e+05 | 244 | 0 | 0 | 5.593e+05 | attn |
| cost | mla @ 32768 | 7.027e+04 | 68.62 | 0 | 0 | 677.5 | mlp |
| cost | mla @ 131072 | 7.027e+04 | 68.62 | 0 | 0 | 9214 | attn |
| cost | mla @ 1048576 | 7.027e+04 | 68.62 | 0 | 0 | 5.593e+05 | attn |
| cost | csa2 @ 32768 | 5184 | 5.062 | 0 | 0 | 677.5 | mlp |
| cost | csa2 @ 131072 | 5184 | 5.062 | 0 | 0 | 9214 | attn |
| cost | csa2 @ 1048576 | 5184 | 5.062 | 0 | 0 | 5.593e+05 | attn |
