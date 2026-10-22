# Matched-checkpoint audit: per-checkpoint rank correlations

*Generated 2026-10-22T02:03:44Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

Each row uses 20 interventions at two layers for one trained checkpoint. These are descriptive correlations, not independent confidence intervals.

| task | seed | n | local_vs_loss_spearman | hidden_vs_loss_spearman | projected_vs_loss_spearman |
| --- | --- | --- | --- | --- | --- |
| retrieval | 0 | 20 | 0.9684 | 0.9669 | 0.9293 |
| retrieval | 1 | 20 | 0.9038 | 0.9759 | 0.8722 |
| retrieval | 2 | 20 | 0.9098 | 0.9805 | 0.7564 |
| dispersed | 0 | 20 | 0.9038 | 0.9534 | 0.9805 |
| dispersed | 1 | 20 | 0.8917 | 0.9248 | 0.9579 |
| dispersed | 2 | 20 | 0.8977 | 0.9248 | 0.9414 |
