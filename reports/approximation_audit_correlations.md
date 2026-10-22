# Matched-checkpoint audit: metric rank correlations

*Generated 2026-10-22T01:38:38Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

Spearman correlations over one-layer interventions. A high local-error correlation is evidence about this synthetic setup, not a general guarantee.

| task | n | local_vs_loss_spearman | hidden_vs_loss_spearman | projected_vs_loss_spearman | local_vs_projected_spearman |
| --- | --- | --- | --- | --- | --- |
| retrieval | 60 | 0.878 | 0.9662 | 0.6859 | 0.6041 |
| dispersed | 60 | 0.8481 | 0.9445 | 0.8995 | 0.7223 |
