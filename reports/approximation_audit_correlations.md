# Matched-checkpoint audit: metric rank correlations

*Generated 2026-10-22T02:03:44Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

Spearman correlations over one-layer interventions. A high local-error correlation is evidence about this synthetic setup, not a general guarantee.

| task | n | local_vs_loss_spearman | hidden_vs_loss_spearman | projected_vs_loss_spearman | local_vs_projected_spearman | mean_seed_local_vs_loss | mean_seed_hidden_vs_loss | mean_seed_projected_vs_loss | std_seed_local_vs_loss | std_seed_hidden_vs_loss | std_seed_projected_vs_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retrieval | 60 | 0.9142 | 0.9657 | 0.582 | 0.4693 | 0.9273 | 0.9744 | 0.8526 | 0.02917 | 0.005627 | 0.07194 |
| dispersed | 60 | 0.8691 | 0.9234 | 0.7228 | 0.5537 | 0.8977 | 0.9343 | 0.9599 | 0.004911 | 0.01347 | 0.01602 |
