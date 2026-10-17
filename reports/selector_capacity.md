# Learned-selector capacity: before and after fitting

*Generated 2026-10-17T10:44:45Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

Cross-entropy is between the selector's induced distribution over keys/blocks and the dense attention mass, so `before` is a randomly initialised selector and `after` is one fitted on the same inputs (`steps` gradient steps). The gap between the two is the price of the training stage that DSA, SeerAttention-R and NSA require. For NSA the metric is 1 - relative output error against dense attention, because its gates act on outputs rather than on a distribution.

| selector | cross_entropy_before | cross_entropy_after | mass_recall_before | mass_recall_after | budget_fraction |
| --- | --- | --- | --- | --- | --- |
| dsa_indexer | 5.721 | 5.353 | 1 | 1 | 0.7894 |
| seerattention_gate | 133.1 | 1.385 | 0.7224 | 0.9881 | 0.5234 |
| nsa_branch_gates | nan | nan | 0.2696 | 0.3014 | nan |
