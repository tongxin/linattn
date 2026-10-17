# MQAR accuracy vs. persistent state size for hybrid stacks

*Generated 2026-10-17T10:47:15Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

MQAR with 4 pairs per sequence, seq_len=128, 6 layers, d_model=96, 250 training steps, one seed. `state_bytes_per_seq` counts the persistent state only (KV for attention layers is counted per token, the linear state per sequence), which is why a hybrid's advantage grows with sequence length.

**Read this table together with `state_capacity.md`.** At the scale a laptop can train (a few hundred thousand parameters, tens of pairs), every architecture here solves MQAR, so the training comparison does *not* discriminate between families -- which is itself worth knowing, because MQAR is routinely cited as evidence that it does. The capacity limit shows up without training instead: see the deterministic `experiments/state_capacity.py` curve, where recall of a fixed-size state degrades as the number of associations grows. Getting MQAR to *separate* attention from linear attention needs either scale we cannot reach here or a much harder task (many more pairs, shorter training, smaller state); we prefer to say that than to report a curve that looks like a result.

| model | accuracy | state_bytes_per_seq | attention_layers | linear_layers | kind |
| --- | --- | --- | --- | --- | --- |
| dense_attention_only | 1 | 4608 | 6 | 0 | gdn |
| hybrid_1in2_gdn | 1 | 3.917e+04 | 3 | 3 | gdn |
| hybrid_1in4_gdn | 1 | 5.069e+04 | 2 | 4 | gdn |
| hybrid_1in4_kda | 1 | 5.069e+04 | 2 | 4 | kda |
| hybrid_1in4_gdn2 | 1 | 5.069e+04 | 2 | 4 | gdn2 |
| hybrid_1in4_ssd | 1 | 5.069e+04 | 2 | 4 | ssd |
| linear_only_gdn | 1 | 7.373e+04 | 0 | 6 | gdn |
