# Associative-recall capacity of a fixed-size delta-rule state

*Generated 2026-10-16T10:32:48Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

No training involved: random key/value vectors are written through the exact delta-rule recurrence (keys L2-normalised, beta=0.99, decay=0.99) and read back with the same keys. `retrieval_cosine` is the mean cosine fidelity between the retrieved and the true value; the state holds `d_state * d_value` floats, so the curve is really about *associations per unit of state* -- the quantity a hybrid trades against exact-attention layers.

| associations | d_state | state_floats | retrieval_cosine | state_norm |
| --- | --- | --- | --- | --- |
| 4 | 8 | 512 | 0.8277 | 3.964 |
| 4 | 32 | 2048 | 0.9583 | 3.888 |
| 4 | 128 | 8192 | 0.9852 | 3.913 |
| 16 | 8 | 512 | 0.5776 | 7.256 |
| 16 | 32 | 2048 | 0.8363 | 7.388 |
| 16 | 128 | 8192 | 0.9456 | 7.322 |
| 64 | 8 | 512 | 0.3376 | 12.38 |
| 64 | 32 | 2048 | 0.5708 | 12 |
| 64 | 128 | 8192 | 0.8105 | 11.99 |
| 256 | 8 | 512 | 0.1419 | 13.96 |
| 256 | 32 | 2048 | 0.2665 | 13.95 |
| 256 | 128 | 8192 | 0.4376 | 13.98 |
| 1024 | 8 | 512 | 0.03476 | 14.09 |
| 1024 | 32 | 2048 | 0.07561 | 14.1 |
| 1024 | 128 | 8192 | 0.1223 | 14.02 |
