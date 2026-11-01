# Associative-recall capacity of a fixed-size delta-rule state

*Generated 2026-11-01T13:23:16Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

No training involved: random key/value vectors are written through the exact delta-rule recurrence (keys L2-normalised, beta=0.99, decay=0.99) and read back with the same keys. `retrieval_cosine` is the mean cosine fidelity between the retrieved and the true value; the state holds `d_state * d_value` floats, so the curve is really about *associations per unit of state* -- the quantity a hybrid trades against exact-attention layers. The error decomposition reports whether a miss changes magnitude or value direction; it is descriptive, not a model-level bound.

| associations | d_state | state_floats | retrieval_cosine | retrieval_error_rel_l2 | error_cosine_to_value | parallel_error_fraction | orthogonal_error_fraction | state_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 8 | 512 | 0.8277 | 0.682 | 0.006706 | 0.08663 | 0.9954 | 3.964 |
| 4 | 32 | 2048 | 0.9583 | 0.2815 | -0.1389 | 0.1562 | 0.9764 | 3.888 |
| 4 | 128 | 8192 | 0.9852 | 0.1672 | -0.135 | 0.1657 | 0.9796 | 3.913 |
| 16 | 8 | 512 | 0.5776 | 1.274 | -0.08382 | 0.1143 | 0.9901 | 7.256 |
| 16 | 32 | 2048 | 0.8363 | 0.6112 | -0.1228 | 0.1486 | 0.9833 | 7.388 |
| 16 | 128 | 8192 | 0.9456 | 0.3259 | -0.2761 | 0.2813 | 0.9458 | 7.322 |
| 64 | 8 | 512 | 0.3376 | 2.196 | -0.09715 | 0.1386 | 0.9847 | 12.38 |
| 64 | 32 | 2048 | 0.5708 | 1.091 | -0.2346 | 0.2467 | 0.9578 | 12 |
| 64 | 128 | 8192 | 0.8105 | 0.5926 | -0.4203 | 0.4284 | 0.8759 | 11.99 |
| 256 | 8 | 512 | 0.1419 | 2.545 | -0.2486 | 0.257 | 0.9568 | 13.96 |
| 256 | 32 | 2048 | 0.2665 | 1.4 | -0.4463 | 0.4494 | 0.8735 | 13.95 |
| 256 | 128 | 8192 | 0.4376 | 0.9091 | -0.6775 | 0.6788 | 0.6887 | 13.98 |
| 1024 | 8 | 512 | 0.03476 | 2.642 | -0.3429 | 0.3455 | 0.9292 | 14.09 |
| 1024 | 32 | 2048 | 0.07561 | 1.538 | -0.577 | 0.5783 | 0.8014 | 14.1 |
| 1024 | 128 | 8192 | 0.1223 | 1.102 | -0.8058 | 0.8061 | 0.5662 | 14.02 |
