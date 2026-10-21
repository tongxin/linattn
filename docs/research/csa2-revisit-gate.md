# CSA2 revisit gate

CSA2 remains an open problem, not a current conclusion. The framework developed
in this research track is a prerequisite for revisiting it.

## Gate 0 — evidence

- [ ] A runnable V4.1-Flash-style checkpoint is available.
- [ ] A matched dense/reference path or teacher is defined.
- [ ] The official config (40 layers, encoder `m=2`, decoder `m=1`, 512 top-k,
      128 SWA, FP4 main KV, FP8 SWA) is represented explicitly.
- [ ] The test can compare one factor at a time: CED, mode schedule, compression,
      selection, FP4, and SWA replay.

## Gate 1 — local approximation

- [ ] Attention operator/output error is reported against dense or a controlled
      teacher.
- [ ] Support/mass recall and distributional distance are reported where a
      normalized sparse distribution exists.
- [ ] The selected support is verified causal and budget-honest.
- [ ] FP4/FP8 effects are separated from selection/pooling effects.

## Gate 2 — propagation

- [ ] Next-layer and final hidden-state drift are reported.
- [ ] The error-propagation experiment distinguishes attenuation from
      amplification.
- [ ] The result is tested with at least two input distributions and two seeds.

## Gate 3 — workload behavior

- [ ] Retrieval-shaped and dispersed-evidence tasks are both present.
- [ ] Coding/agentic tasks report success and token/tool cost.
- [ ] The evaluation uses a matched harness and context budget.

## Gate 4 — deployment claim

- [ ] KV footprint, actual reads, prefill/decode latency and throughput are
      reported together.
- [ ] The claim states whether the result comes from compression, sparsity,
      precision, CED, or serving.

Until Gates 0–4 are satisfied, CSA2 is documented as a promising cost design, not
as an established approximation to full softmax.
