# linattn

**A survey of sparse, compressed-state and co-designed attention -- plus the
reference implementations, kernels and validation harness that back its claims.**

The 2025-2026 attention literature is large, repetitive in its claims and
inconsistent in its measurements. This repository does three things about that:

1. **[docs/survey.md](docs/survey.md)** -- a survey and gap analysis of attention
   variants that exploit KV sparsity, compressed state and hardware-software
   co-design, with the evidence organised by workload (coding, long-horizon
   agents, reasoning, deep research) rather than by paper.
2. **Implementations of 16 registered variants** under one interface -- plus the
   three compressed-state mixers (SSD / Gated DeltaNet / KDA) and a hybrid stack --
   so that a selector, an eviction policy, a native sparse architecture and a
   linear-attention layer can be compared on the same inputs, at the same budget,
   with the same metrics.
3. **Validation that is the point of the exercise**: every variant is checked
   against dense attention at an unrestricted budget, every cache implementation
   against its own streaming/decode path, and every chunked linear-attention
   algorithm against its sequential recurrence -- in float64 *and* in gradients.

## Quick start

```bash
uv venv --python 3.13 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

python -m linattn.cli backends                       # what can this machine run?
python -m linattn.cli variants                       # the catalogue, with citations
python -m linattn.cli cost --seq 131072              # analytic cost model
python -m linattn.cli select --variant quest --seq 4096 --structured

python -m pytest tests -q                            # correctness gates
python bench/run_suite.py --seq 2048 --budget 256     # quality/cost sweep -> reports/
python bench/bench_decode.py --seq 65536              # bytes-read scaling (C kernel)
python experiments/hybrid_recall.py --steps 300       # MQAR vs. state size
python experiments/state_capacity.py                 # exact recall vs state size
python scripts/gen_variant_docs.py                    # regenerate docs/variants/
```

Everything runs on CPU. The Triton kernels are imported lazily, validated by a
parity harness, and skipped with an explicit reason when there is no CUDA GPU.

## The variants

| family | variants | what they share |
| --- | --- | --- |
| dense (oracle) | `dense` | the reference every other number is measured against |
| structural / local | `sliding_window`, `streaming_sink` | window + attention sinks, the always-safe configuration |
| KV eviction | `h2o`, `snapkv`, `pyramidkv`, `random_eviction` | permanently drop tokens; `random_eviction` is the control that keeps the family honest |
| query-aware selection | `quest`, `minference`, `flexprefill`, `xattention`, `moba`, `dsa`, `seerattention_r` | keep the cache, choose which blocks/tokens to read |
| native sparse | `nsa` | compressed + selected + window branches with learned gates |
| compressed state | `mla`, `ssd`/`gdn`/`kda` layers, `hybrid` | fixed-size state instead of a growing cache |

Per-variant pages (mechanism, usage, cost table, caveats, citations) live in
[`docs/variants/`](docs/variants/index.md) and are generated from the code.

## What the survey concludes, in six lines

1. **Eviction scores barely matter.** Keeping the prompt and the recent window is
   what carries quality; on reasoning workloads uniform random eviction matches
   tuned selectors (arXiv:2609.03430).
2. **Block/page selection won because kernels can execute it**, and query-aware
   bounds (Quest), antidiagonal sums (XAttention) and pooled gates (MoBA) are all
   approximations of one quantity: block attention mass (arXiv:2607.09052).
3. **Production converged on a learned indexer in front of a sparse core**
   (DeepSeek DSA/CSA), and the 2026 bottleneck is the *indexer's* O(T²) scoring,
   not the attention it gates.
4. **Compressed state is now a hybrid architecture**, not a compromise: KDA at a
   3:1 ratio beats full MLA at matched tokens with 75% less KV cache
   (arXiv:2510.26692), and V4.1-Flash combines compression with FP4 and
   cross-layer reuse to reach 890 bytes/token (arXiv:2609.19969).
5. **The quality gap between training-free and native sparsity is systematic**:
   training-free methods report parity, native ones report gains, and the
   difference shows up worst on multi-hop and reasoning tasks.
6. **The frontier labs that publish mechanisms are the open-weight ones.** For
   closed models, only pricing is observable, and it is consistent with dense
   attention + cache reuse rather than native sparsity
   ([frontier-gap.md](docs/frontier-gap.md)).

## Validation: what is actually checked

The suite is organised so that a number in `reports/` cannot be produced by a
broken implementation:

| question | test |
| --- | --- |
| is the oracle right? | `tests/test_reference.py` (manual softmax, causal offsets, GQA expansion) |
| does every variant honour its budget? | `tests/test_budgets.py` (reads per query, cache size, recent-window retention) |
| do selectors reduce to dense at an unrestricted budget? | `tests/test_budgets.py::test_full_budget_recovers_dense` |
| is selection causal and non-empty? | `tests/test_selection.py` |
| do the learned selectors have the capacity to work at all? | `tests/test_selection.py`, `experiments/selector_capacity.py` |
| does the chunked state algorithm equal the recurrence? | `tests/test_recurrent.py` (outputs, states, *and gradients*, in float64) |
| do the accelerated paths agree with the references? | `tests/test_kernels.py` (C kernel everywhere; Triton under `-m gpu`) |
| are the docs in sync with the code? | `tests/test_docs.py` |

## Repository layout

```
docs/                survey, evidence, gap analysis, kernels, per-variant pages, citations
src/linattn/         core contract, variants, kernels (triton/ flex/ c/), harness, metrics
tests/               the validation suite described above
bench/               suite and decode benchmarks -> reports/
experiments/         hybrid recall (MQAR) and learned-selector capacity studies
scripts/             doc generation
reports/             generated numbers (JSON + markdown), with the environment recorded
```

## Citing the variants

Each implementation carries its own citation in `paper = (...)` and each doc page
repeats it with a link. The machine-readable list, including the discussion
threads used for the practitioner-side claims, is in
[`docs/citations.bib`](docs/citations.bib); the raw survey corpus (paper metadata
plus the public discussions) is in
[`docs/research/corpus.jsonl`](docs/research/corpus.jsonl).

## Limitations, stated up front

* This repository implements **semantics**, not production kernels. The
  selection masks are explicit boolean masks, which is what makes them auditable,
  and also what makes prefill cost `O(T²)` in the reference path. The accelerated
  paths and the compiler path (FlexAttention block masks) are where that is
  addressed.
* The Triton kernels were written against stable Triton APIs but could not be
  executed in this environment (no NVIDIA GPU). They are covered by parity tests
  that run when a GPU is present, and the C kernel -- which does run here -- is
  validated against torch.
* The hybrid/recall experiment is small-scale (a few million parameters, MQAR) by
  design: it demonstrates the *shape* of the recall/state trade-off that the
  literature reports at scale, not the numbers themselves.
* Every claim about frontier models is either quoted from a released report or
  explicitly labelled as inference from public pricing in
  [frontier-gap.md](docs/frontier-gap.md).
