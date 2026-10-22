# linattn

**A survey of sparse, compressed-state and co-designed attention -- plus the
reference implementations, kernels and validation harness that back its claims.**

The 2025-2026 attention literature is large, repetitive in its claims and
inconsistent in its measurements. This repository does three things about that:

1. **[docs/survey.md](docs/survey.md)** -- a survey and gap analysis of attention
   variants that exploit KV sparsity, compressed state and hardware-software
   co-design, with the evidence organised by workload (coding, long-horizon
   agents, reasoning, deep research) rather than by paper.
2. **Implementations of 34 registered attention variants** under one interface --
   plus seven compressed-state mixers (SSD / GDN / KDA / GDN-2 / Q-Delta / KDN /
   SwiLA) and a hybrid stack --
   so that a selector, an eviction policy, a native sparse architecture and a
   linear-attention layer can be compared on the same inputs, at the same budget,
   with the same metrics.
3. **Validation that is the point of the exercise**: every variant is checked
   against dense attention at an unrestricted budget, every cache implementation
   against its own streaming/decode path, and every chunked linear-attention
   algorithm against its sequential recurrence -- in float64 *and* in gradients.

## Quick start

```bash
make install                                  # creates .venv and installs the package + pytest
source .venv/bin/activate

python -m linattn.cli backends                       # what can this machine run?
python -m linattn.cli variants                       # the catalogue, with citations
python -m linattn.cli cost --seq 131072              # analytic cost model
python -m linattn.cli select --variant quest --seq 4096 --structured

make check                                            # correctness gates, seconds
make bench                                            # quality/cost + decode-IO -> reports/
make bench-frontier                                   # 2025–2026 variants -> reports/frontier_2026.*
make csa2-analysis                                    # CSA2 error decomposition -> reports/csa2_approximation.*
make csa2-official-check                              # published config comparison -> reports/csa2_official_config.*
make approx-validation                                # metric hierarchy across variants -> reports/approximation_validation.*
make theory-checks                                    # bound and trend validation -> reports/theory_checks.*
make error-propagation                                # local vs downstream error -> reports/error_propagation.*
make experiments                                      # the three studies -> reports/
```

Everything runs on CPU. The Triton kernels are imported lazily, validated by a
parity harness, and skipped with an explicit reason when there is no CUDA GPU.

The cost command's `prefill attn FLOPs @T` column is a matmul-only analytic
estimate: one multiply-add is counted as two FLOPs, and the current dense base
formula counts the full `T × T` matrix while some variants use the causal triangle.
The exact formulas, exclusions and worked example are in
[docs/analytic-flops.md](docs/analytic-flops.md).

On Apple Silicon there is a **Metal backend** ([docs/mlx.md](docs/mlx.md)):
MLX-native primitives, a hand-written Metal paged-decode kernel, MLX
re-implementations of dense/sliding/Quest/CSA, and an adapter that runs them through
the same harness (`pytest tests/test_mlx.py`, `make bench-mlx`).

To keep the `@triton.jit` source instead of porting to MLX, the third-party
**`triton-msl`** compiler targets Metal through Triton's out-of-tree backend
interface — alpha, pinned to Triton 3.7.0, and per-kernel support is
source-dependent. Status, install path and caveats:
[docs/triton-on-apple.md](docs/triton-on-apple.md).

### KV storage layer

Attention decides how many entries to read; the KV layer decides **which bytes are in
which tier and what a miss costs**. That layer is modelled and measured here:

- `src/linattn/kv/` -- KV formats in bytes/token (dense fp16/fp8, MLA, CSA2, fixed
  recurrent state), a block prefix cache with LRU / TTL-LRU / FIFO / random and a
  Belady oracle, an agentic trace generator, and a cost model that reports a
  breakdown and names the dominant term;
- `python -m linattn.cli kv-cost --prompt 131072 --hit-rate 0.8 --format mla` prints
  that breakdown (`--hit-rate` is explicit, as is `--kv-read-fraction` for sparse
  variants);
- `experiments/kv_economics.py` -> `reports/kv_economics.md`: format x context x hit
  rate, algorithm leverage (dense / 10% / 2% sparse / recurrent), and a policy sweep
  on an agentic trace;
- `experiments/csa2_model.py` -> `reports/csa2_model.md`: CSA2 end to end --
  footprint (reported 890 B/token vs the composed pieces), the divergence cross-layer
  sharing introduces as layers disagree, and the attention term it leaves untouched.
- `experiments/csa2_approximation.py` -> `reports/csa2_approximation.md`: controlled
  dense-softmax error decomposition for selection, pooling and FP4/FP8 quantisation.
  See [docs/csa2-approximation.md](docs/csa2-approximation.md) for the theory,
  official model-level evidence and public-discussion limits.

The headline: compression (MLA, CSA2) cuts footprint by orders of magnitude and leaves
attention time unchanged; sparsity cuts attention until the MLP becomes the wall; a
KDA state removes the quadratic term entirely. Details: [docs/kv-layer.md](docs/kv-layer.md).
Deferred work (ragged batching, batch invariance as a contract) is in
[docs/roadmap.md](docs/roadmap.md).

### The academic line

The catalogue is not only what ships. Twenty academic/analysis attention variants and
seven state mixers are implemented alongside the production designs. The selection is
now **2025–2026-first**: SAGA, PISA, JSSA, MASA, OVAL, SFA, DropKV, ReST-KV and
BeaconKV sit next to recent linear-attention work (Q-Delta, KDN, SwiLA, GDN-2), with
the 2019–2021 canon retained as lineage. Every page states what is faithful, what is
approximated, and which kernel is not reproduced. See
[docs/academic-attention.md](docs/academic-attention.md), which also records the
tests -- including one classic that turned out to be numerically unusable. The
focused comparison is in
[reports/frontier_2026.md](reports/frontier_2026.md); its deliberately
unflattering small-scale result is part of the evidence, not a leaderboard.

The most important correction in this pass: the earlier survey said token eviction
was "retired". That was too strong. Attention-mass eviction is a weak default, but
output-aware eviction (DropKV, ReST-KV, BeaconKV) is an active 2026 line. The repo now
carries both the old lineage and the new scores, plus the random-eviction control, so
the claim can be measured rather than asserted.

A separate theory track now defines the objects and metrics for comparing efficient
attention with full softmax: [the research plan](docs/research/approximation-research-plan.md)
and [metric hierarchy](docs/research/approximation-metrics.md), with a
[theory map](docs/research/approximation-theory-map.md),
[workload protocol](docs/research/attention-evaluation-protocol.md), and
[CSA2 revisit gate](docs/research/csa2-revisit-gate.md). The complete research
report is [attention-approximation-report.md](docs/research/attention-approximation-report.md).
CSA2 sensitivity remains
an explicit open problem until that framework reaches checkpoint-level evaluation.

## Running the benchmarks after cloning

The only hard dependency is **PyTorch**; there is nothing to compile and no GPU or
network access needed at run time. `reports/` in this repository was produced on a
CPU-only laptop, so anything you generate is directly comparable in kind.

```bash
git clone https://github.com/tongxin/linattn && cd linattn

make install          # uv venv --python 3.13 .venv  +  uv pip install -e ".[dev]"
make check            #   ~5 s   validation suite, writes nothing
make bench            #   ~10 s  reports/suite_<T>_<budget>.md + reports/decode_<T>_<dim>.md
make experiments      #  ~3 min  reports/{selector_capacity,state_capacity,hybrid_recall}.md
make all              # docs + check + bench + experiments
```

`make` uses `.venv/bin/python` when it exists and falls back to `python3`, so the
same targets work with a conda/global install (`make PYTHON=python3 bench`). To
pin a different interpreter for the venv: `make install PY=3.12`.

`make install` is idempotent: it **reuses** an existing `.venv` (printing which
interpreter it found) instead of asking to replace it, then reinstalls the package
in place. Remove `.venv` yourself if you want a clean rebuild, and set `PY=...` to
choose the interpreter for a new one. It finishes by importing `linattn`, `torch`
and `pytest`, so a successful run is also a smoke test.

What you get, and what to read first:

| command | output | what it answers |
| --- | --- | --- |
| `make check` | nothing (exit status) | does every variant honour its budget, reduce to dense at full budget, stay causal, and match the chunked/recurrent state algorithms bit-for-bit in float64? |
| `make bench` | `reports/suite_*.md` | for every variant at one common budget: error vs. dense, attention-mass recall, KV entries read per query, state size, wall clock |
| | `reports/decode_*.md` | decode time vs. **bytes read** (C kernel vs. torch), i.e. the argument the whole literature rests on |
| `make experiments` | `reports/selector_capacity.md` | can a learned indexer/gate rank keys like softmax, before vs. after fitting? |
| | `reports/state_capacity.md` | deterministic capacity curve of a fixed-size delta-rule state |
| | `reports/hybrid_recall.md` | how much exact attention a hybrid needs (and an honest note on where MQAR stops discriminating) |
| `make bench-frontier` | `reports/frontier_2026.md` | the 2025–2026 variants at one budget, including the unflattering small-scale result |
| `make csa2-analysis` | `reports/csa2_approximation.md` | separates selection, pooling and KV-precision error against dense softmax |
| `make csa2-official-check` | `reports/csa2_official_config.md` | compares the surrogate defaults with the disclosed V4.1-Flash CSA2 schedule |
| `make approx-validation` | `reports/approximation_validation.md` | metric hierarchy across sparse, kernel and low-rank variants |
| `make error-propagation` | `reports/error_propagation.md` | local attention error versus downstream residual drift |

If something does not run, `python -m linattn.cli backends` prints exactly what your
machine can do (CUDA / MPS / Triton / FlexAttention / thread count), and every
report header records the same environment so numbers stay attributable. Details,
knobs and troubleshooting: [docs/reproducing.md](docs/reproducing.md).

## The variants

| family | variants | what they share |
| --- | --- | --- |
| dense (oracle) | `dense` | the reference every other number is measured against |
| structural / local | `sliding_window`, `streaming_sink` | window + attention sinks, the always-safe configuration |
| control baseline | `random_eviction` | uniform eviction keeping prompt + recent window: the critique of attention-mass scores (arXiv:2609.03430) |
| query-aware selection | `quest`, `minference`, `flexprefill`, `xattention`, `moba`, `dsa`, `seerattention_r`, `saga`, `pisa`, `jssa`, `masa`, `oval` | keep the cache, choose which blocks/tokens/pages to read |
| feature-axis sparsity | `sfa` | keep every token, sparsify the query/key feature code |
| native sparse | `nsa` | compressed + selected + window branches with learned gates |
| eviction | `h2o`, `snapkv`, `tova`, `dropkv`, `restkv`, `beaconkv` | budgeted cache; the old mass scores and the new output/query-aware scores |
| compressed state | `mla`; `ssd` / `gdn` / `kda` / `gdn2` / `qdelta` / `kdn` / `swila` layers; `hybrid` | fixed-size state instead of a growing cache |
| compressed + quantised (CSA2-style) | **`csa2`** | pooled key groups + FP4/FP8 cache + top-k group selection, with cross-layer sharing modelled by `CSA2Stack` |

`python -m linattn.cli variants` lists both families (attention variants **and**
the state mixers), and `python -m linattn.cli mixers` prints each mixer's decay,
gates, state size and per-token cost. The mixers have their own page,
[docs/variants/state_mixers.md](docs/variants/state_mixers.md), because they take
`x` rather than pre-projected `(q, k, v)`.

The eviction family needs a more careful statement than the one in the first
release. H2O/SnapKV/TOVA used attention-mass scores and are kept as lineage and
controls. The 2026 result that random selection matches them on reasoning traces is
real, but it motivated a new family rather than ending the field: **DropKV** ranks by
output perturbation, **ReST-KV** uses reconstruction and smoothing, and **BeaconKV**
keeps query-cluster representatives for later revisits. The compressed-state family
also gained **Q-Delta**, **KDN** and **SwiLA**; the first four mixers retain the
verified chunkwise path, while the three new ones expose exact reference recurrences
and cite the papers' specialised kernels.

Per-variant pages (mechanism, usage, cost table, caveats, citations) live in
[`docs/variants/`](docs/variants/index.md) and are generated from the code.

## What the survey concludes, in six lines

1. **Attention-mass eviction is a weak default; output-aware eviction is live.**
   Random selection matches tuned mass scores on reasoning traces, but DropKV,
   ReST-KV and BeaconKV change the objective and report gains.
2. **Block/page selection won because kernels can execute it**, and the 2026
   question is the routing objective and its cost: PISA makes routing log-linear,
   JSSA learns a low-rank token subspace, and MASA scores output-error reduction
   rather than attention mass.
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

### Continuous integration

`.github/workflows/ci.yml` runs the validation suite plus the small benchmarks on
every push, and a `-m gpu` job for the Triton parity tests. It is **not part of the
initial pushed commit**: pushing a workflow file requires the `workflow` scope on
the GitHub token, which the CLI token used for the first push did not have. To add
it:

```bash
gh auth refresh -h github.com -s workflow      # interactive, opens a browser
git add .github/workflows/ci.yml
git commit -m "ci: validation + benchmark workflow"
git push
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
