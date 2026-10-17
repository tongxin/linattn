# Triton on Apple Silicon: what actually exists (October 2026)

Short answer: **official Triton still has no Metal backend and ships no macOS
wheel**, but there is one credible third-party compiler — **`triton-msl`** — that
compiles standard `@triton.jit` kernels to Metal Shading Language, plus a CPU-only
path (`triton-cpu`) and a different, MLX-native route (what this repository already
uses). This page records the evidence, the install path, and what it means for the
Triton kernels in `src/linattn/kernels/triton/`.

## 1. The options, compared

| option | target | what it is | maturity | verdict for this repo |
| --- | --- | --- | --- | --- |
| **official Triton** (`triton-lang/triton`) | CUDA, ROCm, Intel XPU | the upstream compiler | no Metal backend; **zero macOS wheels** on PyPI (only `manylinux` `x86_64`/`aarch64`) | cannot run our kernels on a Mac GPU |
| **[`bledden/triton-msl`](https://github.com/bledden/triton-msl)** | Apple GPU (Metal) | `@triton.jit` → Triton IR → MSL → Metal | **alpha**, v0.3.0 (2026-09-16), MIT, ~24★; explicitly "not a universal correctness guarantee"; registers as Triton's out-of-tree `metal` backend | **the only real option**: same kernel source, verified plugin + install path (below) |
| [`triton-lang/triton-cpu`](https://github.com/triton-lang/triton-cpu) | any CPU (incl. arm64 Macs) | Intel's experimental CPU backend | active (214★), experimental | useful for *testing* kernel logic on a Mac, not for GPU validation |
| **[`ZMLX`](https://github.com/Hmbown/ZMLX)** | Apple GPU (MLX) | patches MLX models with fused Metal kernels (`deltanet`, `swiglu`); token-identical outputs | active, ~52★ | adjacent, not Triton: no `@triton.jit` source |
| **MLX `mx.fast.metal_kernel`** | Apple GPU (Metal) | hand-written Metal kernels from Python | shipping (what `kernels/mlx/` uses) | the *practical* local path: no toolchain, no unofficial wheels |
| **PyTorch Inductor on MPS** | Apple GPU | generates Triton IR for `torch.compile` | usable **only with** `triton-msl` registered; there is no backend named `metal` | relevant if you compile models, not for our standalone kernels |

Two things that look like options but are not:

* **The upstream MPS PR.** `triton-lang/triton#9701` ("Add Apple MPS backend") was
  **closed without merging** on 2026-07-06, so there is nothing to wait for in the
  official wheel. (A fork, `DatasunriseOU/triton-pr9701`, carries it privately.)
* **Vulkan/SPIR-V via MoltenVK.** No maintained Triton Vulkan backend exists; the
  out-of-tree backends that are maintained target CUDA/ROCm/XPU. Triton's
  out-of-tree backend interface is, however, exactly what `triton-msl` builds on.

## 2. Installing `triton-msl` (the fast path)

What was verified here, on this machine (macOS 26.5.1, arm64, Python 3.14.6):

| check | result |
| --- | --- |
| upstream MPS PR `triton-lang/triton#9701` | **closed, not merged** (2026-07-06) |
| official Triton wheels on PyPI | **none for macOS** (3.8.0 ships `manylinux` x86_64/aarch64 only) |
| `triton-msl` on PyPI | 0.3.0, pure-Python wheel (0.85 MB), `requires-python >= 3.10` |
| its integration point | entry point **`[triton.backends] metal = triton_msl.backend`** -- Triton's official out-of-tree backend mechanism, so `import triton` discovers it |
| its dependency pins | `pyobjc-framework-Metal*` (direct Metal access); extra `[triton]` pins **`triton==3.7.0`** and `torch>=2.12` |
| its modules | `backend/`, `codegen/`, `inductor/`, `autotuning/`, `kda.py`, `fa_backward.py` -- consistent with the advertised KDA/FlashAttention/decode surface |
| prebuilt Triton macOS wheel | release asset for **cp314 / macOS 15+ / arm64**: 142 MB |

Not verified here: an end-to-end GPU run. The blocker is download volume -- that
142 MB Triton wheel plus a 127 MB `torch` cp314 wheel at ~150 KB/s on this network
is roughly 40 minutes of transfers, and the install was stopped for time. The
commands below are the project's documented ones and the sanity check is from its
README.

Requirements, from its docs: Apple Silicon, macOS 14+, Python 3.10+, Xcode Command
Line Tools, Xcode's Metal toolchain (`xcrun metal --version` must work; on macOS 26
install it once with `sudo xcodebuild -downloadComponent MetalToolchain`),
PyTorch 2.12+, and **Triton 3.7.0**.

There is no official macOS Triton wheel, so you either build Triton 3.7.0 from
source (expensive: LLVM/MLIR) or — if you are on **Python 3.14 / macOS 15+ / arm64**
— use the unofficial prebuilt wheel attached to the project's releases:

```bash
xcode-select --install                      # once
python3 -m venv .venv-msl && source .venv-msl/bin/activate
pip install torch triton-msl
pip install https://github.com/bledden/triton-msl/releases/download/triton-wheel-3.7.0-cp314-macos-arm64/triton-3.7.0+git4da2e268-cp314-cp314-macosx_15_0_arm64.whl
```

Sanity check (from the project's README, verbatim shape):

```python
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

n = 1024
x, y = torch.randn(n, device="mps"), torch.randn(n, device="mps")
out = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
torch.mps.synchronize()
print((out - (x + y)).abs().max().item())   # ~1e-7
```

## 3. What it supports (and what it refuses)

From the project's own capability list, the supported surface is broader than
"elementwise only":

* elementwise kernels, reductions, scans and atomics;
* strided matmul with bounded epilogues, weight-only int8/int4 routes;
* **FlashAttention, decode/biased attention and MLA** routes;
* direct Metal operations for **KDA** and FlashAttention backward;
* PyTorch Inductor registration (`torch.compile` on MPS), plus a narrower MLX launch
  interface.

Limits that matter for a kernel repository:

* support depends on **source, dtype, layout and tile size** — a kernel that works
  at one tile can be refused at another, and refusal is the intended behaviour
  ("preserve the source computation or refuse explicitly");
* no fp64 GPU arithmetic, no microscaling;
* **alpha**: the 0.3.0 notes admit earlier silent-wrong-result bugs, and report
  cold-compile specialisations as slow (~9.5 s for one first-launch observation);
* maintenance risk: a small project with a single visible maintainer, and an
  **unofficial** Triton wheel in the loop (supply-chain considerations apply);
* the wheel is pinned to a specific Triton revision (`git4da2e268`); mixing versions
  is unsupported.

## 4. How to use it with this repository

The Triton kernels here (`kernels/triton/*.py`) are ordinary `@triton.jit` code, so
they are candidates for `triton-msl` — but they were written against CUDA
semantics (`tl.dot`, `static_range`, strided `tl.load`), and per the support matrix
each one must be *tried*, not assumed. The parity harness is already the right tool:

```bash
python -m linattn.cli backends          # reports triton_msl: True (entry-point probe)
```

To try a kernel, copy the body of `kernels/triton/block_sparse.py` into a script,
launch it on `mps` tensors like the README's add example, and check three things in
order:

1. it **compiles** -- or the backend refuses explicitly, which is a portability
   finding rather than a bug (support is per source/dtype/layout/tile);
2. the output matches the torch reference (`block_sparse_attention_torch` is the
   same computation);
3. only then compare timings: cold specialisation is slow (the project measures
   ~9.5 s for one first-launch observation), so warm numbers are the meaningful ones.

The kernels here most likely to need adjustment use `tl.dot` tiling,
`tl.static_range` over block indices, and strided `tl.load` with masked offsets --
all features whose Metal support depends on the tile shape.

## 5. Recommendation

* **For validating the variants on a Mac today**: use the MLX backend
  ([mlx.md](mlx.md)). It needs no compiler toolchain, no unofficial wheels, and its
  parity numbers are in `reports/mlx_decode.md`.
* **For keeping the same `@triton.jit` source across CUDA and a Mac**: install
  `triton-msl` and run the parity harness per kernel. Plan for alpha-quality
  support: pin Triton 3.7.0, keep a CUDA machine as the reference, and treat any
  refusal as a portability finding.
* **Do not** expect an official fix soon: the upstream MPS PR is closed, and Triton's
  released wheels remain Linux-only.
