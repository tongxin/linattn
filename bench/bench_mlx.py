"""Apple-Silicon benchmark: MLX/Metal vs torch-CPU for the decode path.

Two things are measured, because they answer different questions:

1. **Latency vs context length** for ``dense``, ``sliding_window`` and ``quest``
   decode steps, MLX-on-GPU against torch-on-CPU, with the KV entries actually read
   reported next to each timing.  This is the local reproduction of the survey's
   central claim: decode time tracks bytes read.
2. **Device numerics.**  MLX's generic fp32 matmul on this Metal device is compared
   with MLX's own CPU backend, and the custom Metal decode kernel is compared with
   torch.  On the machine that produced ``reports/`` the GPU matmul path runs at
   roughly fp16 accuracy while the hand-written kernel stays at fp32 -- worth
   knowing before reading any MLX parity number.

Run::

    python bench/bench_mlx.py --lengths 4096 16384 65536
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linattn.kernels.mlx import metal_available, mlx_available, mlx_import_error, mx  # noqa: E402
from linattn.kernels.mlx import ops  # noqa: E402
from linattn.kernels.mlx.metal_decode import decode_attention_torch, metal_decode_attention  # noqa: E402
from linattn.kernels.mlx.variants import mlx_forward  # noqa: E402
from linattn.reporting import save_report  # noqa: E402


def time_mlx(fn, repeat: int = 10) -> float:
    fn()
    mx.eval(fn()[0] if isinstance(fn(), tuple) else fn())
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out[0] if isinstance(out, tuple) else out)
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def time_torch(fn, repeat: int = 5) -> float:
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def numerics():
    """fp32 matmul error of MLX-GPU vs MLX-CPU, and the Metal kernel vs torch."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((512, 512)).astype(np.float32)
    b = rng.standard_normal((512, 512)).astype(np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    gpu = np.array(mx.matmul(mx.array(a), mx.array(b)))
    with mx.stream(mx.cpu):
        cpu = np.array(mx.matmul(mx.array(a), mx.array(b)))
    gpu_err = float(np.abs(gpu - ref).max() / np.abs(ref).max())
    cpu_err = float(np.abs(cpu - ref).max() / np.abs(ref).max())

    # custom Metal kernel vs torch, same inputs
    h, hkv, tk, d, page = 4, 2, 4096, 64, 16
    q = rng.standard_normal((1, h, 1, d)).astype(np.float32)
    k = rng.standard_normal((1, hkv, tk, d)).astype(np.float32)
    v = rng.standard_normal((1, hkv, tk, d)).astype(np.float32)
    ids = rng.integers(0, tk // page, (1, h, 8)).astype(np.int32)
    out, _ = metal_decode_attention(ops.to_mlx(q), ops.to_mlx(k), ops.to_mlx(v), ops.to_mlx(ids),
                                    page_size=page, n_kv_heads=hkv)
    want = decode_attention_torch(*(torch.from_numpy(x) for x in (q, k, v)),
                                  torch.from_numpy(ids.astype(np.int64)), page_size=page)
    kernel_err = float(np.abs(np.array(out) - want.numpy()).max() / np.abs(want.numpy()).max())
    return gpu_err, cpu_err, kernel_err


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="*", type=int, default=[4096, 16384, 65536])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=4, dest="kv_heads")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--page", type=int, default=16)
    parser.add_argument("--topk-pages", type=int, default=8, dest="topk_pages")
    parser.add_argument("--torch-max-len", type=int, default=16384, dest="torch_max_len")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not mlx_available():
        print(f"mlx unavailable: {mlx_import_error()}")
        return 1
    print(f"mlx device: {mx.default_device()} | metal: {metal_available()}")

    rng = np.random.default_rng(args.seed)
    h, hkv, d = args.heads, args.kv_heads, args.dim
    records = []
    for t_k in args.lengths:
        q = rng.standard_normal((1, h, 1, d)).astype(np.float32)
        k = rng.standard_normal((1, hkv, t_k, d)).astype(np.float32)
        v = rng.standard_normal((1, hkv, t_k, d)).astype(np.float32)

        for name in ("dense", "sliding_window", "quest"):
            ms = time_mlx(lambda: mlx_forward(name, ops.to_mlx(q), ops.to_mlx(k), ops.to_mlx(v),
                                             window=args.window, page_size=args.page,
                                             topk_pages=args.topk_pages))
            _, stats = mlx_forward(name, ops.to_mlx(q), ops.to_mlx(k), ops.to_mlx(v),
                                   window=args.window, page_size=args.page, topk_pages=args.topk_pages)
            records.append({
                "device": "mlx-metal",
                "variant": name,
                "context": t_k,
                "reads_per_query": stats["reads_per_query"],
                "ms_per_step": ms,
                "metal_kernel": stats["metal"],
            })
            print(f"mlx   {name:<15} T={t_k:>6}  reads={stats['reads_per_query']:>7.0f}  {ms:8.3f} ms")

        if t_k <= args.torch_max_len:
            qt, kt, vt = (torch.from_numpy(x) for x in (q, k, v))
            ms = time_torch(lambda: torch.nn.functional.scaled_dot_product_attention(
                qt, kt.repeat_interleave(h // hkv, dim=1), vt.repeat_interleave(h // hkv, dim=1)))
            records.append({"device": "torch-cpu", "variant": "dense", "context": t_k,
                            "reads_per_query": float(t_k), "ms_per_step": ms, "metal_kernel": 0.0})
            print(f"torch {'dense':<15} T={t_k:>6}  reads={t_k:>7}  {ms:8.3f} ms")

    gpu_err, cpu_err, kernel_err = numerics()
    print(f"numerics: mlx-gpu matmul {gpu_err:.2e} | mlx-cpu matmul {cpu_err:.2e} | metal kernel {kernel_err:.2e}")

    path = save_report(
        "mlx_decode",
        records,
        title="Apple Silicon (MLX/Metal) decode attention vs torch CPU",
        columns=["device", "variant", "context", "reads_per_query", "ms_per_step", "metal_kernel"],
        notes=(
            "Single decode step (T_q=1), head dim %d, %d query heads / %d KV heads, fp32, "
            "best of 10 (MLX) or 5 (torch) runs after one warm-up. `readings_per_query` is "
            "what the selector allowed the kernel to read; the Metal column says whether the "
            "custom kernel handled the step (page-list variants only).\n\n"
            "Device numerics on this machine: MLX-GPU fp32 matmul %.2e relative error, "
            "MLX-CPU %.2e, custom Metal decode kernel vs torch %.2e. The GPU's generic "
            "matmul path runs at roughly fp16 accuracy here, which is why the MLX parity "
            "tests use a loose tolerance on GPU and a tight one on the CPU stream -- while "
            "the hand-written kernel, with explicit fp32 FMAs, stays at fp32."
            % (args.dim, args.heads, args.kv_heads, gpu_err, cpu_err, kernel_err)
        ),
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
