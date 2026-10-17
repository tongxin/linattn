# Apple Silicon (MLX/Metal) decode attention vs torch CPU

*Generated 2026-10-17T12:19:02Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

Single decode step (T_q=1), head dim 64, 8 query heads / 4 KV heads, fp32, best of 10 (MLX) or 5 (torch) runs after one warm-up. `readings_per_query` is what the selector allowed the kernel to read; the Metal column says whether the custom kernel handled the step (page-list variants only).

Device numerics on this machine: MLX-GPU fp32 matmul 7.75e-04 relative error, MLX-CPU 1.12e-06, custom Metal decode kernel vs torch 4.21e-07. The GPU's generic matmul path runs at roughly fp16 accuracy here, which is why the MLX parity tests use a loose tolerance on GPU and a tight one on the CPU stream -- while the hand-written kernel, with explicit fp32 FMAs, stays at fp32.

| device | variant | context | reads_per_query | ms_per_step | metal_kernel |
| --- | --- | --- | --- | --- | --- |
| mlx-metal | dense | 4096 | 4096 | 0.646 | 0 |
| mlx-metal | sliding_window | 4096 | 256 | 0.8642 | 0 |
| mlx-metal | quest | 4096 | 128 | 1.123 | 1 |
| torch-cpu | dense | 4096 | 4096 | 0.4182 | 0 |
| mlx-metal | dense | 16384 | 1.638e+04 | 1.317 | 0 |
| mlx-metal | sliding_window | 16384 | 256 | 1.189 | 0 |
| mlx-metal | quest | 16384 | 128 | 1.591 | 1 |
| torch-cpu | dense | 16384 | 1.638e+04 | 1.698 | 0 |
| mlx-metal | dense | 65536 | 6.554e+04 | 4.113 | 0 |
| mlx-metal | sliding_window | 65536 | 256 | 3.726 | 0 |
| mlx-metal | quest | 65536 | 128 | 5.8 | 1 |
