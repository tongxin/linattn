"""Small residual-stack experiment for local-to-downstream error propagation.

This is not a trained transformer benchmark. It isolates the question that a
layerwise metric cannot answer: does a local attention-output error stay local,
decay, or amplify through the residual stream?
"""

from __future__ import annotations

import sys
import statistics
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from approximation_metrics import cosine_distance, relative_l2  # noqa: E402
from linattn.core import AttnConfig, dense_attention  # noqa: E402
from linattn.harness import make_inputs, structured_inputs  # noqa: E402
from linattn.registry import build  # noqa: E402
from linattn.reporting import save_report  # noqa: E402


class ResidualStack:
    def __init__(self, d_model: int = 64, heads: int = 4, seed: int = 0):
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        gen = torch.Generator().manual_seed(seed)
        self.w = torch.randn(d_model, d_model, generator=gen, dtype=torch.float64) / d_model**0.5
        self.ff = torch.randn(d_model, d_model, generator=gen, dtype=torch.float64) / d_model**0.5
        self.cfg = AttnConfig(
            n_heads=heads,
            n_kv_heads=heads,
            head_dim=self.head_dim,
            block_size=4,
            window_size=4,
            topk_blocks=2,
            topk_tokens=8,
            sink_size=0,
            seed=seed,
        )

    def _qkv(self, h):
        b, t, d = h.shape
        x = h @ self.w
        x = x.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        return x, x, x

    def _attention(self, h, mode: str):
        q, k, v = self._qkv(h)
        if mode == "dense":
            out = dense_attention(q, k, v)
        else:
            layer = build(mode, self.cfg)
            out, _ = layer.forward(q, k, v, chunk=h.shape[1])
        return out.transpose(1, 2).reshape(h.shape)

    def step(self, h, mode="dense"):
        attn = self._attention(h, mode)
        ffn = torch.tanh(h @ self.ff) / self.d_model**0.5
        return h + attn + ffn


def _trajectory(stack, h, inject_at: int | None = None, mode: str = "performer"):
    states = [h]
    for layer in range(6):
        h = stack.step(h, mode if layer == inject_at else "dense")
        states.append(h)
    return states


def _run(input_kind: str, inject_at: int, seed: int = 0) -> Dict:
    stack = ResidualStack(seed=seed)
    if input_kind == "structured":
        q, k, v = structured_inputs(2, 4, 16, 16, kv_heads=4, seed=seed)
    else:
        q, k, v = make_inputs(2, 4, 16, 16, kv_heads=4, seed=seed)
    h0 = (q + k + v).transpose(1, 2).reshape(q.shape[0], q.shape[-2], 4 * 16).double()
    ref = _trajectory(stack, h0)
    approx = _trajectory(stack, h0, inject_at=inject_at)
    local = relative_l2(ref[inject_at + 1], approx[inject_at + 1])
    final = relative_l2(ref[-1], approx[-1])
    return {
        "input": input_kind,
        "inject_layer": inject_at,
        "local_rel_l2": local,
        "final_rel_l2": final,
        "amplification": final / max(local, 1e-12),
        "local_cosine_distance": cosine_distance(ref[inject_at + 1], approx[inject_at + 1]),
        "final_cosine_distance": cosine_distance(ref[-1], approx[-1]),
    }


def main() -> int:
    rows = []
    raw = []
    for input_kind in ("random", "structured"):
        for seed in range(3):
            for inject_at in range(6):
                raw.append(_run(input_kind, inject_at, seed=seed))
    for input_kind in ("random", "structured"):
        for inject_at in range(6):
            subset = [r for r in raw if r["input"] == input_kind and r["inject_layer"] == inject_at]
            rows.append({
                "input": input_kind,
                "inject_layer": inject_at,
                "n_seeds": len(subset),
                "local_rel_l2": statistics.mean(r["local_rel_l2"] for r in subset),
                "local_rel_l2_std": statistics.pstdev(r["local_rel_l2"] for r in subset),
                "final_rel_l2": statistics.mean(r["final_rel_l2"] for r in subset),
                "final_rel_l2_std": statistics.pstdev(r["final_rel_l2"] for r in subset),
                "amplification": statistics.mean(r["amplification"] for r in subset),
                "amplification_std": statistics.pstdev(r["amplification"] for r in subset),
            })
    path = save_report(
        "error_propagation",
        rows,
        title="Local attention error versus downstream residual-stream drift",
        columns=["input", "inject_layer", "n_seeds", "local_rel_l2", "local_rel_l2_std",
                 "final_rel_l2", "final_rel_l2_std", "amplification", "amplification_std"],
        notes=(
            "Six-layer fixed random residual stack; a Performer attention output replaces "
            "dense attention at one layer, then dense dynamics continue. This isolates "
            "propagation, not trained-model quality. Amplification is final/local "
            "relative L2 error. Random and planted-retrieval inputs are controls."
        ),
    )
    print(f"wrote {path}")
    for r in rows:
        print(f"{r['input']:<10} layer={r['inject_layer']} "
              f"local={r['local_rel_l2']:.4f} final={r['final_rel_l2']:.4f} "
              f"amp={r['amplification']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
