"""Matched-checkpoint audit of local attention error and downstream behavior.

This experiment replaces one dense attention layer at a time in tiny but trained
causal language models.  Every comparison holds model weights, inputs and all
other layers fixed.  It therefore tests the causal effect of an attention
approximation rather than correlating benchmark scores with implementation type.

The study is intentionally small.  Its purpose is to validate a measurement
protocol and expose metric failures, not to rank production models.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from approximation_certificates import relative_l2, total_variation
from linattn.core import AttnConfig, dense_attention
from linattn.kv.quant import QuantSpec, roundtrip
from linattn.registry import build
from linattn.reporting import save_report


FILLER = 0
KEY_BASE = 1          # six keys
VALUE_BASE = 7        # 16 values
QUERY_BASE = 23       # six query-key instruction tokens
EVIDENCE = 29
SUM_QUERY = 30
VOCAB = 32


@dataclass(frozen=True)
class TinyConfig:
    vocab: int = VOCAB
    seq_len: int = 48
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 2


class TinyBlock(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = d // cfg.n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.mlp_ratio * d),
            nn.GELU(),
            nn.Linear(cfg.mlp_ratio * d, d),
        )

    def forward(
        self,
        h: torch.Tensor,
        attention_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
        retain_attention_grad: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        b, t, d = h.shape
        x = self.ln1(h)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        attn = attention_fn(q, k, v) if attention_fn is not None else dense_attention(q, k, v)
        if retain_attention_grad:
            attn.retain_grad()
        h = h + self.out(attn.transpose(1, 2).reshape(b, t, d))
        h = h + self.mlp(self.ln2(h))
        return h, attn, (q, k, v)


class TinyCausalLM(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TinyBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_fn: Optional[Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        *,
        capture_grad: bool = False,
    ):
        b, t = tokens.shape
        positions = torch.arange(t, device=tokens.device)
        h = self.embed(tokens) * math.sqrt(self.cfg.d_model) + self.pos(positions).unsqueeze(0)
        attentions, qkvs = [], []
        for layer, block in enumerate(self.blocks):
            fn = None if attention_fn is None else (
                lambda q, k, v, layer=layer: attention_fn(layer, q, k, v)
            )
            h, attn, qkv = block(h, fn, capture_grad)
            attentions.append(attn)
            qkvs.append(qkv)
        logits = self.head(self.ln_f(h))
        return logits, h, attentions, qkvs


def _place_pair(tokens: torch.Tensor, pos: int, key: int, value: int) -> None:
    tokens[..., pos] = KEY_BASE + key
    tokens[..., pos + 1] = VALUE_BASE + value


def sample_batch(
    task: str,
    batch: int,
    seq_len: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create retrieval-shaped or dispersed-evidence examples."""
    tokens = torch.full((batch, seq_len), FILLER, dtype=torch.long)
    targets = torch.empty(batch, dtype=torch.long)
    if task == "retrieval":
        n_pairs = 4
        block = (seq_len - 2) // n_pairs
        keys = torch.stack([torch.randperm(n_pairs, generator=generator) for _ in range(batch)])
        values = torch.randint(0, 16, (batch, n_pairs), generator=generator)
        for i in range(n_pairs):
            offset = torch.randint(1, max(2, block - 1), (batch,), generator=generator)
            start = 1 + i * block + offset
            for row in range(batch):
                _place_pair(tokens[row], int(start[row]), int(keys[row, i]), int(values[row, i]))
        query_choice = torch.randint(0, n_pairs, (batch,), generator=generator)
        tokens[:, -1] = QUERY_BASE + query_choice
        targets = values[torch.arange(batch), query_choice]
        return tokens, targets

    if task == "dispersed":
        n_evidence = 2
        block = (seq_len - 2) // n_evidence
        values = torch.randint(0, 16, (batch, n_evidence), generator=generator)
        for i in range(n_evidence):
            offset = torch.randint(0, max(1, block - 2), (batch,), generator=generator)
            start = 1 + i * block + offset
            for row in range(batch):
                tokens[row, int(start[row])] = EVIDENCE
                tokens[row, int(start[row]) + 1] = VALUE_BASE + int(values[row, i])
        tokens[:, -1] = SUM_QUERY
        targets = values.sum(-1).remainder(16)
        return tokens, targets

    raise ValueError(f"unknown task {task!r}")


def _accuracy(model: TinyCausalLM, task: str, seed: int, examples: int = 512) -> float:
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(0, examples, 128):
            tokens, targets = sample_batch(task, min(128, examples - _), model.cfg.seq_len, gen)
            pred = model(tokens)[0][:, -1].argmax(-1)
            correct += int((pred == targets).sum())
    model.train()
    return correct / examples


def train_task(
    task: str,
    seed: int,
    *,
    steps: int = 900,
    batch: int = 96,
    lr: float = 3e-3,
) -> Tuple[TinyCausalLM, Dict[str, float]]:
    torch.manual_seed(seed)
    model = TinyCausalLM(TinyConfig())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(10_000 + seed)
    final_loss = float("nan")
    for step in range(steps):
        tokens, targets = sample_batch(task, batch, model.cfg.seq_len, gen)
        logits = model(tokens)[0][:, -1]
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final_loss = float(loss.detach())
        if step in (steps // 3, 2 * steps // 3):
            for group in opt.param_groups:
                group["lr"] *= 0.35
    train_acc = _accuracy(model, task, 20_000 + seed)
    eval_acc = _accuracy(model, task, 30_000 + seed)
    return model, {
        "final_train_loss": final_loss,
        "train_accuracy": train_acc,
        "eval_accuracy": eval_acc,
    }


def _attention_cfg(model: TinyCausalLM) -> AttnConfig:
    return AttnConfig(
        n_heads=model.cfg.n_heads,
        n_kv_heads=model.cfg.n_heads,
        head_dim=model.cfg.d_model // model.cfg.n_heads,
        block_size=8,
        window_size=8,
        topk_blocks=2,
        topk_tokens=16,
        sink_size=1,
        cache_budget=16,
        obs_window=8,
    )


def variant_specs(cfg: AttnConfig) -> List[Tuple[str, Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]]:
    """Variants plus matched-precision controls."""
    specs = []
    local_cfgs = {
        "quest-1block": ("quest", cfg.with_(topk_blocks=1)),
        "quest-2block": ("quest", cfg.with_(topk_blocks=2)),
        "moba-1block": ("moba", cfg.with_(topk_blocks=1)),
        "moba-2block": ("moba", cfg.with_(topk_blocks=2)),
        "minference-block": ("minference", cfg.with_(extras={"pattern": "block_sparse"})),
        "mass-oracle-2block": ("mass_oracle", cfg.with_(extras={"topk_blocks": 2, "sinks": 0})),
        "sliding-window": ("sliding_window", cfg.with_(window_size=8)),
        "performer-32": ("performer", cfg.with_(extras={"features": 32})),
    }
    for label, (name, variant_cfg) in local_cfgs.items():
        layer = build(name, variant_cfg)

        def make_fn(layer):
            def fn(_layer_idx, q, k, v):
                out, _ = layer.forward(q, k, v, chunk=q.shape[-2])
                return out
            return fn

        specs.append((label, make_fn(layer)))

    def quantized(bits: int):
        spec = QuantSpec(bits=bits, group_size=16)

        def fn(_layer_idx, q, k, v):
            kq = roundtrip(k, spec).to(k.dtype)
            vq = roundtrip(v, spec).to(v.dtype)
            return dense_attention(q, kq, vq)

        return fn

    specs.extend((("kv-fp8", quantized(8)), ("kv-fp4", quantized(4))))
    return specs


def _rank(x: torch.Tensor) -> torch.Tensor:
    order = x.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(len(x), dtype=torch.float64)
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3:
        return float("nan")
    a = _rank(torch.tensor(x, dtype=torch.float64))
    b = _rank(torch.tensor(y, dtype=torch.float64))
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def evaluate_checkpoint(
    model: TinyCausalLM,
    task: str,
    seed: int,
    *,
    eval_examples: int = 256,
) -> List[Dict[str, float]]:
    """Evaluate one-layer interventions under a fixed checkpoint and batch."""
    gen = torch.Generator().manual_seed(40_000 + seed)
    tokens, targets = sample_batch(task, eval_examples, model.cfg.seq_len, gen)
    model.eval()
    ref_logits, ref_h, ref_attn, ref_qkv = model(tokens, capture_grad=True)
    ref_loss = F.cross_entropy(ref_logits[:, -1], targets)
    ref_loss.backward()
    ref_probs = torch.softmax(ref_logits[:, -1], -1)
    cfg = _attention_cfg(model)
    rows: List[Dict[str, float]] = []

    for label, fn in variant_specs(cfg):
        for layer_idx in range(model.cfg.n_layers):
            with torch.no_grad():
                def one_layer(current, q, k, v, target_layer=layer_idx):
                    if current != target_layer:
                        return dense_attention(q, k, v)
                    return fn(current, q, k, v)

                logits, final_h, attentions, _ = model(tokens, one_layer)
                approx_attn = attentions[layer_idx]
                local_delta = approx_attn - ref_attn[layer_idx]
                grad = ref_attn[layer_idx].grad
                if grad is None:
                    raise RuntimeError("reference attention gradient was not retained")
                projected = (grad * local_delta).sum(dim=(1, 2, 3))
                approx_loss = F.cross_entropy(logits[:, -1], targets)
                approx_probs = torch.softmax(logits[:, -1], -1)
                p = ref_probs.clamp_min(1e-9)
                q = approx_probs.clamp_min(1e-9)
                rows.append({
                    "task": task,
                    "seed": seed,
                    "variant": label,
                    "layer": layer_idx,
                    "local_attention_rel_l2": float(relative_l2(ref_attn[layer_idx], approx_attn).mean()),
                    "final_hidden_rel_l2": float(relative_l2(ref_h[:, -1], final_h[:, -1]).mean()),
                    "logit_kl": float((p * (p / q).log()).sum(-1).mean()),
                    "loss_delta": float((approx_loss - ref_loss).mean()),
                    "task_projected_loss_delta": float(projected.mean()),
                    "task_projected_abs": float(projected.abs().mean()),
                    "reference_loss": float(ref_loss.mean()),
                    "accuracy_delta": float(
                        (logits[:, -1].argmax(-1) == targets).float().mean()
                        - (ref_logits[:, -1].argmax(-1) == targets).float().mean()
                    ),
                })
    model.zero_grad(set_to_none=True)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--eval-examples", type=int, default=256)
    args = parser.parse_args(argv)

    rows: List[Dict[str, float]] = []
    training: List[Dict[str, float]] = []
    for task in ("retrieval", "dispersed"):
        for seed in range(args.seeds):
            model, stats = train_task(task, seed, steps=args.steps)
            training.append({"task": task, "seed": seed, **stats})
            rows.extend(evaluate_checkpoint(model, task, seed, eval_examples=args.eval_examples))

    summary = []
    for task in ("retrieval", "dispersed"):
        for variant in sorted({r["variant"] for r in rows if r["task"] == task}):
            subset = [r for r in rows if r["task"] == task and r["variant"] == variant]
            summary.append({
                "task": task,
                "variant": variant,
                "n": len(subset),
                "local_attention_rel_l2": statistics.mean(r["local_attention_rel_l2"] for r in subset),
                "final_hidden_rel_l2": statistics.mean(r["final_hidden_rel_l2"] for r in subset),
                "logit_kl": statistics.mean(r["logit_kl"] for r in subset),
                "loss_delta": statistics.mean(r["loss_delta"] for r in subset),
                "task_projected_abs": statistics.mean(r["task_projected_abs"] for r in subset),
                "accuracy_delta": statistics.mean(r["accuracy_delta"] for r in subset),
            })

    correlations = []
    for task in ("retrieval", "dispersed"):
        subset = [r for r in rows if r["task"] == task]
        y = [r["loss_delta"] for r in subset]
        correlations.append({
            "task": task,
            "n": len(subset),
            "local_vs_loss_spearman": spearman([r["local_attention_rel_l2"] for r in subset], y),
            "hidden_vs_loss_spearman": spearman([r["final_hidden_rel_l2"] for r in subset], y),
            "projected_vs_loss_spearman": spearman([r["task_projected_abs"] for r in subset], y),
            "local_vs_projected_spearman": spearman(
                [r["local_attention_rel_l2"] for r in subset],
                [r["task_projected_abs"] for r in subset],
            ),
        })

    path = save_report(
        "approximation_audit",
        summary,
        title="Matched-checkpoint audit: local attention error to task behavior",
        columns=["task", "variant", "n", "local_attention_rel_l2", "final_hidden_rel_l2",
                 "logit_kl", "loss_delta", "task_projected_abs", "accuracy_delta"],
        notes=(
            "Two-layer trained causal transformers on synthetic single-key retrieval and "
            "dispersed four-evidence summation. One attention layer is replaced at a "
            "time while weights and other layers remain fixed. task_projected_abs is the "
            "magnitude of the first-order loss effect g.delta_a on the dense trajectory; "
            "it uses the downstream Jacobian but is not a finite-error guarantee. "
            "Synthetic tasks validate the audit instrument, not frontier-model quality."
        ),
    )
    save_report(
        "approximation_audit_correlations",
        correlations,
        title="Matched-checkpoint audit: metric rank correlations",
        columns=["task", "n", "local_vs_loss_spearman", "hidden_vs_loss_spearman",
                 "projected_vs_loss_spearman", "local_vs_projected_spearman"],
        notes=(
            "Spearman correlations over one-layer interventions. A high local-error "
            "correlation is evidence about this synthetic setup, not a general guarantee."
        ),
    )
    save_report(
        "approximation_audit_training",
        training,
        title="Matched-checkpoint audit: tiny-model training controls",
        columns=["task", "seed", "final_train_loss", "train_accuracy", "eval_accuracy"],
        notes="Held-out synthetic examples from the same generators used for evaluation.",
    )
    print(f"wrote {path}")
    for r in training:
        print(f"{r['task']:<10} seed={r['seed']} train_acc={r['train_accuracy']:.3f} "
              f"eval_acc={r['eval_accuracy']:.3f} loss={r['final_train_loss']:.3f}")
    for r in correlations:
        print(f"{r['task']:<10} rho(local,loss)={r['local_vs_loss_spearman']:.3f} "
              f"rho(hidden,loss)={r['hidden_vs_loss_spearman']:.3f} "
              f"rho(projected,loss)={r['projected_vs_loss_spearman']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
