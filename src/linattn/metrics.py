"""Cost models and quality diagnostics shared by the benchmarks and tests.

Two kinds of numbers matter for this literature and they are easy to conflate:

* **quality** -- does the sparse/compressed computation still produce the
  attention output (and the LM prediction) that dense attention would produce?
* **cost** -- how many KV entries does the kernel actually read, and how much
  persistent state does it hold?

The helpers here keep the two separated.  ``ReadBudget`` is deliberately counted
in *KV entries read per query token* because that is what a decode kernel pays
for: on modern accelerators decode attention is memory-bandwidth bound, so
FLOPs alone are a misleading metric (see ``docs/kernels.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch

from linattn.core import AttentionVariant, block_reduce, dense_attention, repeat_kv

__all__ = [
    "ReadBudget",
    "rel_error",
    "max_abs_error",
    "attention_mass",
    "mass_recall",
    "sparsity_stats",
    "efficiency_table",
    "text_table",
    "quality_table",
]


@dataclass
class ReadBudget:
    """KV entries read per query token, and the implied traffic."""

    tokens: float          # KV entries read per query
    bytes_per_query: float # bytes read per query (fp16 KV)
    note: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.tokens:.1f} entries/query ({self.bytes_per_query / 1024:.1f} KiB) {self.note}".strip()


# --------------------------------------------------------------------------------------
# error metrics
# --------------------------------------------------------------------------------------
def rel_error(actual: torch.Tensor, expected: torch.Tensor, eps: float = 1e-6) -> float:
    """Relative L2 error, the standard kernel-parity metric."""
    actual, expected = actual.detach(), expected.detach()
    denom = expected.norm().clamp_min(eps)
    return float((actual - expected).norm() / denom)


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.detach() - expected.detach()).abs().max())


# --------------------------------------------------------------------------------------
# attention mass bookkeeping (the language the sparsity papers use)
# --------------------------------------------------------------------------------------
def attention_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block: int = 1,
    causal: bool = True,
) -> torch.Tensor:
    """Dense attention mass, pooled into query blocks x key blocks.

    Returns ``(B, H, n_q_blocks, n_k_blocks)`` where entry ``[.., a, b]`` is the
    total attention mass paid by query block ``a`` to key block ``b``.  This is the
    quantity every block-selection method is trying to approximate, and the quantity
    :func:`mass_recall` scores selections against.  With ``block=1`` no pooling is
    done and the result is token level ``(B, H, T_q, T_k)``.
    """
    weights = dense_attention(q, k, v, causal=causal, return_weights=True)[1]
    if block > 1:
        for dim in (-1, -2):
            length = weights.shape[dim]
            pad = (-length) % block
            if pad:
                weights = torch.nn.functional.pad(weights, (0, pad) if dim == -1 else (0, 0, 0, pad))
            weights = block_reduce(weights, block, dim, mode="sum")
    return weights


def mass_recall(mass: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Fraction of a query's attention mass captured by the selected blocks.

    ``mass`` is ``(B, H, T_q, n_blocks)`` and ``selected`` is a boolean tensor of
    the same shape (or broadcastable to it).
    """
    if selected.dtype != torch.bool:
        selected = selected.bool()
    total = mass.sum(-1, keepdim=True).clamp_min(1e-12)
    return (mass * selected).sum(-1) / total.squeeze(-1)


def sparsity_stats(selected: torch.Tensor) -> Dict[str, float]:
    """Summary of a boolean selection mask ``(..., n_blocks)``."""
    frac = selected.to(torch.float32).mean()
    per_row = selected.to(torch.float32).sum(-1)
    return {
        "kept_fraction": float(frac),
        "mean_blocks_per_query": float(per_row.mean()),
        "min_blocks_per_query": float(per_row.min()),
        "max_blocks_per_query": float(per_row.max()),
    }


# --------------------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------------------
def text_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    align: str = "l",
) -> str:
    """Render an aligned plain-text table suitable for a terminal."""
    if len(align) != len(headers):
        raise ValueError("align must have one entry per column")
    body = [["" if cell is None else str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in body:
        if len(row) != len(headers):
            raise ValueError("every row must have one cell per column")
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def pad(text: str, width: int, mode: str) -> str:
        return text.rjust(width) if mode == "r" else text.ljust(width)

    lines = [
        "  ".join(pad(str(h), w, a) for h, w, a in zip(headers, widths, align)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(pad(cell, w, a) for cell, w, a in zip(row, widths, align)).rstrip() for row in body]
    return "\n".join(lines)


def efficiency_table(
    variants: Sequence[AttentionVariant],
    seq_len: int,
    batch: int = 1,
    *,
    markdown: bool = False,
) -> str:
    """Analytic memory/FLOPs/read-budget table for a set of variants.

    The default is an aligned plain-text table for the CLI; pass
    ``markdown=True`` when embedding the table in documentation.
    """
    headers = ["variant", "family", "KV/state @T (MiB)", "storage precision",
               "prefill attn FLOPs @T (G)", "KV reads/query"]
    rows = []
    for v in variants:
        mem = v.memory_bytes(seq_len, batch) / 2**20
        fl = v.flops(seq_len, batch) / 1e9
        reads = v.read_budget(seq_len, batch)
        precision = getattr(v, "storage_precision", "fp16 KV")
        rows.append([v.name, v.family, f"{mem:.1f}", precision, f"{fl:.1f}", f"{reads:,.0f}"])

    if not markdown:
        return text_table(headers, rows, align="llrlrr")

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for name, family, mem, precision, fl, reads in rows:
        lines.append(f"| `{name}` | {family} | {mem} | {precision} | {fl} | {reads} |")
    return "\n".join(lines)


def quality_table(records: Sequence[Dict]) -> str:
    """Markdown table from a list of measurement dicts produced by ``harness``."""
    if not records:
        return "_(no records)_"
    keys = ["variant", "rel_err", "mass_recall", "kv_reads", "latency_ms", "peak_mib"]
    rows = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for rec in records:
        cells = []
        for key in keys:
            value = rec.get(key, "")
            cells.append(f"{value:.4g}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)
