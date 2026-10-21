"""Small, auditable metric vocabulary for attention approximation studies.

The functions operate on already-normalized tensors. They do not decide whether a
variant's approximation is acceptable; they make the object being measured
explicit so later experiments cannot silently compare a mask recall to a task
score and call them the same thing.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1).float()


def relative_l2(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-12) -> float:
    r, e = _flat(reference), _flat(estimate)
    return float((r - e).norm() / r.norm().clamp_min(eps))


def cosine_distance(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-12) -> float:
    r, e = _flat(reference), _flat(estimate)
    cosine = torch.dot(r, e) / (r.norm() * e.norm()).clamp_min(eps)
    return float((1.0 - cosine).clamp(-2.0, 2.0))


def total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.float().reshape(-1, p.shape[-1])
    q = q.float().reshape(-1, q.shape[-1])
    return float(0.5 * (p - q).abs().sum(-1).mean())


def jensen_shannon(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.float().clamp_min(0)
    q = q.float().clamp_min(0)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * ((a + eps) / (b + eps)).log()).sum(-1)
    value = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(value.mean())


def hellinger(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.float().clamp_min(0)
    q = q.float().clamp_min(0)
    return float(torch.sqrt(0.5 * (p.sqrt() - q.sqrt()).square().sum(-1)).mean())


def mass_recall(reference: torch.Tensor, support: torch.Tensor) -> float:
    """Fraction of reference mass retained by a boolean support."""
    ref = reference.float()
    support = support.to(torch.bool)
    return float((ref * support).sum() / ref.sum().clamp_min(1e-12))


def topk_agreement(reference: torch.Tensor, estimate: torch.Tensor, k: int) -> float:
    """Intersection-over-k for the top-k support on the final axis."""
    k = max(1, min(k, reference.shape[-1], estimate.shape[-1]))
    a = torch.topk(reference, k, dim=-1).indices
    b = torch.topk(estimate, k, dim=-1).indices
    return float((a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean())


def attention_operator_metrics(
    p_ref: torch.Tensor,
    p_hat: torch.Tensor,
    o_ref: torch.Tensor,
    o_hat: torch.Tensor,
    *,
    k: int = 8,
) -> Dict[str, float]:
    return {
        "operator_rel_l2": relative_l2(o_ref, o_hat),
        "operator_cosine_distance": cosine_distance(o_ref, o_hat),
        "attention_tv": total_variation(p_ref, p_hat),
        "attention_js": jensen_shannon(p_ref, p_hat),
        "attention_hellinger": hellinger(p_ref, p_hat),
        "topk_agreement": topk_agreement(p_ref, p_hat, k),
    }


def _demo() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 64)
    p = torch.softmax(logits, -1)
    q = torch.softmax(logits + 0.1 * torch.randn_like(logits), -1)
    v = torch.randn(64, 16)
    print(attention_operator_metrics(p, q, p @ v, q @ v, k=8))


if __name__ == "__main__":
    _demo()
