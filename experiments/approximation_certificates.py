"""Auditable error certificates for attention approximations.

The functions in this module are deliberately small.  They distinguish:

* the attention distribution ``p`` from the value-weighted output ``p @ V``;
* an exact sparse-support decomposition from its conservative norm bound;
* key-induced and value-induced errors under cache quantisation.

They are research instruments, not claims that a certificate composes through a
trained transformer.  Propagation must be measured separately.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _as_last_dim(x: torch.Tensor) -> torch.Tensor:
    return x.float().reshape(-1, x.shape[-1])


def total_variation(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-row total variation distance, returned with the leading dimensions."""
    p = p.float()
    q = q.float()
    return 0.5 * (p - q).abs().sum(-1)


def relative_l2(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-row relative L2 error, returned with the leading dimensions."""
    delta = (reference.float() - estimate.float()).square().sum(-1).sqrt()
    denom = reference.float().square().sum(-1).sqrt().clamp_min(eps)
    return delta / denom


def sparse_support_certificate(
    p: torch.Tensor,
    v: torch.Tensor,
    support: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """Exact omitted contribution and a conservative bound for hard selection.

    ``p`` and ``support`` have shape ``(..., T)``; ``v`` has shape
    ``(..., T, D)``.  The selected operator renormalises the retained weights.
    Its error is

        O - O_S = (1-a) (mu_omitted - mu_selected),

    where ``a`` is retained attention mass.  The function returns this exact
    vector as well as the standard ``2(1-a)V_max`` bound.
    """
    support = support.to(device=p.device, dtype=torch.bool)
    if support.shape != p.shape:
        raise ValueError("support must have the same shape as p")
    if v.shape[-2] != p.shape[-1]:
        raise ValueError("v must have the same key dimension as p")
    vv = v
    while vv.ndim < p.ndim + 1:
        vv = vv.unsqueeze(-3)

    retained = (p * support).sum(-1)
    selected = support & (p > 0)
    p_selected_weight = p * selected
    omitted_weight = p * (~selected)

    p_ref = p
    p_selected = p_selected_weight / retained.clamp_min(eps).unsqueeze(-1)
    out_ref = torch.matmul(p_ref.unsqueeze(-2), vv).squeeze(-2)
    out_selected = torch.matmul(p_selected.unsqueeze(-2), vv).squeeze(-2)

    selected_mean = torch.matmul(
        (p_selected_weight / retained.clamp_min(eps).unsqueeze(-1)).unsqueeze(-2), vv
    ).squeeze(-2)
    omitted_mass = (1.0 - retained).clamp_min(0.0)
    omitted_mean = torch.matmul(
        (omitted_weight / omitted_mass.clamp_min(eps).unsqueeze(-1)).unsqueeze(-2), vv
    ).squeeze(-2)

    exact_omitted = omitted_mass.unsqueeze(-1) * (omitted_mean - selected_mean)
    # The identity is undefined when nothing is omitted; define the empty term as zero.
    exact_omitted = torch.where(
        (omitted_mass > eps).unsqueeze(-1), exact_omitted, torch.zeros_like(exact_omitted)
    )
    error = out_ref - out_selected
    value_max = vv.float().norm(dim=-1).amax(-1).expand(p.shape[:-1])
    bound = 2.0 * omitted_mass * value_max
    error_norm = error.norm(dim=-1)
    return {
        "retained_mass": retained,
        "omitted_mass": omitted_mass,
        "out_ref": out_ref,
        "out_selected": out_selected,
        "exact_omitted": exact_omitted,
        "error": error,
        "error_norm": error_norm,
        "relative_error": relative_l2(out_ref, out_selected),
        "value_max": value_max,
        "bound": bound,
        "bound_ratio": error_norm / bound.clamp_min(eps),
        "attention_tv": omitted_mass,
    }


def softmax_logit_perturbation_bound(logit_error: torch.Tensor) -> torch.Tensor:
    """Bound TV(softmax(z), softmax(z+delta)) from max_j |delta_j|.

    If ``eta = max_j |delta_j|``, every likelihood ratio lies in
    ``[exp(-2 eta), exp(2 eta)]``.  The extremal two-point distributions give
    ``TV <= tanh(eta)``.
    """
    eta = logit_error.float().abs().amax(-1)
    return torch.tanh(eta)


def quantized_kv_certificate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_quant: torch.Tensor,
    v_quant: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Decompose KV-quantisation error into key-induced and value-induced terms.

    Inputs use ``(..., T, D)`` tensors and already share the same head layout.
    The passed tensors are expected to be the *effective* q/k/v after any GQA
    expansion.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    z = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    z_quant = torch.matmul(q.float(), k_quant.float().transpose(-1, -2)) * scale
    p = torch.softmax(z, -1)
    p_quant = torch.softmax(z_quant, -1)
    out_ref = torch.matmul(p, v.float())
    key_only = torch.matmul(p_quant, v.float())
    out_quant = torch.matmul(p_quant, v_quant.float())

    eta = (z_quant - z).abs().amax(-1)
    tv = total_variation(p, p_quant)
    tv_bound = torch.tanh(eta)
    value_max = v.float().norm(dim=-1).amax(-1)
    value_error = (v_quant.float() - v.float()).norm(dim=-1).amax(-1)

    key_error = key_only - out_ref
    value_error_vector = out_quant - key_only
    total_error = out_quant - out_ref
    key_bound = 2.0 * tv_bound * value_max.unsqueeze(-1)
    value_bound = value_error.unsqueeze(-1).expand_as(key_bound)
    return {
        "attention_tv": tv,
        "attention_tv_bound": tv_bound,
        "key_logit_error_max": eta,
        "key_error": key_error,
        "value_error": value_error_vector,
        "total_error": total_error,
        "key_bound": key_bound,
        "value_bound": value_bound,
        "total_bound": key_bound + value_bound,
        "relative_error": relative_l2(out_ref, out_quant),
    }


def task_aligned_local_error(
    gradient: torch.Tensor,
    local_error: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """First-order loss effect of a local error vector.

    For a scalar loss ``L`` and an intervened tensor ``a``, ``g = dL/da`` is
    evaluated on the dense reference trajectory.  ``g . delta_a`` is the
    first-order predicted change in loss and already includes the downstream
    Jacobian.  It is a local diagnostic, not a finite-error guarantee.
    """
    g = gradient.float().reshape(-1, gradient.shape[-1])
    e = local_error.float().reshape(-1, local_error.shape[-1])
    if g.shape != e.shape:
        raise ValueError("gradient and local_error must have the same shape")
    projected = (g * e).sum(-1)
    g_norm = g.norm(dim=-1).clamp_min(1e-12)
    e_norm = e.norm(dim=-1)
    cosine = projected / (g_norm * e_norm.clamp_min(1e-12))
    return {
        "predicted_loss_delta": projected,
        "absolute_effect": projected.abs(),
        "forecast_norm": projected.abs() / g_norm,
        "alignment": cosine,
    }


def distribution_value_nonidentifiability() -> Dict[str, torch.Tensor]:
    """Construct two errors with identical TV but different output error.

    The first distribution pair swaps equal mass between two keys whose values
    coincide; the second swaps the same mass between orthogonal values.  This is
    a minimal control for the claim that attention-distribution error identifies
    value-weighted output error.
    """
    p = torch.tensor([0.5, 0.5], dtype=torch.float64)
    q = torch.tensor([0.9, 0.1], dtype=torch.float64)
    same_values = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    orthogonal_values = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    out_ref_same = p @ same_values
    out_alt_same = q @ same_values
    out_ref_orth = p @ orthogonal_values
    out_alt_orth = q @ orthogonal_values
    return {
        "attention_tv": total_variation(p, q),
        "same_value_error": (out_ref_same - out_alt_same).norm(),
        "orthogonal_value_error": (out_ref_orth - out_alt_orth).norm(),
    }


if __name__ == "__main__":  # pragma: no cover - small demonstration
    print(distribution_value_nonidentifiability())
