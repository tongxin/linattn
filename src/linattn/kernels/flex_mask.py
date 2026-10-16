"""Express variant selections as ``flex_attention`` block masks.

PyTorch's FlexAttention takes a ``BlockMask``: a coarse, block-level description
of which query blocks may read which key blocks.  The compiler then generates a
fused kernel for that mask, which is exactly the interface a selection algorithm
needs -- the algorithm's job is to choose the mask well, and the kernel's job is
to execute it.

``to_block_mask`` converts a variant's token-level boolean selection into a
``BlockMask`` (allowing a query block to read every key block that *some* query in
it needs) and ``check_block_mask`` verifies that the block mask is a superset of
the variant's own mask -- i.e. that expressing the pattern in the flex interface
did not silently drop reads.  The check is portable, so it runs on CPU even
though the generated kernel does not.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

__all__ = ["flex_available", "to_block_mask", "check_block_mask", "mask_to_block_mask"]


def flex_available() -> bool:
    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401

        return True
    except Exception:
        return False


def mask_to_block_mask(mask: torch.Tensor, q_block: int, k_block: int):
    """Reduce a token mask ``(B,H,T_q,T_k)`` to a block mask ``(B,H,n_qb,n_kb)``.

    A block is kept if *any* query in it reads *any* key in it, matching
    FlexAttention's granularity (it cannot express sub-block sparsity).
    """
    b, h, t_q, t_k = mask.shape
    n_qb = (t_q + q_block - 1) // q_block
    n_kb = (t_k + k_block - 1) // k_block
    padded = torch.zeros(b, h, n_qb * q_block, n_kb * k_block, dtype=mask.dtype, device=mask.device)
    padded[..., :t_q, :t_k] = mask
    return padded.view(b, h, n_qb, q_block, n_kb, k_block).any(-3).any(-1)


def to_block_mask(mask: torch.Tensor, q_block: int = 64, k_block: int = 64):
    """Build a real ``torch.nn.attention.flex_attention.BlockMask`` from a token mask."""
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    reduced = mask_to_block_mask(mask, q_block, k_block)
    b, h, n_qb, n_kb = reduced.shape

    def mask_mod(batch, head, q_idx, kv_idx):
        return reduced[batch, head, q_idx, kv_idx]

    return create_block_mask(mask_mod, B=b, H=h, Q_LEN=n_qb, KV_LEN=n_kb, device=mask.device)


def check_block_mask(mask: torch.Tensor, reduced: torch.Tensor, q_block: int, k_block: int) -> Tuple[bool, float]:
    """Verify the block reduction is a *superset* of the token mask.

    Returns ``(ok, coverage)`` where coverage is the kept-block fraction; a
    coverage well above the token-level kept fraction means the block interface is
    wasting reads, which is precisely the co-design trade-off worth reporting.
    """
    full = reduced.repeat_interleave(q_block, dim=-2).repeat_interleave(k_block, dim=-1)
    full = full[..., : mask.shape[-2], : mask.shape[-1]]
    ok = bool((mask & ~full).sum() == 0)
    return ok, float(reduced.float().mean())
