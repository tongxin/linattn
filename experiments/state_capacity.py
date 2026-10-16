"""How many associations can a fixed-size state actually hold?

The claim in the survey is that compressed state cannot do exact recall, and that
hybrids keep a few attention layers precisely to buy retrieval back.  Training a
small MQAR model does *not* demonstrate that: at the scale a laptop can train, every
architecture we tried solves the task (see ``reports/hybrid_recall.md``).  This
experiment isolates the capacity question with no training at all:

1. write ``N`` random key->value associations into a delta-rule state using the exact
   recurrence, with keys L2-normalised as every delta-rule model does;
2. read them back with the same keys and measure cosine fidelity against the values;
3. repeat for several state sizes and association counts.

The result is the capacity curve: fidelity survives while the associations stay
separable in key space, then collapses.  An attention layer has no such failure mode
at all -- which is why the frontier hybrids keep them.

Run::

    python experiments/state_capacity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linattn.reporting import save_report  # noqa: E402


def store_and_query_state(keys: torch.Tensor, values: torch.Tensor, *, beta: float = 0.99, decay: float = 0.99):
    """Write ``N`` associations with the delta rule, then read them back.

    ``keys``: ``(B, N, d_state)``; ``values``: ``(B, N, d_value)``.
    Returns ``(retrieved (B, N, d_value), state_norm)``.

    The write and read phases are *separated*: reading a key immediately after
    writing it would always return its own value (it is the most recent entry), and
    the experiment would report perfect fidelity for any state size.  Real recall
    means reading keys back after all the interference has accumulated.
    """
    b, n, dk = keys.shape
    dv = values.shape[-1]
    k = F.normalize(keys, dim=-1)
    state = torch.zeros(b, dk, dv, dtype=keys.dtype)
    for i in range(n):
        state = state * decay
        state = state + beta * torch.einsum("bd,be->bde", k[:, i], values[:, i])
    # read phase
    retrieved = torch.einsum("bnd,bde->bne", k, state)
    return retrieved, float(state.norm())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", nargs="*", type=int, default=[4, 16, 64, 256, 1024])
    parser.add_argument("--states", nargs="*", type=int, default=[8, 32, 128])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--d-value", type=int, default=64, dest="d_value")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    records = []
    for n_pairs in args.counts:
        for d_state in args.states:
            keys = torch.randn(args.batch, n_pairs, d_state, dtype=torch.float64)
            values = F.normalize(torch.randn(args.batch, n_pairs, args.d_value, dtype=torch.float64), dim=-1)
            with torch.no_grad():
                retrieved, state_norm = store_and_query_state(keys, values)
            fidelity = F.cosine_similarity(retrieved, values, dim=-1).mean()
            state_floats = d_state * args.d_value
            records.append({
                "associations": n_pairs,
                "d_state": d_state,
                "state_floats": state_floats,
                "retrieval_cosine": float(fidelity),
                "state_norm": state_norm,
            })
            print(f"N={n_pairs:5d}  d_state={d_state:4d}  state_floats={state_floats:6d}  "
                  f"cos_fidelity={float(fidelity):.3f}", flush=True)

    path = save_report(
        "state_capacity",
        records,
        title="Associative-recall capacity of a fixed-size delta-rule state",
        columns=["associations", "d_state", "state_floats", "retrieval_cosine", "state_norm"],
        notes=(
            "No training involved: random key/value vectors are written through the exact "
            "delta-rule recurrence (keys L2-normalised, beta=0.99, decay=0.99) and read "
            "back with the same keys. `retrieval_cosine` is the mean cosine fidelity "
            "between the retrieved and the true value; the state holds `d_state * "
            "d_value` floats, so the curve is really about *associations per unit of "
            "state* -- the quantity a hybrid trades against exact-attention layers."
        ),
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
