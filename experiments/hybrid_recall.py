"""How much exact attention does a hybrid need?  A small, self-contained answer.

The claim in the survey is that compressed state cannot do exact associative
recall, and that hybrids keep a minority of exact-attention layers precisely to
buy retrieval back.  This experiment makes that claim measurable on a CPU:

* task: multi-query associative recall (MQAR), the standard synthetic probe,
  ``k1 v1 k2 v2 ... ? k1 -> v1`` with 4 pairs per sequence;
* models: pure dense attention, hybrids with 1-in-N layers dense, and a
  linear-only stack, using the three linear-attention kinds implemented here
  (SSD, Gated DeltaNet, KDA);
* reported: final MQAR accuracy and the persistent state size per sequence.

Run::

    python experiments/hybrid_recall.py --steps 300

Runtime is a few minutes on a laptop CPU.  Results land in
``reports/hybrid_recall.{json,md}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linattn.reporting import save_report  # noqa: E402
from linattn.variants.hybrid import HybridConfig, HybridLM, evaluate_mqar, make_mqar_batch, train_mqar  # noqa: E402
import torch  # noqa: E402


CONFIGS = [
    ("dense_attention_only", dict(kind="gdn", attn_period=1, use_linear_layers=False)),
    ("hybrid_1in2_gdn", dict(kind="gdn", attn_period=2)),
    ("hybrid_1in4_gdn", dict(kind="gdn", attn_period=4)),
    ("hybrid_1in4_kda", dict(kind="kda", attn_period=4)),
    ("hybrid_1in4_ssd", dict(kind="ssd", attn_period=4)),
    ("linear_only_gdn", dict(kind="gdn", attn_period=0, use_linear_layers=True)),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=96, dest="d_model")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--d-state", type=int, default=32, dest="d_state")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    records = []
    for name, overrides in CONFIGS:
        cfg = HybridConfig(
            d_model=args.d_model,
            n_layers=args.layers,
            d_state=args.d_state,
            seq_len=args.seq,
            vocab_size=64,
            **overrides,
        )
        result = train_mqar(cfg, steps=args.steps, batch=args.batch, seed=args.seed)
        model = HybridLM(cfg)
        records.append({
            "model": name,
            "accuracy": result["accuracy"],
            "loss": result["loss"],
            "state_bytes_per_seq": result["state_bytes"],
            "attention_layers": sum(1 for k in model.kinds if k == "attn"),
            "linear_layers": sum(1 for k in model.kinds if k == "linear"),
            "kind": cfg.kind,
        })
        print(f"{name:<24} acc={result['accuracy']:.3f}  state={result['state_bytes']} B"
              f"  layers={model.kinds.count('attn')}attn/{model.kinds.count('linear')}lin")

    path = save_report(
        "hybrid_recall",
        records,
        title="MQAR accuracy vs. persistent state size for hybrid stacks",
        columns=["model", "accuracy", "state_bytes_per_seq", "attention_layers", "linear_layers", "kind"],
        notes=(
            f"MQAR with 4 pairs per sequence, seq_len={args.seq}, {args.layers} layers, "
            f"d_model={args.d_model}, {args.steps} training steps, one seed. "
            "`state_bytes_per_seq` counts the persistent state only (KV for attention "
            "layers is counted per token, the linear state per sequence), which is why "
            "a hybrid's advantage grows with sequence length.\n\n"
            "**Read this table together with `state_capacity.md`.** At the scale a "
            "laptop can train (a few hundred thousand parameters, tens of pairs), "
            "every architecture here solves MQAR, so the training comparison does "
            "*not* discriminate between families -- which is itself worth knowing, "
            "because MQAR is routinely cited as evidence that it does. The capacity "
            "limit shows up without training instead: see the deterministic "
            "`experiments/state_capacity.py` curve, where recall of a fixed-size state "
            "degrades as the number of associations grows. Getting MQAR to "
            "*separate* attention from linear attention needs either scale we cannot "
            "reach here or a much harder task (many more pairs, shorter training, "
            "smaller state); we prefer to say that than to report a curve that looks "
            "like a result."
        ),
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
