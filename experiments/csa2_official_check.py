"""Compare the repository's CSA2 surrogate with the published V4.1-Flash config.

This does not try to reverse-engineer the 890 B/token packing. It checks the
parameters that are actually disclosed and prints where the current wrapper
differs from the released 40-layer model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linattn.core import AttnConfig  # noqa: E402
from linattn.kv.format import PRESETS  # noqa: E402
from linattn.registry import build  # noqa: E402
from linattn.reporting import save_report  # noqa: E402


OFFICIAL = {
    "layers": 40,
    "encoder_layers": 20,
    "decoder_layers": 20,
    "attention_heads": 64,
    "head_dim": 512,
    "kv_heads": 1,
    "indexer_heads": 32,
    "indexer_dim": 128,
    "index_topk": 512,
    "swa_window": 128,
    "candidate_blocks": 2048,
    "candidate_block_size": 8,
    "main_kv_bits": 4,
    "swa_kv_bits": 8,
    "reported_bytes_per_token": 890.0,
}


def _per_layer_core(t: int) -> float:
    reads = OFFICIAL["index_topk"] + OFFICIAL["swa_window"]
    return 4.0 * OFFICIAL["attention_heads"] * t * reads * OFFICIAL["head_dim"]


def _wrapper(compress: int, t: int):
    cfg = AttnConfig(
        n_heads=OFFICIAL["attention_heads"],
        n_kv_heads=OFFICIAL["kv_heads"],
        head_dim=OFFICIAL["head_dim"],
        block_size=8,
        window_size=OFFICIAL["swa_window"],
        topk_blocks=OFFICIAL["index_topk"],
        topk_tokens=OFFICIAL["index_topk"],
    )
    return build("csa2", cfg.with_(extras={
        "compress_ratio": compress,
        "bits": OFFICIAL["main_kv_bits"],
        "quant_group": 16,
        "topk_groups": OFFICIAL["index_topk"],
        "window": OFFICIAL["swa_window"],
        "indexer_heads": OFFICIAL["indexer_heads"],
        "indexer_dim": OFFICIAL["indexer_dim"],
    }))


def main() -> int:
    rows = []
    for item, value, unit, note in (
        ("official layers", OFFICIAL["layers"], "layers", "20 encoder + 20 decoder CED"),
        ("official attention heads", OFFICIAL["attention_heads"], "heads", "query heads"),
        ("official head dim", OFFICIAL["head_dim"], "dims", "main attention"),
        ("official indexer", f"{OFFICIAL['indexer_heads']} x {OFFICIAL['indexer_dim']}", "heads x dim", "indexer Q/K width"),
        ("official sparse top-k", OFFICIAL["index_topk"], "entries/query", "main KV entries"),
        ("official SWA window", OFFICIAL["swa_window"], "tokens", "FP8 SWA"),
        ("official main KV precision", "FP4 E2M1 + E4M3 scale/16", "format", "QAT"),
        ("official reported global KV", OFFICIAL["reported_bytes_per_token"], "B/token", "whole model target"),
    ):
        rows.append({"check": "official", "item": item, "value": value, "unit": unit, "note": note})

    current_default_layer = build("csa2", AttnConfig())
    rows.extend([
        {"check": "wrapper_default", "item": "compression ratio", "value": current_default_layer.compress_ratio,
         "unit": "m", "note": "official is m=2 encoder / m=1 decoder"},
        {"check": "wrapper_default", "item": "attention heads", "value": current_default_layer.cfg.n_heads,
         "unit": "heads", "note": "official is 64"},
        {"check": "wrapper_default", "item": "head dim", "value": current_default_layer.cfg.head_dim,
         "unit": "dims", "note": "official is 512"},
        {"check": "wrapper_default", "item": "KV heads", "value": current_default_layer.cfg.n_kv_heads,
         "unit": "heads", "note": "official config is 1"},
        {"check": "wrapper_default", "item": "indexer", "value": f"{current_default_layer.n_indexer_heads} x {current_default_layer.indexer_dim}",
         "unit": "heads x dim", "note": "official is 32 x 128"},
        {"check": "wrapper_default", "item": "top-k groups", "value": current_default_layer.topk_groups,
         "unit": "groups", "note": "official is 512 KV entries"},
        {"check": "wrapper_default", "item": "window", "value": current_default_layer.window,
         "unit": "tokens", "note": "official is 128; wrapper defaults to raw FP16 window"},
    ])

    for t in (32768, 131072, 1048576):
        core = _per_layer_core(t)
        rows.append({"check": "analytic", "item": f"core FLOPs @T={t}",
                     "value": core, "unit": "FLOPs/layer",
                     "note": "4·H·T·(512+128)·d before indexer/CED/projection"})
        for m in (2, 1):
            layer = _wrapper(m, t)
            rows.append({"check": "wrapper_official_dims", "item": f"wrapper FLOPs @T={t}, m={m}",
                         "value": layer.flops(t), "unit": "FLOPs/layer",
                         "note": "includes current surrogate indexer estimate"})

    for name in ("csa2_reported", "csa2", "csa2_reach890"):
        fmt = PRESETS[name]
        rows.append({
            "check": "kv_preset",
            "item": f"{name} @61-layer preset convention",
            "value": fmt.bytes_for(1_000_000, 61) / 1_000_000,
            "unit": "B/token",
            "note": (fmt.note or "") + "; not a recomputation of the official 40-layer packing",
        })

    path = save_report(
        "csa2_official_config",
        rows,
        title="CSA2 official V4.1-Flash configuration vs the repository surrogate",
        columns=["check", "item", "value", "unit", "note"],
        notes=(
            "The 1107 tag was not found in the official public metadata as of "
            "2026-10-20; the public release is V4.1-Flash and its 1017 minor update. "
            "Official values come from the technical report and public config.json. "
            "The wrapper is a single-layer surrogate: it does not model the "
            "Full/Reindex/Reuse schedule, CED half-depth prefill, FP8 SWA, or the "
            "exact 890 B/token packing. The 890 preset is a target, not a derivation. "
            "The official schedule performs indexing in only 8 of 38 CSA2 layers "
            "(3 encoder Full, 1 decoder Full, 4 decoder Reindex); the other 30 are "
            "Reuse and run no indexer. The wrapper charges an indexer on every call "
            "and searches the full compressed prefix, so its cost is a loose upper "
            "bound rather than the official model cost."
        ),
    )
    print(f"wrote {path}")
    for r in rows:
        if r["check"] in ("analytic", "wrapper_official_dims", "kv_preset"):
            print(f"{r['check']:<22} {r['item']:<34} {r['value']:>14} {r['unit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
