"""Tiny report writer: every benchmark emits JSON (machine) + Markdown (human).

Keeping the two side by side matters more than it sounds: the numbers in
``docs/`` are supposed to be traceable to a file in ``reports/``, and the markdown
header always records the machine and the backends, so a number can never be
attributed to the wrong environment.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

from linattn.kernels import describe_backends

__all__ = ["save_report", "reports_dir"]


def reports_dir(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[2] / "reports"
    base.mkdir(parents=True, exist_ok=True)
    return base


def save_report(
    name: str,
    records: Sequence[Dict],
    *,
    title: str,
    columns: Sequence[str],
    notes: str = "",
    root: Path | None = None,
) -> Path:
    """Write ``reports/<name>.json`` and ``reports/<name>.md``."""
    base = reports_dir(root)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "name": name,
        "generated_utc": stamp,
        "environment": describe_backends(),
        "records": list(records),
    }
    json_path = base / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [f"# {title}", "", f"*Generated {stamp} by `{Path(__file__).name.rsplit('/', 1)[0]}` on "
             f"{payload['environment']['platform']} (torch {payload['environment']['torch']}, "
             f"cuda={payload['environment']['cuda']}, triton={payload['environment']['triton_installed']}).*", ""]
    if notes:
        lines += [notes, ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for rec in records:
        cells = []
        for col in columns:
            value = rec.get(col, "")
            if isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    md_path = base / f"{name}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path
