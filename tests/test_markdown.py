"""Documentation hygiene: no patch artefacts, no broken table rows.

Two real incidents motivated this file: a survey section was inserted with literal
``+`` prefixes (which also swallowed the next heading), and the same happened again
when a KV-layer section was written from a patch draft. Both are invisible in a diff
unless you render the page, so they get a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGES = sorted(DOCS.glob("*.md")) + sorted((DOCS / "variants").glob("*.md")) + [
    Path(__file__).resolve().parents[1] / "README.md"
]


@pytest.mark.parametrize("path", PAGES, ids=lambda p: str(Path(p).relative_to(Path(p).parents[2])))
def test_no_patch_markers(path):
    text = Path(path).read_text(encoding="utf-8")
    offenders = [i + 1 for i, line in enumerate(text.split("\n")) if line.startswith("+")]
    assert not offenders, f"{path}: stray '+' patch markers on lines {offenders[:5]}"


@pytest.mark.parametrize("path", PAGES, ids=lambda p: str(Path(p).relative_to(Path(p).parents[2])))
def test_headings_are_not_glued_to_text(path):
    """A heading must be its own line: 'text ## Heading' renders as a paragraph."""
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n")):
        stripped = line.strip()
        if stripped.startswith("#"):
            assert line.startswith("#"), f"{path}:{i + 1}: heading indented or glued: {line[:60]!r}"
