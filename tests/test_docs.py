"""Keep the documentation honest: every variant must have a page and a citation."""

from __future__ import annotations

from pathlib import Path

import pytest

from linattn.registry import VARIANTS, describe

DOCS = Path(__file__).resolve().parents[1] / "docs"


@pytest.mark.parametrize("name", list(VARIANTS))
def test_variant_has_doc_page(name):
    page = DOCS / "variants" / f"{name}.md"
    assert page.exists(), f"missing docs/variants/{name}.md (run scripts/gen_variant_docs.py)"
    text = page.read_text(encoding="utf-8")
    assert len(text) > 400, f"{page} looks empty"
    assert "http" in text, f"{page} has no citation URL"


@pytest.mark.parametrize("name", list(VARIANTS))
def test_variant_has_citation_in_code(name):
    meta = describe(name)
    assert meta["citation"], f"{name} has no paper= citation in its class"


def test_top_level_docs_exist():
    for name in ("survey.md", "evaluation-evidence.md", "frontier-gap.md", "kernels.md",
                 "kv-layer.md", "academic-attention.md", "analytic-flops.md",
                 "csa2-approximation.md",
                 "csa2-official-config.md",
                 "triton-on-apple.md", "mlx.md", "roadmap.md", "citations.bib"):
        assert (DOCS / name).exists(), f"missing docs/{name}"


def test_approximation_research_docs_exist():
    for name in (
        "approximation-research-plan.md",
        "approximation-metrics.md",
        "approximation-theory-map.md",
        "attention-approximation-report.md",
        "attention-evaluation-protocol.md",
        "error-propagation.md",
    ):
        assert (DOCS / "research" / name).exists(), f"missing docs/research/{name}"


@pytest.mark.parametrize("kind", ["ssd", "gdn", "kda", "gdn2"])
def test_state_mixers_page_documents_each_kind(kind):
    """The mixers are not ``AttentionVariant``s, so they get their own page."""
    page = DOCS / "variants" / "state_mixers.md"
    assert page.exists(), "missing docs/variants/state_mixers.md (run scripts/gen_variant_docs.py)"
    text = page.read_text(encoding="utf-8")
    assert f"`{kind}`" in text, f"{kind} not documented on the state-mixers page"
    assert "arxiv.org" in text
