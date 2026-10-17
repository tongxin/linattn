"""The CLI catalogue must list everything that exists.

GDN / KDA / GDN-2 existed in ``variants/recurrent.py`` from the start but were
absent from ``python -m linattn.cli variants``, because the CLI only iterated the
attention-variant registry.  These tests pin the two catalogues together.
"""

from __future__ import annotations

import pytest

from linattn import cli
from linattn.registry import MIXERS, VARIANTS, build, build_mixer, describe, describe_mixer
from linattn.variants.recurrent import STATE_KINDS


def test_variant_listing_includes_every_registered_variant(capsys):
    assert cli.main(["variants"]) == 0
    out = capsys.readouterr().out
    for name in VARIANTS:
        assert name in out, f"{name} missing from `cli variants`"
    for kind in MIXERS:
        assert kind in out, f"state mixer {kind} missing from `cli variants`"
    assert "state mixers" in out and "hybrid stack" in out


def test_registries_cover_every_implemented_kind():
    """No implementation without a registry entry (the bug this test exists for)."""
    assert set(MIXERS) == set(STATE_KINDS)
    assert all(describe_mixer(k)["citation"] for k in MIXERS)
    assert all(describe(v)["citation"] for v in VARIANTS)


@pytest.mark.parametrize("kind", list(STATE_KINDS))
def test_build_mixer_works_for_every_kind(kind):
    layer = build_mixer(kind, d_model=32, n_heads=2, d_state=8, chunk=8)
    assert layer.cfg.kind == kind
    import torch

    y, state = layer.forward_chunked(torch.randn(1, 8, 32))
    assert y.shape == (1, 8, 32) and state.dim() == 4


def test_mixer_command_reports_state_and_cost(capsys):
    assert cli.main(["mixers", "--d-model", "256", "--heads", "4", "--d-state", "16"]) == 0
    out = capsys.readouterr().out
    for kind in MIXERS:
        assert kind in out
    assert "state floats" in out


def test_attention_registry_still_builds_everything():
    for name in VARIANTS:
        variant = build(name)
        assert variant.name == name
