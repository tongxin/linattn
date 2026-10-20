"""The CLI catalogue must list everything that exists.

GDN / KDA / GDN-2 existed in ``variants/recurrent.py`` from the start but were
absent from ``python -m linattn.cli variants``, because the CLI only iterated the
attention-variant registry.  These tests pin the two catalogues together.
"""

from __future__ import annotations

import pytest

from linattn import cli
from linattn.kernels.mlx.variants import SUPPORTED as MLX_SUPPORTED
from linattn.registry import (
    KERNEL_VARIANTS,
    MIXERS,
    VARIANTS,
    build,
    build_any,
    build_mixer,
    describe,
    describe_kernel,
    describe_mixer,
)
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


def test_kernel_registry_covers_every_mlx_variant():
    assert set(KERNEL_VARIANTS) == {f"mlx_{name}" for name in MLX_SUPPORTED}
    assert all(describe_kernel(name)["citation"] for name in KERNEL_VARIANTS)
    for name in KERNEL_VARIANTS:
        assert build(name).name == name and build_any(name).name == name


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


def test_cost_table_is_aligned_plain_text(capsys):
    assert cli.main(["cost", "--seq", "1024", "--variants", "dense", "mla"]) == 0
    out = capsys.readouterr().out
    assert "|" not in out and "`" not in out
    assert out.splitlines()[:4] == [
        "variant  family      state @T (MiB)  prefill attn FLOPs @T (G)  KV reads/query",
        "-------  ----------  --------------  -------------------------  --------------",
        "dense    dense                  2.0                        2.1           1,024",
        "mla      compressed             0.0                        1.1           1,024",
    ]


def test_cost_lists_every_registered_variant(capsys):
    assert cli.main(["cost", "--seq", "1024"]) == 0
    out = capsys.readouterr().out
    for name in VARIANTS:
        assert name in out


def test_cost_accepts_a_backend_variant_only_when_requested(capsys):
    assert cli.main(["cost", "--seq", "1024"]) == 0
    out = capsys.readouterr().out
    assert "mlx_" not in out

    assert cli.main(["cost", "--seq", "1024", "--variants", "mlx_quest"]) == 0
    out = capsys.readouterr().out
    assert "mlx_quest" in out
    assert "mlx_" not in out.replace("mlx_quest", "")


def test_cost_includes_every_state_mixer(capsys):
    assert cli.main(["cost", "--seq", "1024", "--variants", "dense"]) == 0
    out = capsys.readouterr().out
    assert "state mixers" in out
    rows = {line.split()[0]: line for line in out.splitlines() if line and line.split()[0] in MIXERS}
    assert set(rows) == set(MIXERS)
    for kind, row in rows.items():
        assert float(row.split()[-1]) > 0, f"{kind} missing its decode FLOPs/token"


def test_efficiency_table_can_still_render_markdown():
    from linattn.metrics import efficiency_table

    table = efficiency_table([build("dense")], 1024, markdown=True)
    assert table.splitlines()[0] == "| variant | family | state @T (MiB) | prefill attn FLOPs @T (G) | KV reads/query |"
    assert table.splitlines()[2].startswith("| `dense` | dense |")


def test_attention_registry_still_builds_everything():
    for name in VARIANTS:
        variant = build(name)
        assert variant.name == name
