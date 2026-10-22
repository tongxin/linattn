from __future__ import annotations

from experiments.theory_checks import quant_rows, selection_bound_rows


def test_sparse_selection_bound_is_satisfied():
    rows = selection_bound_rows()
    assert rows
    assert all(row["satisfied"] == 1.0 for row in rows)
    assert max(row["exact_decomposition_error"] for row in rows) < 1e-12
    assert max(row["implementation_parity"] for row in rows) < 1e-12


def test_kv_precision_ordering_in_the_theory_check():
    rows = quant_rows()
    by_key = {}
    for row in rows:
        by_key.setdefault((row["input"], row["seed"]), {})[row["bits"]] = row["rel_l2_error"]
    for errors in by_key.values():
        assert errors[16] == 0.0
        assert errors[8] < errors[4]


def test_quantization_certificate_holds_and_decomposition_is_exact():
    rows = quant_rows()
    assert all(row["certificate_satisfied"] == 1.0 for row in rows)
    assert max(row["exact_decomposition_error"] for row in rows) < 1e-6
