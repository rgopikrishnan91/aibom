"""Truth-table tests for check_compat."""
from __future__ import annotations

from aikaboom.plugins.license_compat.engine import CompatVerdict, check_compat


def test_compatible_single_upstream(tiny_matrix):
    v = check_compat("mit", frozenset({"apache-2.0"}), tiny_matrix)
    assert v.status == "compatible"
    assert v.incompatible_with == frozenset()


def test_violation_single_upstream(tiny_matrix):
    v = check_compat("apache-2.0", frozenset({"gpl-3.0"}), tiny_matrix)
    assert v.status == "violation"
    assert v.incompatible_with == frozenset({"gpl-3.0"})


def test_violation_partial_block(tiny_matrix):
    # mit is OK downstream of apache-2.0 (Yes), but NOT of cc-by-nc-4.0 (No)
    v = check_compat("mit", frozenset({"apache-2.0", "cc-by-nc-4.0"}), tiny_matrix)
    assert v.status == "violation"
    assert v.incompatible_with == frozenset({"cc-by-nc-4.0"})


def test_unknown_upstream(tiny_matrix):
    v = check_compat("mit", frozenset({"UNKNOWN"}), tiny_matrix)
    assert v.status == "unknown_upstream"


def test_unknown_downstream(tiny_matrix):
    v = check_compat(None, frozenset({"apache-2.0"}), tiny_matrix)
    assert v.status == "unknown_downstream"


def test_compatible_same_license(tiny_matrix):
    v = check_compat("apache-2.0", frozenset({"apache-2.0"}), tiny_matrix)
    assert v.status == "compatible"


def test_missing_data_when_downstream_not_in_matrix(tiny_matrix):
    v = check_compat("never-heard-of-this", frozenset({"apache-2.0"}), tiny_matrix)
    assert v.status == "missing_data"


def test_empty_upstream_set_is_compatible_trivially(tiny_matrix):
    # No upstream constraints => any downstream is compatible.
    v = check_compat("apache-2.0", frozenset(), tiny_matrix)
    assert v.status == "compatible"
