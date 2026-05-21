"""Edge cases for matrix loader and license normaliser.

Covers the defensive branches in ``matrix.py`` that the happy-path tests
don't reach: non-string inputs, ``MISSING`` markers, list-typed license
fields, and stringified-list license fields.
"""
from __future__ import annotations

import json

from aikaboom.plugins.license_compat.matrix import (
    _clean,
    load_matrix,
    normalize_license_field,
    resolve_license,
)


def test_clean_returns_none_for_non_string():
    assert _clean(123) is None  # type: ignore[arg-type]


def test_clean_returns_none_for_missing_marker():
    assert _clean("MISSING") is None
    assert _clean("missing") is None


def test_clean_returns_none_for_blank():
    assert _clean("   ") is None


def test_resolve_license_non_string_returns_empty():
    # The matrix arg isn't touched on this branch, so any LicenseMatrix works.
    from aikaboom.plugins.license_compat.matrix import LicenseMatrix
    m = LicenseMatrix(
        name_alias_lookup={}, details={}, upstream_compat_index={},
        allowed_licenses=frozenset(), missing_licenses=frozenset(),
    )
    r = resolve_license(42, m)  # type: ignore[arg-type]
    assert r.primary_name is None
    assert r.is_unknown is False
    assert r.is_missing is False


def test_resolve_license_blank_string_returns_unknown_none(tiny_matrix):
    r = resolve_license("   ", tiny_matrix)
    assert r.primary_name is None
    assert r.is_missing is False
    assert r.is_unknown is False


def test_normalize_license_field_none_returns_empty():
    assert normalize_license_field(None) == []


def test_normalize_license_field_list_filters_non_strings():
    assert normalize_license_field(["mit", 42, "apache-2.0"]) == ["mit", "apache-2.0"]


def test_normalize_license_field_missing_marker_returns_empty():
    assert normalize_license_field("MISSING") == []
    assert normalize_license_field("   ") == []


def test_normalize_license_field_stringified_list():
    assert normalize_license_field("['mit', 'apache-2.0']") == ["mit", "apache-2.0"]


def test_normalize_license_field_bad_stringified_list_falls_through():
    # Doesn't end with ']' — skips the literal_eval block entirely.
    assert normalize_license_field("[unbalanced") == ["[unbalanced"]


def test_normalize_license_field_malformed_bracketed_falls_through():
    # Brackets match but literal_eval raises — exception handler returns [s].
    # ``[broken,]`` parses as a list with a bare Name, which literal_eval
    # rejects with ValueError.
    assert normalize_license_field("[broken,]") == ["[broken,]"]


def test_normalize_license_field_non_string_non_list_returns_empty():
    assert normalize_license_field(123) == []  # type: ignore[arg-type]


def test_load_matrix_skips_non_dict_license_entry(tmp_path):
    """Matrix entries that aren't dicts are silently skipped."""
    matrix_file = tmp_path / "m.json"
    matrix_file.write_text(json.dumps({
        "licenses": [
            "not-a-dict",                              # skipped (line 70)
            {"name": 123, "category": "X"},            # skipped (line 73)
            {"name": "mit", "aliases": [], "category": "PERMISSIVE",
             "compatibilities": []},
        ],
    }))
    allowed = tmp_path / "a.json"
    allowed.write_text("[]")
    missing = tmp_path / "x.json"
    missing.write_text("{}")
    m = load_matrix(matrix_path=matrix_file, allowed_path=allowed, missing_path=missing)
    # mit was kept, the other two were skipped, and UNKNOWN was auto-injected.
    assert "mit" in m.details
    assert "UNKNOWN" in m.details
    assert len(m.details) == 2
