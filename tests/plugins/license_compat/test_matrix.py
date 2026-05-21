"""LicenseMatrix loader + license resolver tests."""
from __future__ import annotations

import pytest

from aikaboom.plugins.license_compat.matrix import (
    LicenseMatrix,
    load_matrix,
    resolve_license,
)


def test_load_matrix_indexes_aliases(tiny_matrix):
    assert tiny_matrix.name_alias_lookup["apache 2.0"] == "apache-2.0"
    assert tiny_matrix.name_alias_lookup["apache-2.0"] == "apache-2.0"
    assert tiny_matrix.name_alias_lookup["the mit license"] == "mit"


def test_load_matrix_builds_upstream_compat_index(tiny_matrix):
    # apache-2.0 is compatible-upstream-of mit (mit lists apache-2.0 Yes)
    assert "mit" in tiny_matrix.upstream_compat_index["apache-2.0"]
    # apache-2.0 self-compat ("Same")
    assert "apache-2.0" in tiny_matrix.upstream_compat_index["apache-2.0"]
    # gpl-3.0 -> apache-2.0 is "No"
    assert "apache-2.0" not in tiny_matrix.upstream_compat_index["gpl-3.0"]


def test_load_matrix_injects_unknown_token(tiny_matrix):
    assert "UNKNOWN" in tiny_matrix.details
    assert tiny_matrix.name_alias_lookup["unknown"] == "UNKNOWN"


def test_load_matrix_reads_allowed_licenses(tiny_matrix):
    assert tiny_matrix.allowed_licenses == {"apache-2.0", "mit", "lgpl-3.0", "cc-by-nc-4.0"}


def test_load_matrix_reads_missing_licenses(tiny_matrix):
    assert tiny_matrix.missing_licenses == {"proprietary-corp-x"}


def test_resolve_license_canonicalises_alias(tiny_matrix):
    r = resolve_license("Apache 2.0", tiny_matrix)
    assert r.primary_name == "apache-2.0"
    assert r.is_unknown is False
    assert r.is_missing is False


def test_resolve_license_flags_unknown_string(tiny_matrix):
    r = resolve_license("WeirdMadeUpLic-7", tiny_matrix)
    assert r.primary_name == "UNKNOWN"
    assert r.is_unknown is True


def test_resolve_license_handles_missing_marker(tiny_matrix):
    r = resolve_license("MISSING", tiny_matrix)
    assert r.primary_name is None
    assert r.is_missing is True


def test_resolve_license_strips_other_parentheses(tiny_matrix):
    r = resolve_license("apache-2.0 (other)", tiny_matrix)
    assert r.primary_name == "apache-2.0"


def test_load_matrix_uses_bundled_defaults_when_no_path():
    # When called with no arguments, the bundled matrix is loaded.
    m = load_matrix()
    assert isinstance(m, LicenseMatrix)
    assert len(m.details) > 100  # bundled matrix has thousands
