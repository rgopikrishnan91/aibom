"""Edge cases for the recommendation engine.

Closes gaps for empty-upstream recommendation and CC-version dedup
(prefer 4.0 over older base versions in the same family).
"""
from __future__ import annotations

import json
from collections import Counter

from aikaboom.plugins.license_compat.engine import recommend
from aikaboom.plugins.license_compat.matrix import load_matrix


def test_recommend_empty_upstreams_returns_unsolvable(tiny_matrix):
    r = recommend(frozenset(), tiny_matrix, Counter())
    assert r.is_solvable is False
    assert r.by_category == {}


def test_recommend_collapses_older_cc_versions_when_4_0_present(tmp_path):
    """When candidates contain both ``cc-by-4.0`` and ``cc-by-3.0``, the
    older 3.0 row is dropped (CC versions filter at engine.py:83-84)."""
    # Build a tiny matrix where the only upstream's compat list includes
    # multiple CC versions of the same base family.
    matrix_file = tmp_path / "m.json"
    matrix_file.write_text(json.dumps({
        "licenses": [
            {
                "name": "cc-by-4.0",
                "aliases": [], "category": "CC",
                "compatibilities": [
                    {"name": "cc-by-4.0", "compatibility": "Same"},
                ],
            },
            {
                "name": "cc-by-3.0",
                "aliases": [], "category": "CC",
                "compatibilities": [
                    {"name": "cc-by-3.0", "compatibility": "Same"},
                ],
            },
            {
                # An upstream whose compat list nominates BOTH cc-by versions.
                "name": "upstream-lic",
                "aliases": [], "category": "PERMISSIVE",
                "compatibilities": [],
            },
        ],
    }))
    allowed = tmp_path / "a.json"
    allowed.write_text(json.dumps([
        {"key": "cc-by-4.0"},
        {"key": "cc-by-3.0"},
    ]))
    missing = tmp_path / "x.json"
    missing.write_text("{}")
    m = load_matrix(matrix_path=matrix_file, allowed_path=allowed, missing_path=missing)
    # Manually inject both CCs as compatible downstreams of upstream-lic so
    # they both appear in the candidate set.
    new_idx = dict(m.upstream_compat_index)
    new_idx["upstream-lic"] = frozenset({"cc-by-4.0", "cc-by-3.0"})
    from dataclasses import replace
    m2 = replace(m, upstream_compat_index=new_idx)

    r = recommend(frozenset({"upstream-lic"}), m2, Counter())
    assert r.is_solvable is True
    cc = r.by_category["CC"]
    # cc-by-3.0 is dropped because cc-by-4.0 exists for the same base.
    assert "cc-by-4.0" in cc
    assert "cc-by-3.0" not in cc
