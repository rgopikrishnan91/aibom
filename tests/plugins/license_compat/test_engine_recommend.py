"""Recommendation logic tests."""
from __future__ import annotations

from collections import Counter

from aikaboom.plugins.license_compat.engine import recommend


def test_recommend_returns_intersection_filtered_by_whitelist(tiny_matrix):
    freqs = Counter({"apache-2.0": 100, "mit": 80, "lgpl-3.0": 10})
    r = recommend(frozenset({"apache-2.0"}), tiny_matrix, freqs)
    # apache-2.0 is upstream-compat with apache-2.0, mit, lgpl-3.0 (per Yes/Same in matrix).
    # whitelist has apache-2.0, mit, lgpl-3.0, cc-by-nc-4.0.
    assert "PERMISSIVE" in r.by_category
    assert "apache-2.0" in r.by_category["PERMISSIVE"]
    assert "mit" in r.by_category["PERMISSIVE"]
    assert r.is_solvable is True


def test_recommend_orders_by_frequency_desc(tiny_matrix):
    freqs = Counter({"mit": 1000, "apache-2.0": 1})
    r = recommend(frozenset({"apache-2.0"}), tiny_matrix, freqs)
    # mit wins on frequency
    assert r.by_category["PERMISSIVE"][0] == "mit"


def test_recommend_no_compatible_intersection_returns_empty(tiny_matrix):
    # downstream of gpl-3.0 AND cc-by-nc-4.0: nothing satisfies both
    r = recommend(frozenset({"gpl-3.0", "cc-by-nc-4.0"}), tiny_matrix, Counter())
    assert r.by_category == {}
    assert r.is_solvable is False


def test_recommend_returns_top_k_per_category(tiny_matrix):
    # synthesize a category overload; tiny matrix only has small categories so this
    # validates the cap is applied — set k=1 and assert at most 1 per category.
    freqs = Counter({"apache-2.0": 10, "mit": 9, "lgpl-3.0": 8})
    r = recommend(frozenset({"apache-2.0"}), tiny_matrix, freqs, top_k_per_category=1)
    for cat, items in r.by_category.items():
        assert len(items) <= 1


def test_recommend_excludes_non_whitelisted(tiny_matrix):
    # Inject a candidate that wouldn't be in the allowed list. Easy way: use
    # an empty allowed set via a fresh matrix override.
    matrix = type(tiny_matrix)(
        name_alias_lookup=tiny_matrix.name_alias_lookup,
        details=tiny_matrix.details,
        upstream_compat_index=tiny_matrix.upstream_compat_index,
        allowed_licenses=frozenset(),
        missing_licenses=tiny_matrix.missing_licenses,
        timestamp=tiny_matrix.timestamp,
    )
    r = recommend(frozenset({"apache-2.0"}), matrix, Counter())
    assert r.by_category == {}
    # is_solvable reflects pre-whitelist solvability
    assert r.is_solvable is True
