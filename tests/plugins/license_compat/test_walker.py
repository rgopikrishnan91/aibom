"""Graph walker tests."""
from __future__ import annotations

from collections import Counter

from aikaboom.plugins.license_compat.walker import (
    LineageEdge,
    compute_license_frequencies,
    enumerate_edges,
    resolve_artifact_license,
)
from aikaboom.plugins import Scope


def test_enumerate_graph_wide_returns_all_lineage_edges(lineage_3node_store):
    edges = list(enumerate_edges(lineage_3node_store, Scope.graph_wide()))
    pairs = {(e.downstream_iri, e.upstream_iri, e.predicate.rsplit("#", 1)[-1]) for e in edges}
    assert ("https://example.org/ModelA", "https://example.org/DatasetB", "trainedOn") in pairs
    assert ("https://example.org/DatasetB", "https://example.org/PaperC", "dependsOn") in pairs


def test_resolve_artifact_license_picks_highest_trust(lineage_3node_store, tiny_matrix):
    r = resolve_artifact_license(lineage_3node_store, "https://example.org/ModelA", tiny_matrix)
    # ClaimMA1 (apache-2.0, trust=0.9) wins over ClaimMA2 (gpl-3.0, trust=0.2)
    assert r.licenses == frozenset({"apache-2.0"})


def test_resolve_artifact_license_unknown_when_no_claim(lineage_3node_store, tiny_matrix):
    r = resolve_artifact_license(lineage_3node_store, "https://example.org/DoesNotExist", tiny_matrix)
    assert r.licenses == frozenset()


def test_compute_license_frequencies(lineage_3node_store, tiny_matrix):
    freqs = compute_license_frequencies(lineage_3node_store, tiny_matrix)
    # 2 claims with apache-2.0, 1 with gpl-3.0, 1 with cc-by-nc-4.0
    assert freqs["apache-2.0"] >= 2
    assert freqs["cc-by-nc-4.0"] >= 1
