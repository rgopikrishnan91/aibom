"""find_breaking_nodes tests."""
from __future__ import annotations

from collections import Counter

from aikaboom.plugins.license_compat.engine import (
    CompatVerdict,
    Finding,
    Findings,
    find_breaking_nodes,
)


def _vfinding(d_iri: str, u_iri: str, blocked_upstream: str) -> Finding:
    return Finding(
        downstream_iri=d_iri,
        downstream_label=d_iri,
        upstream_iri=u_iri,
        upstream_label=u_iri,
        predicate="trainedOn",
        downstream_license="gpl-3.0",
        upstream_licenses=frozenset({blocked_upstream}),
        verdict=CompatVerdict(
            downstream="gpl-3.0",
            upstreams=frozenset({blocked_upstream}),
            status="violation",
            incompatible_with=frozenset({blocked_upstream}),
        ),
        recommendation=None,
    )


def test_breaking_node_blame_count(tiny_matrix):
    # cc-by-nc-4.0 blocks three different downstreams
    findings = Findings([
        _vfinding("D1", "X_NC", "cc-by-nc-4.0"),
        _vfinding("D2", "X_NC", "cc-by-nc-4.0"),
        _vfinding("D3", "X_NC", "cc-by-nc-4.0"),
    ])
    nodes = find_breaking_nodes(findings, tiny_matrix, Counter())
    assert len(nodes) == 1
    assert nodes[0].artifact_iri == "X_NC"
    assert nodes[0].blamed_in == 3
    assert nodes[0].affected_downstream == frozenset({"D1", "D2", "D3"})


def test_breaking_nodes_sorted_by_blame_desc(tiny_matrix):
    findings = Findings([
        _vfinding("D1", "A", "cc-by-nc-4.0"),
        _vfinding("D2", "A", "cc-by-nc-4.0"),
        _vfinding("D3", "B", "cc-by-nc-4.0"),
    ])
    nodes = find_breaking_nodes(findings, tiny_matrix, Counter())
    assert nodes[0].artifact_iri == "A"
    assert nodes[0].blamed_in == 2
    assert nodes[1].artifact_iri == "B"
    assert nodes[1].blamed_in == 1


def test_breaking_node_fix_recommendations_use_downstream_union(tiny_matrix):
    # X blocks D1 (which is gpl-3.0) — fix_recommendations should be
    # licenses that, if X took them instead, would clear all blame.
    findings = Findings([_vfinding("D1", "X", "cc-by-nc-4.0")])
    nodes = find_breaking_nodes(findings, tiny_matrix, Counter({"apache-2.0": 1}))
    assert nodes[0].fix_recommendations.is_solvable is True
