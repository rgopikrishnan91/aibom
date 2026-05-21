"""find_compatible_subchains tests."""
from __future__ import annotations

from aikaboom.plugins.license_compat.engine import (
    CompatVerdict,
    Finding,
    Findings,
    find_compatible_subchains,
)


def _finding(d_iri: str, u_iri: str, status: str, predicate: str = "trainedOn") -> Finding:
    return Finding(
        downstream_iri=d_iri,
        downstream_label=d_iri,
        upstream_iri=u_iri,
        upstream_label=u_iri,
        predicate=predicate,
        verdict=CompatVerdict(
            downstream="mit" if status == "compatible" else "gpl-3.0",
            upstreams=frozenset({"apache-2.0"}),
            status=status,
            incompatible_with=frozenset() if status == "compatible" else frozenset({"apache-2.0"}),
        ),
        recommendation=None,
    )


def test_single_compatible_edge_yields_one_subchain_size_2():
    f = Findings([_finding("A", "B", "compatible")])
    chains = find_compatible_subchains(f)
    assert len(chains) == 1
    assert chains[0].size == 2
    assert chains[0].artifacts == frozenset({"A", "B"})


def test_chain_of_compatible_edges_merges_into_one_component():
    f = Findings([
        _finding("A", "B", "compatible"),
        _finding("B", "C", "compatible"),
        _finding("C", "D", "compatible"),
    ])
    chains = find_compatible_subchains(f)
    assert len(chains) == 1
    assert chains[0].size == 4


def test_violation_splits_components():
    f = Findings([
        _finding("A", "B", "compatible"),
        _finding("B", "C", "violation"),
        _finding("C", "D", "compatible"),
    ])
    chains = find_compatible_subchains(f)
    sizes = sorted(c.size for c in chains)
    assert sizes == [2, 2]


def test_isolated_violation_node_appears_as_size_1_component():
    # Z has only a violation edge and shouldn't disappear.
    f = Findings([
        _finding("A", "B", "compatible"),
        _finding("Z", "A", "violation"),
    ])
    chains = find_compatible_subchains(f)
    sizes = sorted(c.size for c in chains)
    assert sizes == [1, 2]


def test_chains_sorted_by_size_desc():
    f = Findings([
        _finding("A", "B", "compatible"),
        _finding("C", "D", "compatible"),
        _finding("D", "E", "compatible"),
    ])
    chains = find_compatible_subchains(f)
    assert chains[0].size >= chains[1].size
