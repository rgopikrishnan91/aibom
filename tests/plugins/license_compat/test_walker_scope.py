"""Walker scope: single vs graph-wide, depth, cycle safety."""
from __future__ import annotations

from aikaboom.plugins.license_compat.walker import enumerate_edges
from aikaboom.plugins import Scope


def test_scope_single_starts_from_artifact(lineage_3node_store):
    edges = list(enumerate_edges(
        lineage_3node_store,
        Scope.single("https://example.org/ModelA"),
    ))
    # From ModelA the walker reaches DatasetB and then PaperC.
    iris = {e.upstream_iri for e in edges} | {e.downstream_iri for e in edges}
    assert "https://example.org/ModelA" in iris
    assert "https://example.org/DatasetB" in iris
    assert "https://example.org/PaperC" in iris


def test_scope_single_depth_bound(lineage_3node_store):
    # depth=1 means we see only direct upstreams of ModelA.
    edges = list(enumerate_edges(
        lineage_3node_store,
        Scope.single("https://example.org/ModelA", depth=1),
    ))
    upstreams = {e.upstream_iri for e in edges}
    assert "https://example.org/DatasetB" in upstreams
    assert "https://example.org/PaperC" not in upstreams


def test_walker_cycle_safe(tmp_path, monkeypatch):
    """A 2-node cycle must not loop forever."""
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_path))

    from rdflib import Graph

    from aikaboom.store import BomStore

    store = BomStore.open()
    cycle_ttl = """
    @prefix aibom: <https://aikaboom.dev/aibom#> .
    @prefix ex: <https://example.org/> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    ex:A aibom:hasVersion ex:A_v1 .
    ex:A_v1 aibom:hasClaim ex:CA .
    ex:CA aibom:trainedOn ex:B ; aibom:hasLicense "mit" ;
          aibom:trustScore "0.5"^^xsd:double ;
          aibom:createdAt "2026-01-01T00:00:00Z" .
    ex:B aibom:hasVersion ex:B_v1 .
    ex:B_v1 aibom:hasClaim ex:CB .
    ex:CB aibom:dependsOn ex:A ; aibom:hasLicense "mit" ;
          aibom:trustScore "0.5"^^xsd:double ;
          aibom:createdAt "2026-01-02T00:00:00Z" .
    """
    p = tmp_path / "cycle.ttl"
    p.write_text(cycle_ttl)
    g = Graph()
    g.parse(str(p), format="turtle")
    store._backend.add_quads([(s, pr, o, None) for s, pr, o in g])

    edges = list(enumerate_edges(store, Scope.single("https://example.org/A", depth=10)))
    seen = {(e.downstream_iri, e.upstream_iri) for e in edges}
    # Each edge appears at most once.
    assert len(seen) == len(edges)
    store._backend.close()
