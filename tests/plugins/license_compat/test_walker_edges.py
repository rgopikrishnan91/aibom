"""Edge cases for the graph walker.

Covers the artifact-level ``hasLicense`` fallback, the ``MISSING`` license
marker, the UNKNOWN-license tagging branch, and the defensive raise in
``enumerate_edges`` for unknown scope kinds.
"""
from __future__ import annotations

import pytest
import rdflib

from aikaboom.plugins import Scope
from aikaboom.plugins.license_compat.walker import (
    enumerate_edges,
    resolve_artifact_license,
)


def test_enumerate_edges_raises_on_unknown_scope_kind(lineage_3node_store):
    bogus = Scope.single("https://example.org/x")
    # The frozen dataclass doesn't allow direct mutation; rebuild with an
    # invalid kind via ``object.__setattr__`` to exercise the defensive raise.
    object.__setattr__(bogus, "kind", "graph-fragment")
    with pytest.raises(ValueError, match="Unknown scope kind"):
        list(enumerate_edges(lineage_3node_store, bogus))


def test_resolve_artifact_license_handles_missing_marker(
    tmp_path, monkeypatch, tiny_matrix
):
    """If the only claim's hasLicense is the literal "MISSING", the resolver
    flags ``has_missing`` and returns an empty license set."""
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_path))
    from aikaboom.store import BomStore

    store = BomStore.open()
    ttl = """
    @prefix aibom: <https://aikaboom.dev/aibom#> .
    @prefix ex: <https://example.org/> .
    ex:MissingArt aibom:hasVersion ex:MissingArt_v1 .
    ex:MissingArt_v1 aibom:hasClaim ex:ClaimM1 .
    ex:ClaimM1 aibom:hasLicense "['MISSING']" .
    """
    g = rdflib.Graph().parse(data=ttl, format="turtle")
    store._backend.add_quads([(s, p, o, None) for s, p, o in g])

    r = resolve_artifact_license(store, "https://example.org/MissingArt", tiny_matrix)
    assert r.has_missing is True
    assert r.licenses == frozenset()
    store._backend.close()


def test_resolve_artifact_license_tags_unknown(
    tmp_path, monkeypatch, tiny_matrix
):
    """A claim with a license string not in the matrix should set
    ``has_unknown`` and still register UNKNOWN as a resolved license."""
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_path))
    from aikaboom.store import BomStore

    store = BomStore.open()
    ttl = """
    @prefix aibom: <https://aikaboom.dev/aibom#> .
    @prefix ex: <https://example.org/> .
    ex:WeirdArt aibom:hasVersion ex:WeirdArt_v1 .
    ex:WeirdArt_v1 aibom:hasClaim ex:ClaimW1 .
    ex:ClaimW1 aibom:hasLicense "BogusFakeLic-99" .
    """
    g = rdflib.Graph().parse(data=ttl, format="turtle")
    store._backend.add_quads([(s, p, o, None) for s, p, o in g])

    r = resolve_artifact_license(store, "https://example.org/WeirdArt", tiny_matrix)
    assert r.has_unknown is True
    assert "UNKNOWN" in r.licenses
    store._backend.close()


def test_resolve_artifact_license_falls_back_to_artifact_level(
    tmp_path, monkeypatch, tiny_matrix
):
    """When there's no claim-level license, the walker falls back to the
    artifact's own ``hasLicense`` triple (walker.py:174-177)."""
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_path))
    from aikaboom.store import BomStore

    store = BomStore.open()
    ttl = """
    @prefix aibom: <https://aikaboom.dev/aibom#> .
    @prefix ex: <https://example.org/> .
    ex:DirectArt aibom:hasLicense "apache-2.0" .
    """
    g = rdflib.Graph().parse(data=ttl, format="turtle")
    store._backend.add_quads([(s, p, o, None) for s, p, o in g])

    r = resolve_artifact_license(store, "https://example.org/DirectArt", tiny_matrix)
    assert r.licenses == frozenset({"apache-2.0"})
    store._backend.close()
