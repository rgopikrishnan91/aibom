"""Shared fixtures for license_compat tests."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_matrix_paths():
    return {
        "matrix": FIXTURES / "tiny_matrix.json",
        "allowed": FIXTURES / "tiny_allowed.json",
        "missing": FIXTURES / "tiny_missing.json",
    }


@pytest.fixture
def tiny_matrix(tiny_matrix_paths):
    from aikaboom.plugins.license_compat.matrix import load_matrix
    return load_matrix(
        matrix_path=tiny_matrix_paths["matrix"],
        allowed_path=tiny_matrix_paths["allowed"],
        missing_path=tiny_matrix_paths["missing"],
    )


def _load_ttl_into_store(store, ttl_path: Path) -> None:
    """Parse a turtle file via rdflib and bulk-add triples to the store.

    The RDFLib backend's ``import_`` only recognises ``nquads`` and ``jsonld``
    in its format map, so we go through rdflib directly to keep the fixture
    backend-agnostic and avoid editing backend code.
    """
    from rdflib import Graph

    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    store._backend.add_quads([(s, p, o, None) for s, p, o in g])


@pytest.fixture
def lineage_3node_store(tmp_path, monkeypatch):
    """Build a BomStore (RDFLib backend) populated from lineage_3node.ttl."""
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_path))

    from aikaboom.store import BomStore
    store = BomStore.open()
    _load_ttl_into_store(store, FIXTURES / "lineage_3node.ttl")
    yield store
    store._backend.close()
