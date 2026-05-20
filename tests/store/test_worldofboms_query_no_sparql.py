"""POST /worldofboms/query no longer honours an `sparql` field.

The web UI used to expose a raw SPARQL textarea inside the lineage
side-panel. Per the worldofBOMs followups spec section C, that surface
is removed: the query handler accepts only the preset branch now.

graph_view.raw_query is NOT removed — CLI + tests still use it.
"""

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(graph_dir))
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DISABLE", "0")
    from aikaboom.web.app import app
    app.config["TESTING"] = True
    return app.test_client()


def test_query_with_sparql_field_is_rejected(client):
    """A request body with `sparql: ...` returns 400 with a clear message."""
    r = client.post("/worldofboms/query", json={
        "sparql": "SELECT ?s WHERE { ?s ?p ?o } LIMIT 5",
    })
    assert r.status_code == 400
    body = r.get_json()
    assert "sparql" in (body.get("error") or "").lower()


def test_query_with_preset_still_works(client):
    """The preset branch is unchanged.

    Specifically: posting {preset: 'datasets'} against an empty graph
    returns 200 with `rows` populated to `[]` (not 4xx/5xx). This pins
    the contract that `artifact=''` on an empty graph is a valid call —
    if `lineage_query` later requires a non-empty artifact, this test
    must be updated rather than silently testing the wrong invariant.
    """
    r = client.post("/worldofboms/query", json={
        "preset": "datasets", "direction": "both",
    })
    # Empty graph → empty rows or some valid response, but NOT 4xx/5xx.
    assert r.status_code == 200
    body = r.get_json()
    assert "rows" in body
