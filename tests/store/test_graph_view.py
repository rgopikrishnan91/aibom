"""Read-side graph queries for the worldofBOMs visualization."""

import pytest

from aikaboom.store.store import BomStore
from aikaboom.store.naming import Identifier
from aikaboom.store import graph_view


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    return BomStore.open()


def _save_model_with_dataset(store, run_meta, model="acme/m", dataset="squad"):
    bom = {"repo_id": model, "use_case": "complete",
           "direct_fields": {"trainedOnDatasets": {"value": dataset,
                             "source": "huggingface", "conflict": None}},
           "rag_fields": {}}
    store.save_claim(bom, run_meta, identifiers=[Identifier("huggingface", model)])


def test_full_graph_returns_nodes_and_edges(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta)
    g = graph_view.full_graph(store)
    assert len(g["nodes"]) == 2          # model + (placeholder) dataset
    assert len(g["edges"]) == 1
    edge = g["edges"][0]
    assert edge["predicate"] == "trainedOn"
    labels = {n["label"] for n in g["nodes"]}
    assert "acme/m" in labels


def test_full_graph_marks_placeholder_nodes(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta)
    g = graph_view.full_graph(store)
    placeholders = [n for n in g["nodes"] if n["is_placeholder"]]
    assert len(placeholders) == 1


def test_ego_graph_upstream_returns_dependencies(store, sample_run_meta):
    # m --trainedOn--> squad   (squad is m's dependency = upstream of m)
    _save_model_with_dataset(store, sample_run_meta, model="acme/m", dataset="squad")
    g = graph_view.full_graph(store)
    m_iri = next(n["iri"] for n in g["nodes"] if n["label"] == "acme/m")
    ego = graph_view.ego_graph(store, m_iri, direction="up", depth=None)
    labels = {n["label"] for n in ego["nodes"]}
    assert "acme/m" in labels and "squad" in labels
    assert ego["focus"] == m_iri


def test_ego_graph_downstream_excludes_dependencies(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta, model="acme/m", dataset="squad")
    g = graph_view.full_graph(store)
    m_iri = next(n["iri"] for n in g["nodes"] if n["label"] == "acme/m")
    ego = graph_view.ego_graph(store, m_iri, direction="down", depth=None)
    # m has no dependents -> downstream ego is just m itself.
    assert {n["label"] for n in ego["nodes"]} == {"acme/m"}


def test_ego_graph_both_is_the_union(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta, model="acme/m", dataset="squad")
    g = graph_view.full_graph(store)
    squad_iri = next(n["iri"] for n in g["nodes"] if n["label"] == "squad")
    ego = graph_view.ego_graph(store, squad_iri, direction="both", depth=None)
    assert {n["label"] for n in ego["nodes"]} == {"acme/m", "squad"}


def test_ego_graph_depth_caps_hops(store, sample_run_meta):
    from rdflib import URIRef
    a, b, c = "bom:artifact/a", "bom:artifact/b", "bom:artifact/c"
    dep = "https://aikaboom.dev/aibom#dependsOn"
    art = "https://aikaboom.dev/aibom#Artifact"
    rtype = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    for x in (a, b, c):
        store._backend.add_quads([(URIRef(x), URIRef(rtype), URIRef(art), None)])
    store._backend.add_quads([
        (URIRef(a), URIRef(dep), URIRef(b), None),
        (URIRef(b), URIRef(dep), URIRef(c), None),
    ])
    ego = graph_view.ego_graph(store, a, direction="up", depth=1)
    iris = {n["iri"] for n in ego["nodes"]}
    assert a in iris and b in iris  # one hop reaches b
    assert c not in iris            # depth=1 stops before c


def test_ego_graph_rejects_invalid_direction(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta)
    g = graph_view.full_graph(store)
    iri = g["nodes"][0]["iri"]
    with pytest.raises(ValueError):
        graph_view.ego_graph(store, iri, direction="upstream", depth=None)


def test_lineage_query_lists_datasets_in_lineage(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta, model="acme/m", dataset="squad")
    g = graph_view.full_graph(store)
    m_iri = next(n["iri"] for n in g["nodes"] if n["label"] == "acme/m")
    rows = graph_view.lineage_query(store, m_iri, preset="datasets", direction="up")
    assert any(r["label"] == "squad" for r in rows)


def test_lineage_query_lists_models_in_lineage(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta, model="acme/m", dataset="squad")
    g = graph_view.full_graph(store)
    m_iri = next(n["iri"] for n in g["nodes"] if n["label"] == "acme/m")
    rows = graph_view.lineage_query(store, m_iri, preset="models", direction="both")
    assert any(r["label"] == "acme/m" for r in rows)


def test_lineage_query_unknown_preset_raises(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta)
    g = graph_view.full_graph(store)
    iri = g["nodes"][0]["iri"]
    with pytest.raises(ValueError):
        graph_view.lineage_query(store, iri, preset="nonsense", direction="up")


def test_lineage_query_includes_placeholder_models(store, sample_run_meta):
    from aikaboom.store.naming import Identifier
    bom = {"repo_id": "acme/fine", "use_case": "complete",
           "direct_fields": {"modelLineage": {"value": "gpt2",
                             "source": "huggingface", "conflict": None}},
           "rag_fields": {}}
    store.save_claim(bom, sample_run_meta,
                     identifiers=[Identifier("huggingface", "acme/fine")])
    g = graph_view.full_graph(store)
    fine_iri = next(n["iri"] for n in g["nodes"] if n["label"] == "acme/fine")
    rows = graph_view.lineage_query(store, fine_iri, preset="models", direction="up")
    labels = {r["label"] for r in rows}
    assert "gpt2" in labels   # the placeholder model is found


def test_raw_query_runs_select(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta)
    rows = graph_view.raw_query(store, "SELECT ?s WHERE { ?s a ?c } LIMIT 5")
    assert isinstance(rows, list)


@pytest.mark.parametrize("bad", [
    "INSERT DATA { <a:x> <a:y> <a:z> }",
    "DELETE WHERE { ?s ?p ?o }",
    "  delete  { ?s ?p ?o } where { ?s ?p ?o }",
    "DROP ALL",
])
def test_raw_query_rejects_mutations(store, bad):
    with pytest.raises(ValueError):
        graph_view.raw_query(store, bad)


def test_ego_spdx_bundle_has_context_and_graph(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta, model="acme/m", dataset="squad")
    g = graph_view.full_graph(store)
    m_iri = next(n["iri"] for n in g["nodes"] if n["label"] == "acme/m")
    bundle = graph_view.ego_spdx_bundle(store, m_iri, direction="both")
    assert "@context" in bundle and "@graph" in bundle
    assert isinstance(bundle["@graph"], list) and len(bundle["@graph"]) > 0
    # Children must actually appear — the "squad" dataset should be in the bundle
    # as a Package element, and a Relationship element must exist to connect it.
    graph_elements = bundle["@graph"]
    package_names = {
        e.get("name") for e in graph_elements
        if isinstance(e, dict) and "Package" in (e.get("type") or "")
    }
    assert "squad" in package_names, (
        f"Expected child 'squad' as a Package in the bundle; found names: {package_names}"
    )
    has_relationship = any(
        isinstance(e, dict) and e.get("type") == "Relationship"
        for e in graph_elements
    )
    assert has_relationship, "Expected at least one Relationship element in the bundle"


def test_ego_spdx_bundle_full_scope(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta)
    bundle = graph_view.ego_spdx_bundle(store, artifact_iri=None, direction="both")
    assert "@graph" in bundle


def test_raw_query_allows_keyword_named_variables(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta)
    rows = graph_view.raw_query(store, "SELECT ?add WHERE { ?add ?p ?o } LIMIT 3")
    assert isinstance(rows, list)
