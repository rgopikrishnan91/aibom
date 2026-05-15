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
