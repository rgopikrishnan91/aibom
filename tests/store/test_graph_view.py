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
