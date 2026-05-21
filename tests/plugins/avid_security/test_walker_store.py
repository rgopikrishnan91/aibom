"""Integration test: walk_components against a real BomStore populated from TTL fixtures.

Mirrors the pattern from tests/plugins/license_compat/conftest.py and test_walker.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aikaboom.plugins import Scope
from aikaboom.plugins.avid_security.walker import Component, walk_components

FIXTURES = Path(__file__).parent / "fixtures"


def _load_ttl_into_store(store, ttl_path: Path) -> None:
    """Parse a Turtle file via rdflib and bulk-add triples to the store.

    Mirrors license_compat/conftest.py:_load_ttl_into_store exactly.
    """
    from rdflib import Graph

    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    store._backend.add_quads([(s, p, o, None) for s, p, o in g])


@pytest.fixture
def avid_corpus_store(tmp_path, monkeypatch):
    """BomStore (RDFLib backend) populated from avid_bom_corpus.ttl."""
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_path))

    from aikaboom.store import BomStore
    store = BomStore.open()
    _load_ttl_into_store(store, FIXTURES / "avid_bom_corpus.ttl")
    yield store
    store._backend.close()


# ── graph_wide scope ───────────────────────────────────────────────────────


def test_graph_wide_yields_expected_count(avid_corpus_store):
    components = list(walk_components(avid_corpus_store, Scope.graph_wide()))
    # 2 Models + 1 Dataset in the fixture
    assert len(components) == 3


def test_graph_wide_component_kinds(avid_corpus_store):
    components = list(walk_components(avid_corpus_store, Scope.graph_wide()))
    kinds = {c.kind for c in components}
    assert "Model" in kinds
    assert "Dataset" in kinds


def test_graph_wide_hf_paths(avid_corpus_store):
    components = list(walk_components(avid_corpus_store, Scope.graph_wide()))
    labels = {c.hf_path for c in components}
    assert "bert-base-uncased" in labels
    assert "dslim/bert-base-NER" in labels
    assert "squad_v2" in labels


def test_graph_wide_supplier_populated(avid_corpus_store):
    components = list(walk_components(avid_corpus_store, Scope.graph_wide()))
    by_label = {c.hf_path: c for c in components}
    assert by_label["bert-base-uncased"].developer == "Hugging Face"
    assert by_label["dslim/bert-base-NER"].developer == "dslim"


def test_graph_wide_supplier_none_when_absent(avid_corpus_store):
    components = list(walk_components(avid_corpus_store, Scope.graph_wide()))
    by_label = {c.hf_path: c for c in components}
    assert by_label["squad_v2"].developer is None


def test_graph_wide_base_models_populated(avid_corpus_store):
    components = list(walk_components(avid_corpus_store, Scope.graph_wide()))
    by_label = {c.hf_path: c for c in components}
    # bert-fine-tuned was trainedOn bert-base-uncased
    assert "bert-base-uncased" in by_label["dslim/bert-base-NER"].base_models


def test_graph_wide_base_models_empty_for_standalone(avid_corpus_store):
    components = list(walk_components(avid_corpus_store, Scope.graph_wide()))
    by_label = {c.hf_path: c for c in components}
    assert by_label["bert-base-uncased"].base_models == ()


def test_graph_wide_spdx_id_is_artifact_iri(avid_corpus_store):
    components = list(walk_components(avid_corpus_store, Scope.graph_wide()))
    by_label = {c.hf_path: c for c in components}
    assert by_label["bert-base-uncased"].spdx_id == "https://example.org/BertBase"


def test_graph_wide_scope_in_bom_is_principal(avid_corpus_store):
    components = list(walk_components(avid_corpus_store, Scope.graph_wide()))
    assert all(c.scope_in_bom == "principal" for c in components)


def test_graph_wide_bare_name_property(avid_corpus_store):
    components = list(walk_components(avid_corpus_store, Scope.graph_wide()))
    by_label = {c.hf_path: c for c in components}
    assert by_label["bert-base-uncased"].bare_name == "bert-base-uncased"
    assert by_label["dslim/bert-base-NER"].bare_name == "bert-base-ner"


# ── single scope ───────────────────────────────────────────────────────────


def test_single_scope_returns_requested_artifact(avid_corpus_store):
    bert_iri = "https://example.org/BertBase"
    components = list(walk_components(avid_corpus_store, Scope.single(bert_iri, depth=1)))
    by_label = {c.hf_path: c for c in components}
    assert "bert-base-uncased" in by_label


def test_single_scope_fine_tune_includes_artifact(avid_corpus_store):
    fine_tune_iri = "https://example.org/BertFineTuned"
    components = list(walk_components(avid_corpus_store, Scope.single(fine_tune_iri, depth=1)))
    by_label = {c.hf_path: c for c in components}
    assert "dslim/bert-base-NER" in by_label
    assert by_label["dslim/bert-base-NER"].base_models == ("bert-base-uncased",)
