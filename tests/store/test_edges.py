"""Edge extraction + persistence: artifact-to-artifact relationships."""

import pytest
from aikaboom.store.store import BomStore
from aikaboom.store.naming import Identifier
from aikaboom.store.edges import extract_relationship_targets, resolve_edge_target, canon_name


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    return BomStore.open()


def test_canon_name_lowercases_and_collapses():
    assert canon_name("SQuAD") == "squad"
    assert canon_name("Common_Crawl") == canon_name("common-crawl")


def test_resolve_edge_target_mints_placeholder_when_unknown(store):
    iri, minted = resolve_edge_target(store, "totally-unknown-dataset")
    assert iri.startswith("bom:artifact/")
    assert minted is True
    # The placeholder is flagged.
    rows = list(store._backend.select(
        f"SELECT ?o WHERE {{ <{iri}> <https://aikaboom.dev/aibom#isPlaceholder> ?o }}"))
    assert len(rows) == 1


def test_resolve_edge_target_finds_existing_artifact_by_label(store, sample_bom,
                                                              sample_run_meta):
    # sample_bom's repo_id is "mistralai/Mistral-7B-v0.1" -> canonicalLabel.
    store.save_claim(sample_bom, sample_run_meta,
                     identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")])
    iri, minted = resolve_edge_target(store, "mistralai/Mistral-7B-v0.1")
    assert minted is False  # matched the real artifact, no placeholder


def _bom_with(field, value):
    return {"direct_fields": {field: {"value": value, "source": "huggingface",
                                      "conflict": None}}, "rag_fields": {}}


def test_extracts_trainedon_target():
    bom = _bom_with("trainedOnDatasets", "squad")
    assert ("trainedOn", "squad") in extract_relationship_targets(bom)


def test_extracts_testedon_and_dependson():
    assert ("testedOn", "glue") in extract_relationship_targets(
        _bom_with("testedOnDatasets", "glue"))
    assert ("dependsOn", "bert-base") in extract_relationship_targets(
        _bom_with("modelLineage", "bert-base"))


def test_splits_multi_value_strings():
    targets = extract_relationship_targets(_bom_with("trainedOnDatasets", "squad, glue; mnli"))
    names = {t for _, t in targets}
    assert names == {"squad", "glue", "mnli"}


def test_drops_non_walkable_targets():
    # arXiv refs are filtered by _is_walkable_target
    targets = extract_relationship_targets(_bom_with("sourceInfo", "arXiv:2108.07732"))
    assert targets == []


def test_ignores_unknown_and_empty_fields():
    assert extract_relationship_targets({"direct_fields": {}, "rag_fields": {}}) == []
    assert extract_relationship_targets(_bom_with("trainedOnDatasets", None)) == []


def test_reads_from_rag_fields():
    bom = {"direct_fields": {},
           "rag_fields": {"sourceInfo": {"value": "wikitext",
                          "source": "huggingface", "conflict": None}}}
    assert ("dependsOn", "wikitext") in extract_relationship_targets(bom)


def test_splits_arrow_lineage_chains():
    targets = extract_relationship_targets(_bom_with("modelLineage", "bert-base -> distilbert"))
    names = {t for _, t in targets}
    assert "bert-base" in names and "distilbert" in names
