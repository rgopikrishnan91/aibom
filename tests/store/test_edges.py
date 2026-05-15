"""Edge extraction + persistence: artifact-to-artifact relationships."""

import pytest
from aikaboom.store.store import BomStore
from aikaboom.store.naming import Identifier as _Id
from aikaboom.store.edges import extract_relationship_targets, resolve_edge_target, canon_name, add_relationship_edges
from aikaboom.store.edges import promote_placeholders_for


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
                     identifiers=[_Id("huggingface", "mistralai/Mistral-7B-v0.1")])
    # store.resolve (step 1) misses here: the stored identifier platform is
    # "huggingface" but the probe platform is "name-only" -> forces the
    # _find_artifact_by_label (step 2) path.
    iri, minted = resolve_edge_target(store, "mistralai/Mistral-7B-v0.1")
    assert minted is False  # matched the real artifact, no placeholder
    assert iri.startswith("bom:artifact/")
    rows = list(store._backend.select(
        f"SELECT ?o WHERE {{ <{iri}> <https://aikaboom.dev/aibom#isPlaceholder> ?o }}"))
    assert len(rows) == 0


def test_resolve_edge_target_placeholder_is_idempotent(store):
    iri1, minted1 = resolve_edge_target(store, "repeated-dataset")
    iri2, minted2 = resolve_edge_target(store, "repeated-dataset")
    assert iri1 == iri2
    assert minted1 is True
    assert minted2 is False  # second call hits store.resolve, not _mint_placeholder


def _bom_with(field, value):
    return {"direct_fields": {field: {"value": value, "source": "huggingface",
                                      "conflict": None}}, "rag_fields": {}}


def test_extracts_trainedon_target():
    bom = _bom_with("trainedOnDatasets", "squad")
    assert ("trainedOn", "squad", "Dataset") in extract_relationship_targets(bom)


def test_extracts_testedon_and_dependson():
    assert ("testedOn", "glue", "Dataset") in extract_relationship_targets(
        _bom_with("testedOnDatasets", "glue"))
    assert ("dependsOn", "bert-base", "Model") in extract_relationship_targets(
        _bom_with("modelLineage", "bert-base"))


def test_splits_multi_value_strings():
    targets = extract_relationship_targets(_bom_with("trainedOnDatasets", "squad, glue; mnli"))
    names = {t for _, t, _ in targets}
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
    assert ("dependsOn", "wikitext", "Dataset") in extract_relationship_targets(bom)


def test_splits_arrow_lineage_chains():
    targets = extract_relationship_targets(_bom_with("modelLineage", "bert-base -> distilbert"))
    names = {t for _, t, _ in targets}
    assert "bert-base" in names and "distilbert" in names


def test_add_relationship_edges_writes_edge(store):
    src = "bom:artifact/source-model"
    bom = _bom_with("trainedOnDatasets", "squad")
    add_relationship_edges(store, src, bom)
    rows = list(store._backend.select(
        f"SELECT ?t WHERE {{ <{src}> <{'https://aikaboom.dev/aibom#trainedOn'}> ?t }}"))
    assert len(rows) == 1


def test_add_relationship_edges_is_idempotent(store):
    src = "bom:artifact/source-model"
    bom = _bom_with("trainedOnDatasets", "squad")
    add_relationship_edges(store, src, bom)
    add_relationship_edges(store, src, bom)  # second call must not duplicate
    rows = list(store._backend.select(
        f"SELECT ?t WHERE {{ <{src}> <{'https://aikaboom.dev/aibom#trainedOn'}> ?t }}"))
    assert len(rows) == 1


def test_placeholder_is_promoted_into_a_later_real_bom(store, sample_run_meta):
    # 1. A model BOM names dataset "squad" -> placeholder minted + edge.
    model_bom = _bom_with("trainedOnDatasets", "squad")
    store.save_claim(model_bom, sample_run_meta,
                     identifiers=[_Id("huggingface", "acme/model-z")])
    before = store.stats()["artifacts"]   # model + placeholder
    # 2. A real BOM for "squad" arrives.
    real_bom = {"repo_id": "squad", "use_case": "complete",
                "direct_fields": {}, "rag_fields": {}}
    store.save_claim(real_bom, sample_run_meta,
                     identifiers=[_Id("huggingface", "squad")])
    assert store.stats()["artifacts"] == before  # real squad replaces the placeholder
    # The placeholder was merged away: the model's trainedOn edge now points
    # at the real artifact, and no placeholder artifact remains.
    placeholders = list(store._backend.select(
        "SELECT ?a WHERE { ?a <https://aikaboom.dev/aibom#isPlaceholder> true }"))
    assert placeholders == []
    edges = list(store._backend.select(
        "SELECT ?s ?t WHERE { ?s <https://aikaboom.dev/aibom#trainedOn> ?t }"))
    assert len(edges) == 1


def test_confident_inexact_match_records_potential_duplicate(store, sample_run_meta):
    """A placeholder minted near an existing artifact gets a potentialDuplicateOf edge."""
    from aikaboom.store.naming import Identifier
    # Existing real artifact under owner "qwen".
    store.save_claim({"repo_id": "qwen/chat", "use_case": "complete",
                      "direct_fields": {}, "rag_fields": {}},
                     sample_run_meta, identifiers=[Identifier("huggingface", "qwen/chat")])
    # Edge target "QwenLM/chat" — same supplier, inexact name -> placeholder + soft edge.
    iri, minted = resolve_edge_target(store, "QwenLM/chat")
    assert minted is True
    dup = list(store._backend.select(
        f"SELECT ?o WHERE {{ <{iri}> <https://aikaboom.dev/aibom#potentialDuplicateOf> ?o }}"))
    assert len(dup) == 1


def test_mint_places_dataset_kind_on_trainedon_placeholder(store):
    iri, minted = resolve_edge_target(store, "some-dataset", kind="Dataset")
    assert minted is True
    rows = list(store._backend.select(
        f"SELECT ?c WHERE {{ <{iri}> a ?c }}"))
    types = {str(r["c"]) for r in rows}
    assert "https://aikaboom.dev/aibom#Dataset" in types


def test_data_bom_promotes_placeholder(store, sample_run_meta):
    # A model BOM names dataset "gluebench" -> placeholder minted.
    model_bom = _bom_with("trainedOnDatasets", "gluebench")
    store.save_claim(model_bom, sample_run_meta,
                     identifiers=[_Id("huggingface", "acme/mod")])
    # A real DATA BOM for "gluebench" arrives, keyed by dataset_id.
    data_bom = {"dataset_id": "gluebench", "use_case": "complete",
                "direct_fields": {}, "rag_fields": {}}
    store.save_claim(data_bom, sample_run_meta,
                     identifiers=[_Id("huggingface", "gluebench")])
    placeholders = list(store._backend.select(
        "SELECT ?a WHERE { ?a <https://aikaboom.dev/aibom#isPlaceholder> true }"))
    assert placeholders == []   # the placeholder was promoted/merged away
    edges = list(store._backend.select(
        "SELECT ?s ?t WHERE { ?s <https://aikaboom.dev/aibom#trainedOn> ?t }"))
    assert len(edges) == 1
