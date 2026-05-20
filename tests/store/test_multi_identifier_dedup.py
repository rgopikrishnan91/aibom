import pytest

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    return BomStore.open()


def test_save_with_hf_then_resolve_with_arxiv_finds_same_artifact(
    store,
    sample_bom,
    sample_run_meta,
):
    """A later resolve with only arxiv finds the prior HF+arxiv save and its claims."""
    claim_iri = store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[
            Identifier("huggingface", "mistralai/Mistral-7B-v0.1"),
            Identifier("arxiv", "2310.06825"),
        ],
    )
    result = store.resolve(
        identifiers=[Identifier("arxiv", "2310.06825")],
        use_case="license",
        mode="rag",
    )
    assert result.existing_artifact is not None
    # Cross-identifier lookup must surface the saved claim — bug if matching_claims is empty.
    assert len(result.matching_claims) == 1
    assert result.matching_claims[0]["iri"] == claim_iri


def test_name_variants_collapse_to_one_artifact(store, sample_bom, sample_run_meta):
    """`Mistral-7B-v0.1` and `MistralAI/Mistral-7B-v0.1` end up on one node."""
    store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "MistralAI/Mistral-7B-v0.1")],
    )
    store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
    )
    stats = store.stats()
    assert stats["artifacts"] == 1
    assert stats["claims"] == 2


def test_collision_returns_multiple_artifacts(store, sample_bom, sample_run_meta):
    """When the same set straddles two separately-saved artifacts, collision_artifacts populates."""
    # Save artifact A with only HF.
    store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "owner-a/model")],
    )
    # Save artifact B with only arxiv. These are independent records that
    # happen to refer to the same upstream thing.
    store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("arxiv", "1234.56789")],
    )
    # Resolve with BOTH identifiers — collision case.
    result = store.resolve(
        identifiers=[
            Identifier("huggingface", "owner-a/model"),
            Identifier("arxiv", "1234.56789"),
        ],
    )
    assert result.existing_artifact is not None
    assert len(result.collision_artifacts) == 1


def test_save_with_subset_of_identifiers_reuses_artifact(
    store, sample_bom, sample_run_meta,
):
    """A second save with a strict subset of the first save's identifiers
    must land on the same artifact, not create a new one.

    Regression for the e2e scenario in test_e2e_reuse_via_process where a
    BOM saved with (hf, arxiv) was re-saved with arxiv only and produced
    two artifact nodes. Root cause: pick_primary depended on the call's
    identifier set, so the second call computed a different artifact IRI
    even though resolve() would have found the first.
    """
    store.save_claim(
        sample_bom, sample_run_meta,
        identifiers=[
            Identifier("huggingface", "mistralai/Mistral-7B-v0.1"),
            Identifier("arxiv", "2310.06825"),
        ],
    )
    store.save_claim(
        sample_bom, sample_run_meta,
        identifiers=[Identifier("arxiv", "2310.06825")],
    )
    stats = store.stats()
    assert stats["artifacts"] == 1, f"expected 1 artifact, got {stats}"
    assert stats["claims"] == 2, f"expected 2 claims, got {stats}"

    # Topology — both claims must hang off the same artifact via the
    # hasVersion → hasClaim chain. Mirrors the equivalent assertion in
    # tests/store/test_e2e_reuse_via_process.py::test_identical_model_reuses_artifact.
    from aikaboom.store import vocab
    rows = list(store._backend.select(f"""
        SELECT DISTINCT ?artifact WHERE {{
            ?artifact <{vocab.hasVersion}> ?version .
            ?version <{vocab.hasClaim}> ?claim .
        }}
    """))
    assert len(rows) == 1, f"expected 1 artifact-with-claims, got {rows}"
    claim_rows = list(store._backend.select(f"""
        SELECT ?claim WHERE {{
            <{rows[0]["artifact"]}> <{vocab.hasVersion}> ?v .
            ?v <{vocab.hasClaim}> ?claim .
        }}
    """))
    assert len(claim_rows) == 2, f"expected 2 claims under that artifact, got {claim_rows}"
