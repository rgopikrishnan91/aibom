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
