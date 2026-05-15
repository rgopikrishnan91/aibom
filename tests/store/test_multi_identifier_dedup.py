import pytest

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    return BomStore.open()


def test_save_with_hf_then_resolve_with_arxiv_finds_same_artifact(
    store, sample_bom, sample_run_meta,
):
    """If a BOM was saved with HF+arxiv ids, a later resolve with only arxiv finds it."""
    store.save_claim(
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
