import pytest

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore, ResolveResult


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    return BomStore.open()


class TestResolve:
    def test_resolve_with_no_matches_signals_new(self, store):
        result = store.resolve(
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
            use_case="license",
            mode="rag",
        )
        assert isinstance(result, ResolveResult)
        assert result.existing_artifact is None
        assert result.matching_claims == []

    def test_resolve_finds_saved_claim(self, store, sample_bom, sample_run_meta):
        ids = [Identifier("huggingface", "mistralai/Mistral-7B-v0.1")]
        claim_iri = store.save_claim(sample_bom, sample_run_meta, identifiers=ids)
        result = store.resolve(identifiers=ids, use_case="license", mode="rag")
        assert result.existing_artifact is not None
        assert any(c["iri"] == claim_iri for c in result.matching_claims)
