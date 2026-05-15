import pytest

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")  # deterministic for tests
    return BomStore.open()


class TestSaveClaim:
    def test_save_returns_claim_iri(self, store, sample_bom, sample_run_meta):
        claim_iri = store.save_claim(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        assert claim_iri.startswith("bom:claim/")

    def test_stats_reports_one_claim_after_save(self, store, sample_bom, sample_run_meta):
        store.save_claim(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        stats = store.stats()
        assert stats["claims"] == 1
        assert stats["artifacts"] == 1
        assert stats["versions"] == 1

    def test_find_claims_returns_saved_claim(self, store, sample_bom, sample_run_meta):
        claim_iri = store.save_claim(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        claims = store.find_claims_for(
            [Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
            use_case="license",
            mode="rag",
        )
        assert any(c["iri"] == claim_iri for c in claims)
