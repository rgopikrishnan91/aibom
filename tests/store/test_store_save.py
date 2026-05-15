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


def test_reconstruct_bom_returns_saved_fields(store, sample_bom, sample_run_meta):
    """save_claim → reconstruct_bom returns the same direct_fields values."""
    claim_iri = store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
    )
    reconstructed = store.reconstruct_bom(claim_iri)
    assert reconstructed["direct_fields"]["suppliedBy"]["value"] == "mistralai"
    assert reconstructed["direct_fields"]["suppliedBy"]["source"] == "huggingface"
    assert reconstructed["direct_fields"]["packageVersion"]["value"] == "27d67f1b"


def test_sparql_injection_in_use_case_is_safe(store, sample_bom, sample_run_meta):
    """A `use_case` containing a SPARQL-special char doesn't break the query."""
    store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
    )
    # An attacker-supplied use_case containing a quote shouldn't crash or inject.
    # We just need the call to return cleanly (and return zero matches since
    # the escaped literal will not match the stored "license" value).
    claims = store.find_claims_for(
        [Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        use_case='license"} UNION { ?x ?y ?z',
        mode="rag",
    )
    assert claims == []  # no matches; the escaped string is a literal, not SPARQL


def test_merge_artifacts_transfers_incoming_edges(store):
    """merge_artifacts(into, from_) redirects edges that pointed at from_."""
    from rdflib import URIRef
    a = "bom:artifact/real"
    b = "bom:artifact/placeholder"
    model = "bom:artifact/model"
    # model --dependsOn--> placeholder b
    store._backend.add_quads([
        (URIRef(model), URIRef("https://aikaboom.dev/aibom#dependsOn"), URIRef(b), None),
        (URIRef(b), URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
         URIRef("https://aikaboom.dev/aibom#Artifact"), None),
    ])
    store.merge_artifacts(into=a, from_=b)
    # The edge now points at a; nothing points at b; b has no outgoing triples.
    rows = list(store._backend.select(
        f"SELECT ?s WHERE {{ ?s <https://aikaboom.dev/aibom#dependsOn> <{a}> }}"))
    assert len(rows) == 1
    leftover = list(store._backend.select(f"SELECT ?p WHERE {{ <{b}> ?p ?o }}"))
    assert leftover == []
