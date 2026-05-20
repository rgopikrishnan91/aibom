"""Successful SPDX validation should record an implicit-validate vote."""

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore
from aikaboom.store.trust import VoteKind


def test_implicit_validate_vote_increments_trust(
    tmp_store_dir, monkeypatch, sample_bom, sample_run_meta
):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_store_dir))
    store = BomStore.open()
    claim_iri = store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
    )
    score_before = store.trust_score(claim_iri)
    store.record_trust_vote(claim_iri, VoteKind.IMPLICIT_VALIDATE)
    score_after = store.trust_score(claim_iri)
    assert score_after > score_before
