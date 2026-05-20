"""Trust votes, score aggregation, canonical-claim pointer."""

import pytest

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore
from aikaboom.store.trust import VoteKind


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_AGENT_ID", "test-agent")
    return BomStore.open()


@pytest.fixture
def claim(store, sample_bom, sample_run_meta):
    return store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
    )


def test_explicit_trust_vote_increases_score(store, claim):
    score_before = store.trust_score(claim)
    store.record_trust_vote(claim, VoteKind.TRUSTED)
    score_after = store.trust_score(claim)
    assert score_after > score_before


def test_explicit_flag_decreases_score(store, claim):
    store.record_trust_vote(claim, VoteKind.FLAGGED)
    assert store.trust_score(claim) < 0


def test_implicit_use_weighs_less_than_explicit(store, sample_bom, sample_run_meta):
    c1 = store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "owner-a/model")],
    )
    c2 = store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "owner-b/model")],
    )
    store.record_trust_vote(c1, VoteKind.TRUSTED)
    store.record_trust_vote(c2, VoteKind.IMPLICIT_USE)
    assert store.trust_score(c1) > store.trust_score(c2)


def test_canonical_claim_points_to_highest_trust(store, sample_bom, sample_run_meta):
    """Two claims on the same version: canonical points to the one with more trust."""
    ids = [Identifier("huggingface", "mistralai/Mistral-7B-v0.1")]
    store.save_claim(sample_bom, sample_run_meta, identifiers=ids)
    run_meta_b = dict(sample_run_meta)
    run_meta_b["llm_model"] = "openai/gpt-4o-mini"
    c2 = store.save_claim(sample_bom, run_meta_b, identifiers=ids)
    store.record_trust_vote(c2, VoteKind.TRUSTED)
    store.recompute_canonical_for_claim(c2)
    canonical = store.canonical_claim_for(ids, version_hint="27d67f1b")
    assert canonical == c2
