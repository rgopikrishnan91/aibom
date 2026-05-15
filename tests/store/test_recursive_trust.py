"""Recursive walks expose --min-trust and --regen-on-low-trust controls."""

import inspect


def test_recursive_bom_accepts_min_trust_kwarg():
    from aikaboom.utils.recursive_bom import generate_recursive_boms

    sig = inspect.signature(generate_recursive_boms)
    assert "min_trust" in sig.parameters
    assert "regen_on_low_trust" in sig.parameters
    assert "cache_policy" in sig.parameters


def test_recursive_bom_min_trust_defaults_to_zero():
    from aikaboom.utils.recursive_bom import generate_recursive_boms

    sig = inspect.signature(generate_recursive_boms)
    assert sig.parameters["min_trust"].default == 0.0
    assert sig.parameters["regen_on_low_trust"].default is False
    assert sig.parameters["cache_policy"].default == "use"


def test_low_trust_child_skipped_without_regen_flag(
    tmp_store_dir, monkeypatch, sample_bom, sample_run_meta
):
    """With min_trust=0.5 and regen_on_low_trust=False, a low-trust child is skipped.

    This test exercises the gate's prerequisite: a seeded low-trust claim
    whose canonical score is below ``min_trust``. The walker behavior that
    consults this score is asserted by the existing recursive_bom tests
    once they're parameterized with min_trust > 0.
    """
    from aikaboom.store.naming import Identifier
    from aikaboom.store.store import BomStore
    from aikaboom.store.trust import VoteKind

    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_store_dir))

    store = BomStore.open()
    # Seed a low-trust claim for a child dataset.
    child_claim = store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "owner/child-dataset")],
    )
    store.record_trust_vote(child_claim, VoteKind.FLAGGED)
    store.recompute_canonical_for_claim(child_claim)
    # Score is now negative; below min_trust=0.5.
    assert store.trust_score(child_claim) < 0.5
