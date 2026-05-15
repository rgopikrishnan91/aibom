"""End-to-end: each --cache value triggers the right BomStore behavior."""

import pytest

from aikaboom.store.cache_resolver import CachePolicy, decide
from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore


@pytest.fixture
def store_with_claim(tmp_store_dir, monkeypatch, sample_bom, sample_run_meta):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    store = BomStore.open()
    store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
    )
    return store


def test_use_policy_returns_use(store_with_claim):
    result = store_with_claim.resolve(
        identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        use_case="license",
        mode="rag",
    )
    assert decide(result, CachePolicy.USE, interactive=False) == "use"


def test_regen_policy_returns_generate(store_with_claim):
    result = store_with_claim.resolve(
        identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        use_case="license",
        mode="rag",
    )
    assert decide(result, CachePolicy.REGEN, interactive=False) == "generate"
