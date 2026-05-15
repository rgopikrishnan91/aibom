"""Placeholder artifacts for unresolvable recursive references."""
import pytest

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    return BomStore.open()


def test_placeholder_excluded_from_primary_match(store, sample_bom, sample_run_meta):
    """An artifact created with platform='name-only' is flagged and not matched as primary."""
    store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("name-only", "some internal dataset")],
    )
    # A subsequent resolve with the same name-only id should still find it
    # (placeholders are *queryable*, just not promoted to primary).
    result = store.resolve(
        identifiers=[Identifier("name-only", "some internal dataset")],
    )
    assert result.existing_artifact is not None
