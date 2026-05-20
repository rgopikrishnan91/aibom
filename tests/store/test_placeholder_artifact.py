"""Placeholder artifacts for unresolvable recursive references."""

import pytest

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    return BomStore.open()


def test_placeholder_artifact_marker_is_written(store, sample_bom, sample_run_meta):
    """An artifact created with platform='name-only' is marked aibom:isPlaceholder true."""
    from aikaboom.store import vocab

    store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("name-only", "some internal dataset")],
    )
    # Verify the artifact still resolves by the same name-only identifier.
    result = store.resolve(
        identifiers=[Identifier("name-only", "some internal dataset")],
    )
    assert result.existing_artifact is not None
    # And verify the isPlaceholder marker exists on the artifact.
    rows = list(
        store._backend.select(
            f"SELECT ?p WHERE {{ <{result.existing_artifact}> <{vocab.isPlaceholder}> ?p }}"
        )
    )
    assert len(rows) == 1
    assert str(rows[0]["p"]).lower() in ("true", "1")
