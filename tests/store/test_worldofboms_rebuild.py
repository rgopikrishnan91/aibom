"""POST /worldofboms/rebuild re-ingests every row in bom-history/index.json.

Idempotent: relying on the same store dedup verified in
tests/store/test_e2e_reuse_via_process.py, calling rebuild twice yields
the expected stats (artifacts stay flat, claims grow per rebuild).
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    history = tmp_path / "bom-history"; history.mkdir()
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(graph_dir))
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DISABLE", "0")
    monkeypatch.setenv("AIKABOOM_HISTORY_DIR", str(history))

    # Two fixture BOMs with distinct identifiers → 2 artifacts.
    bom_a = {
        "repo_id": "owner-a/model", "model_id": "owner-a_model",
        "use_case": "license",
        "direct_fields": {"license": {"value": "Apache-2.0", "source": "huggingface", "conflict": None}},
        "rag_fields": {}, "beta_fields": [],
    }
    bom_b = {
        "repo_id": "owner-b/model", "model_id": "owner-b_model",
        "use_case": "license",
        "direct_fields": {"license": {"value": "MIT", "source": "huggingface", "conflict": None}},
        "rag_fields": {}, "beta_fields": [],
    }
    (history / "aaa11111_model_aibom.json").write_text(json.dumps(bom_a))
    (history / "bbb22222_model_aibom.json").write_text(json.dumps(bom_b))
    (history / "index.json").write_text(json.dumps([
        {"hash": "aaa11111", "subject": "owner-a/model", "bom_type": "ai",
         "created_at": "2026-05-20T00:00:00+00:00",
         "artifacts": {"bom": "aaa11111_model_aibom.json"}},
        {"hash": "bbb22222", "subject": "owner-b/model", "bom_type": "ai",
         "created_at": "2026-05-20T00:00:01+00:00",
         "artifacts": {"bom": "bbb22222_model_aibom.json"}},
    ]))

    from aikaboom.web.app import app
    app.config["TESTING"] = True
    app.config["HISTORY_FOLDER"] = str(history)
    app.config["HISTORY_INDEX"] = str(history / "index.json")
    return app.test_client()


def test_rebuild_ingests_every_history_row(client):
    """First rebuild produces 2 artifacts + 2 claims from 2 history rows."""
    r = client.post("/worldofboms/rebuild")
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["processed"] == 2
    assert body["artifacts"] == 2
    assert body["claims"] == 2


def test_rebuild_is_idempotent(client):
    """A second rebuild does not duplicate artifact nodes (each call creates new claims, not artifacts)."""
    client.post("/worldofboms/rebuild")
    r = client.post("/worldofboms/rebuild")
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["processed"] == 2
    assert body["artifacts"] == 2
    # Two rebuilds × two rows = 4 claims (each rebuild writes a fresh
    # claim_iri; the store dedup keeps artifacts flat). If save_claim's
    # contract changes to dedup on (artifact, run_meta), this drops to 2.
    assert body["claims"] == 4


def test_rebuild_skips_row_with_no_bom_artifact(client, tmp_path):
    """A row whose artifacts dict has no `bom` key is skipped silently."""
    # Overwrite the seeded index with one row missing the bom key.
    import json
    history = tmp_path / "bom-history"
    (history / "index.json").write_text(json.dumps([
        {"hash": "ccc33333", "subject": "no-bom/row", "bom_type": "ai",
         "created_at": "2026-05-20T00:00:00+00:00",
         "artifacts": {}},  # ← no 'bom' key
    ]))
    r = client.post("/worldofboms/rebuild")
    assert r.status_code == 200
    body = r.get_json()
    assert body["processed"] == 0
    assert body["failed"] == 0
    assert body["artifacts"] == 0


def test_rebuild_skips_row_with_missing_file(client, tmp_path):
    """A row whose bom file doesn't exist on disk is skipped silently."""
    import json
    history = tmp_path / "bom-history"
    (history / "index.json").write_text(json.dumps([
        {"hash": "ddd44444", "subject": "ghost/row", "bom_type": "ai",
         "created_at": "2026-05-20T00:00:00+00:00",
         "artifacts": {"bom": "ddd44444_doesnt_exist.json"}},
    ]))
    r = client.post("/worldofboms/rebuild")
    assert r.status_code == 200
    body = r.get_json()
    assert body["processed"] == 0
    assert body["failed"] == 0


def test_rebuild_skips_bom_with_no_identifiers(client, tmp_path):
    """A BOM JSON with no repo_id and no row-level identifiers is skipped."""
    import json
    history = tmp_path / "bom-history"
    bom = {"use_case": "license", "direct_fields": {}, "rag_fields": {}, "beta_fields": []}
    (history / "eee55555_model_aibom.json").write_text(json.dumps(bom))
    (history / "index.json").write_text(json.dumps([
        {"hash": "eee55555", "subject": "no/ids", "bom_type": "ai",
         "created_at": "2026-05-20T00:00:00+00:00",
         "artifacts": {"bom": "eee55555_model_aibom.json"}},
    ]))
    r = client.post("/worldofboms/rebuild")
    assert r.status_code == 200
    body = r.get_json()
    assert body["processed"] == 0
