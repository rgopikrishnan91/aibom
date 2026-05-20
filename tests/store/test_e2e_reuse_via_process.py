"""End-to-end: a second /process for the same model reuses the first artifact.

The store's de-dup story is exercised piecewise in test_multi_identifier_dedup,
test_store_resolve, test_edges, etc. This test closes the loop by driving the
public Flask /process route twice and asserting BomStore.stats() shows the
expected node counts (no duplicates).

Per the worldofBOMs followups spec
(docs/superpowers/specs/2026-05-20-worldofboms-followups-design.md), section B.
"""

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A Flask test client with a real (per-test) graph store on disk."""
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(graph_dir))
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DISABLE", "0")
    # Skip the link-fallback agent — no Gemini key in tests, and we don't
    # want the route's exception path muddying the assertions.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Re-route bom-history so the test doesn't pollute the repo copy.
    history_dir = tmp_path / "bom-history"
    history_dir.mkdir()

    # Re-route file-output dirs so we don't write into the project tree.
    upload_dir = tmp_path / "results"
    upload_dir.mkdir()
    repo_results_dir = tmp_path / "data" / "results"
    repo_results_dir.mkdir(parents=True)

    from aikaboom.web.app import app

    app.config["TESTING"] = True
    app.config["HISTORY_FOLDER"] = str(history_dir)
    app.config["HISTORY_INDEX"] = str(history_dir / "index.json")
    app.config["UPLOAD_FOLDER"] = str(upload_dir)
    app.config["REPO_RESULTS_FOLDER"] = str(repo_results_dir)
    return app.test_client()


@pytest.fixture
def fake_ai_processor(monkeypatch):
    """Patch get_processor so /process never calls an LLM.

    Returns a callable that takes the repo_id to embed in the fake BOM, so
    each scenario can pin its own model name.
    """
    from aikaboom.web import app as appmod

    class _FakeAIProc:
        use_case = "license"

        def __init__(self, repo_id: str):
            self._repo_id = repo_id

        def process_ai_model(self, **_kwargs):
            return {
                "repo_id": self._repo_id,
                "model_id": self._repo_id.replace("/", "_"),
                "use_case": "license",
                "direct_fields": {
                    "license": {
                        "value": "Apache-2.0",
                        "source": "huggingface",
                        "conflict": None,
                    },
                },
                "rag_fields": {},
                "beta_fields": [],
            }

    def install(repo_id: str):
        monkeypatch.setattr(
            appmod, "get_processor",
            lambda **_kw: _FakeAIProc(repo_id),
        )

    return install


def _post_process(client, repo_id: str):
    """POST /process for an AI BOM with a single repo_id identifier.

    Uses cache_policy=regen so every call writes a new claim — the
    test exercises store-level de-dup (artifacts stay 1) not cache-hit
    short-circuiting (which would keep claims at 1).
    """
    resp = client.post(
        "/process",
        json={
            "bom_type": "ai",
            "repo_id": repo_id,
            "use_case": "license",
            "mode": "rag",
            "cache_policy": "regen",
            "skip_fallback": True,  # avoid GEMINI_API_KEY path
            "validate_spdx": False,
        },
        content_type="application/json",
    )
    return resp


def _open_store():
    """Open the live BomStore against the env-configured backend."""
    from aikaboom.store.store import BomStore
    return BomStore.open()


def test_identical_model_reuses_artifact(client, fake_ai_processor):
    """Two /process calls for the same repo_id → 1 artifact, 2 claims."""
    fake_ai_processor("mistralai/Mistral-7B-v0.1")

    r1 = _post_process(client, "mistralai/Mistral-7B-v0.1")
    assert r1.status_code == 200, r1.get_data(as_text=True)

    r2 = _post_process(client, "mistralai/Mistral-7B-v0.1")
    assert r2.status_code == 200, r2.get_data(as_text=True)

    stats = _open_store().stats()
    assert stats["artifacts"] == 1, f"expected 1 artifact, got {stats}"
    assert stats["claims"] == 2, f"expected 2 claims, got {stats}"


def test_cross_identifier_reuses_artifact(client, fake_ai_processor):
    """BOM 1 saved with (hf=X, arxiv=Y); BOM 2 with only arxiv=Y → 1 artifact, 2 claims."""
    repo_id = "mistralai/Mistral-7B-v0.1"
    arxiv_url = "https://arxiv.org/abs/2310.06825"

    fake_ai_processor(repo_id)
    r1 = client.post(
        "/process",
        json={
            "bom_type": "ai",
            "repo_id": repo_id,
            "arxiv_url": arxiv_url,
            "use_case": "license", "mode": "rag",
            "cache_policy": "regen",
            "skip_fallback": True, "validate_spdx": False,
        },
        content_type="application/json",
    )
    assert r1.status_code == 200, r1.get_data(as_text=True)

    # Second run — same fake processor, but the request body omits repo_id
    # so only the arxiv identifier is supplied. The store's resolve() must
    # still find the artifact saved above.
    r2 = client.post(
        "/process",
        json={
            "bom_type": "ai",
            "arxiv_url": arxiv_url,
            "use_case": "license", "mode": "rag",
            "cache_policy": "regen",
            "skip_fallback": True, "validate_spdx": False,
        },
        content_type="application/json",
    )
    assert r2.status_code == 200, r2.get_data(as_text=True)

    stats = _open_store().stats()
    assert stats["artifacts"] == 1, f"expected 1 artifact, got {stats}"
    assert stats["claims"] == 2, f"expected 2 claims, got {stats}"


def test_dependency_edge_reuses_existing_artifact(client, fake_ai_processor, monkeypatch):
    """BOM A saved for model M; BOM B for M' lists M as trainedOn → edge points to M's IRI, no duplicate."""
    from aikaboom.web import app as appmod

    # First run: stash artifact for the dependency target.
    fake_ai_processor("upstream/teacher-model")
    r1 = _post_process(client, "upstream/teacher-model")
    assert r1.status_code == 200, r1.get_data(as_text=True)

    # Second run: BOM for the student model that lists the teacher in its
    # trainedOnDatasets field. Reinstall the fake processor to return a BOM
    # whose direct_fields carry the relationship target.
    class _StudentProc:
        use_case = "license"

        def process_ai_model(self, **_kwargs):
            return {
                "repo_id": "downstream/student-model",
                "model_id": "downstream_student-model",
                "use_case": "license",
                "direct_fields": {
                    "license": {"value": "Apache-2.0", "source": "huggingface", "conflict": None},
                    "trainedOnDatasets": {
                        "value": "upstream/teacher-model",
                        "source": "huggingface",
                        "conflict": None,
                    },
                },
                "rag_fields": {},
                "beta_fields": [],
            }

    monkeypatch.setattr(appmod, "get_processor", lambda **_kw: _StudentProc())
    r2 = _post_process(client, "downstream/student-model")
    assert r2.status_code == 200, r2.get_data(as_text=True)

    stats = _open_store().stats()
    # 2 artifacts (teacher + student), 2 claims. No third "ghost" artifact
    # for the teacher reference inside the student BOM.
    assert stats["artifacts"] == 2, f"expected 2 artifacts, got {stats}"
    assert stats["claims"] == 2, f"expected 2 claims, got {stats}"

    # And the trainedOn edge points to the teacher's IRI, not a fresh one.
    from aikaboom.store import vocab
    store = _open_store()
    rows = list(store._backend.select(f"""
        SELECT ?src ?tgt WHERE {{
            ?src <{vocab.trainedOn}> ?tgt .
        }}
    """))
    assert len(rows) == 1, f"expected 1 trainedOn edge, got {rows}"
    teacher_iri = rows[0]["tgt"]
    # The teacher IRI must appear as the subject of a hasVersion/hasClaim
    # chain — i.e., it's the teacher artifact we saved in r1, not a
    # placeholder created on the fly.
    has_claims = list(store._backend.select(f"""
        SELECT ?v WHERE {{
            <{teacher_iri}> <{vocab.hasVersion}> ?v .
        }}
    """))
    assert has_claims, f"trainedOn target {teacher_iri} has no version/claim — duplicate artifact"
