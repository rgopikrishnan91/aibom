"""Shared fixtures for store tests."""
import json
import os
from pathlib import Path
import pytest


SAMPLE_BOM = {
    "repo_id": "mistralai/Mistral-7B-v0.1",
    "model_id": "mistralai_Mistral-7B-v0.1",
    "use_case": "license",
    "direct_fields": {
        "releaseTime": {
            "value": "2025-07-24T16:44:02+00:00",
            "source": "huggingface",
            "conflict": None,
        },
        "suppliedBy": {
            "value": "mistralai",
            "source": "huggingface",
            "conflict": None,
        },
        "packageVersion": {
            "value": "27d67f1b",
            "source": "huggingface",
            "conflict": None,
        },
    },
    "rag_fields": {},
    "beta_fields": [],
}


SAMPLE_RUN_META = {
    "provider": "openrouter",
    "llm_model": "anthropic/claude-3-haiku",
    "prompt_version": "v12",
    "code_version": "abc1234",
    "mode": "rag",
    "use_case": "license",
}


@pytest.fixture
def sample_bom():
    """A minimal-but-realistic BOM JSON dict."""
    return json.loads(json.dumps(SAMPLE_BOM))  # deep copy


@pytest.fixture
def sample_run_meta():
    """A GenerationRun parameter dict."""
    return dict(SAMPLE_RUN_META)


@pytest.fixture
def tmp_store_dir(tmp_path, monkeypatch):
    """An empty graph store dir, configured via env var."""
    store_dir = tmp_path / "graph"
    store_dir.mkdir()
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(store_dir))
    return store_dir


def load_golden_bom(name: str) -> dict:
    """Load a BOM from the project's Golden_Set or results/ directory."""
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in [
        repo_root / "results" / f"{name}.json",
        repo_root / "Golden_Set" / f"{name}.json",
    ]:
        if candidate.exists():
            return json.loads(candidate.read_text())
    raise FileNotFoundError(f"No BOM named {name!r} in results/ or Golden_Set/")
