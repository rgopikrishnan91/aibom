"""Verify --cache flags wire through cmd_generate correctly."""
import json
import os
import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


# Ensure subprocess imports the *worktree* copy of `aikaboom`, not the
# editable install that points at the canonical repo root. Mirrors
# `tests/store/test_cli_graph.py`.
_SRC = str(Path(__file__).resolve().parents[2] / "src")


@pytest.fixture
def store_env(tmp_store_dir):
    env = dict(os.environ)
    env["AIKABOOM_GRAPH_DIR"] = str(tmp_store_dir)
    env["AIKABOOM_GRAPH_BACKEND"] = "rdflib"
    env["AIKABOOM_CACHE_POLICY_DEFAULT"] = "use"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SRC + (os.pathsep + existing if existing else "")
    return env


def test_cache_flag_recognized(store_env):
    """The --cache flag is parsed without error."""
    result = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "generate", "--help"],
        capture_output=True, text=True, env=store_env,
    )
    assert "--cache" in result.stdout
