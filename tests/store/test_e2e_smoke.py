"""End-to-end: stats on empty store, export empty, import-export roundtrip."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure subprocess imports the *worktree* copy of `aikaboom`, not the
# editable install that points at the canonical repo root. Without this,
# the new `graph`/`bom` subcommands appear missing in subprocess runs.
_SRC = str(Path(__file__).resolve().parents[2] / "src")


@pytest.fixture
def isolated_env(tmp_path):
    env = dict(os.environ)
    env["AIKABOOM_GRAPH_DIR"] = str(tmp_path / "graph")
    env["AIKABOOM_GRAPH_BACKEND"] = "rdflib"
    env["AIKABOOM_GRAPH_DISABLE"] = "0"
    env["AIKABOOM_CACHE_POLICY_DEFAULT"] = "use"  # silent caching
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SRC + (os.pathsep + existing if existing else "")
    return env


def test_stats_starts_empty(isolated_env):
    result = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "graph", "stats"],
        capture_output=True,
        text=True,
        env=isolated_env,
    )
    assert result.returncode == 0, result.stderr
    stats = json.loads(result.stdout)
    assert stats["claims"] == 0


def test_export_empty_graph_succeeds(isolated_env, tmp_path):
    dump = tmp_path / "empty.nq"
    result = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "graph", "export", str(dump)],
        capture_output=True,
        text=True,
        env=isolated_env,
    )
    assert result.returncode == 0, result.stderr
    assert dump.exists()


def test_import_export_roundtrip(isolated_env, tmp_path):
    # Seed the graph with a trivial quad.
    dump = tmp_path / "seed.nq"
    dump.write_text(
        '<bom:artifact/x> <https://aikaboom.dev/aibom#canonicalLabel> "Test" '
        "<urn:x-arq:DefaultGraphNode> .\n"
    )
    r1 = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "graph", "import", str(dump)],
        capture_output=True,
        text=True,
        env=isolated_env,
    )
    assert r1.returncode == 0, r1.stderr
    out = tmp_path / "out.nq"
    r2 = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "graph", "export", str(out)],
        capture_output=True,
        text=True,
        env=isolated_env,
    )
    assert r2.returncode == 0, r2.stderr
    assert out.stat().st_size > 0
