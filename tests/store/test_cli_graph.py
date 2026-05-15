"""CLI tests for `aikaboom graph` subcommands."""
import os
import subprocess
import sys
from pathlib import Path


# Ensure subprocess imports the *worktree* copy of `aikaboom`, not the
# editable install that points at the canonical repo root. Without this,
# the new `graph`/`bom` subcommands appear missing in subprocess runs.
_SRC = str(Path(__file__).resolve().parents[2] / "src")


def run_cli(args, env=None):
    final_env = dict(env) if env is not None else dict(os.environ)
    existing = final_env.get("PYTHONPATH", "")
    final_env["PYTHONPATH"] = _SRC + (os.pathsep + existing if existing else "")
    return subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", *args],
        capture_output=True, text=True, env=final_env,
    )


def test_graph_stats_runs(tmp_store_dir, monkeypatch):
    env = {**dict(os.environ), "AIKABOOM_GRAPH_DIR": str(tmp_store_dir),
           "AIKABOOM_GRAPH_BACKEND": "rdflib"}
    result = run_cli(["graph", "stats"], env=env)
    assert result.returncode == 0, result.stderr
    assert "claims" in result.stdout or "Claims" in result.stdout


def test_graph_export_import_roundtrip(tmp_store_dir, tmp_path):
    env = {**dict(os.environ), "AIKABOOM_GRAPH_DIR": str(tmp_store_dir),
           "AIKABOOM_GRAPH_BACKEND": "rdflib"}
    dump = tmp_path / "dump.nq"
    result = run_cli(["graph", "export", str(dump)], env=env)
    assert result.returncode == 0
    assert dump.exists()
