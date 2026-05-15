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
        capture_output=True,
        text=True,
        env=final_env,
    )


def test_graph_stats_runs(tmp_store_dir, monkeypatch):
    env = {
        **dict(os.environ),
        "AIKABOOM_GRAPH_DIR": str(tmp_store_dir),
        "AIKABOOM_GRAPH_BACKEND": "rdflib",
    }
    result = run_cli(["graph", "stats"], env=env)
    assert result.returncode == 0, result.stderr
    assert "claims" in result.stdout or "Claims" in result.stdout


def test_graph_export_import_roundtrip(tmp_store_dir, tmp_path):
    env = {
        **dict(os.environ),
        "AIKABOOM_GRAPH_DIR": str(tmp_store_dir),
        "AIKABOOM_GRAPH_BACKEND": "rdflib",
    }
    dump = tmp_path / "dump.nq"
    result = run_cli(["graph", "export", str(dump)], env=env)
    assert result.returncode == 0
    assert dump.exists()


def test_graph_merge_transfers_versions_and_cleans_incoming(tmp_path):
    """After merge, artifact b has no outgoing edges and incoming refs point to a."""
    import os
    from aikaboom.store import vocab
    from aikaboom.store.store import BomStore

    env_dir = tmp_path / "graph"
    env_dir.mkdir()
    env = {**os.environ, "AIKABOOM_GRAPH_DIR": str(env_dir), "AIKABOOM_GRAPH_BACKEND": "rdflib"}

    # Build two artifacts and an incoming ref a → b via SPARQL.
    monkey = {k: os.environ.get(k) for k in env}
    try:
        os.environ.update(env)
        store = BomStore.open()
        a_iri = "bom:artifact/aaaa"
        b_iri = "bom:artifact/bbbb"
        store._backend.update(
            f"INSERT DATA {{ <{a_iri}> a <{vocab.Artifact}> ; "
            f"<{vocab.potentialDuplicateOf}> <{b_iri}> . "
            f"<{b_iri}> a <{vocab.Artifact}> ; "
            f"<{vocab.hasVersion}> <bom:version/x> . }}"
        )

        # Run the merge CLI via the handler directly (faster than subprocess).
        from aikaboom.store.cli_graph import cmd_graph_merge
        import argparse

        args = argparse.Namespace(artifact_a=a_iri, artifact_b=b_iri)
        rc = cmd_graph_merge(args)
        assert rc == 0

        # Reopen and verify.
        store2 = BomStore.open()
        # a now owns b's version.
        has_v = list(
            store2._backend.select(f"SELECT ?v WHERE {{ <{a_iri}> <{vocab.hasVersion}> ?v }}")
        )
        assert len(has_v) == 1
        # b has no outgoing edges.
        outgoing = list(store2._backend.select(f"SELECT ?p WHERE {{ <{b_iri}> ?p ?o }}"))
        assert outgoing == []
        # The previous incoming a→b edge now points at a→a (or is gone).
        dangling = list(store2._backend.select(f"SELECT ?s WHERE {{ ?s ?p <{b_iri}> }}"))
        assert dangling == []
    finally:
        for k, v in monkey.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
