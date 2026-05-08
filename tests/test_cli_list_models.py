"""Tests for the ``aikaboom list-models`` subcommand.

Phase 10 retired the ``--free`` flag (free-model filtering) and the
``--pick-free-model`` flag on ``generate`` (auto-picker). See
``tests/test_phase10_fixes.py::test_free_flag_no_longer_in_cli_argparse``
for the regression guarding the removal.
"""
import json
import os
import subprocess
import sys


def _run_cli(args, env_overrides=None, timeout=30):
    env = os.environ.copy()
    # Strip provider env so detection is deterministic.
    for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_BASE_URL"):
        env.pop(k, None)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "aikaboom.cli"] + args,
        capture_output=True, text=True, env=env, timeout=timeout
    )


class TestListModelsCommand:

    def test_help(self):
        r = _run_cli(["list-models", "--help"])
        assert r.returncode == 0
        # Phase 10 retired --free; --limit and --json remain.
        assert "--free" not in r.stdout
        assert "--limit" in r.stdout
        assert "--json" in r.stdout

    def test_default_runs(self):
        """In offline environments the catalog returns ``[]`` rather than
        the curated fallback (Phase 10). The command exits cleanly and
        prints "No models returned."""
        r = _run_cli(["list-models"])
        assert r.returncode == 0

    def test_limit(self):
        """--limit still caps row count even with an empty catalog."""
        r = _run_cli(["list-models", "--limit", "2"])
        assert r.returncode == 0

    def test_json_output(self):
        r = _run_cli(["list-models", "--json"])
        assert r.returncode == 0
        start = r.stdout.find("[")
        assert start != -1, f"No JSON in output: {r.stdout!r}"
        data = json.loads(r.stdout[start:])
        assert isinstance(data, list)
        # Catalog may be empty in this environment; that's fine.
        for m in data:
            assert isinstance(m, dict) and "id" in m


class TestPickFreeModelFlagRetired:
    """Phase 10 retired ``--pick-free-model``; sending it now produces an
    argparse error (the flag is unrecognized)."""

    def test_pick_free_model_flag_unrecognized(self):
        r = _run_cli([
            "generate", "--type", "ai", "--repo", "test/m",
            "--pick-free-model",
        ])
        assert r.returncode != 0
        # argparse prints "unrecognized arguments" or similar.
        assert "--pick-free-model" in (r.stderr + r.stdout) or "unrecognized" in r.stderr.lower()
