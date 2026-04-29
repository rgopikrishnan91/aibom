"""Tests for the aikaboom list-models subcommand and --pick-free-model flag."""
import json
import os
import subprocess
import sys


def _run_cli(args, env_overrides=None, timeout=30):
    env = os.environ.copy()
    # Strip provider env so detection is deterministic
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
        assert "--free" in r.stdout
        assert "--limit" in r.stdout
        assert "--json" in r.stdout

    def test_free_prints_models(self):
        r = _run_cli(["list-models", "--free"])
        assert r.returncode == 0
        # Should print at least the curated fallback ids in offline environments
        assert ":free" in r.stdout

    def test_limit(self):
        r = _run_cli(["list-models", "--free", "--limit", "2"])
        assert r.returncode == 0
        # 2 model rows + 2 header lines (header + separator)
        # We don't pin exact lines, just assert at most 2 :free occurrences
        # in the output (the curated fallback has 5).
        free_lines = [ln for ln in r.stdout.splitlines() if ":free" in ln]
        assert len(free_lines) <= 2

    def test_json_output(self):
        r = _run_cli(["list-models", "--free", "--json"])
        assert r.returncode == 0
        # Locate the JSON portion (output may include a warning line first)
        start = r.stdout.find("[")
        assert start != -1, f"No JSON in output: {r.stdout!r}"
        data = json.loads(r.stdout[start:])
        assert isinstance(data, list)
        assert all(isinstance(m, dict) and "id" in m for m in data)


class TestPickFreeModelFlag:

    def test_pick_free_model_with_explicit_model_errors(self):
        r = _run_cli([
            "generate", "--type", "ai", "--repo", "test/m",
            "--pick-free-model", "--model", "gpt-4o",
        ])
        assert r.returncode != 0
        assert "mutually exclusive" in r.stderr.lower()

    def test_pick_free_model_with_wrong_provider_errors(self):
        r = _run_cli([
            "generate", "--type", "ai", "--repo", "test/m",
            "--pick-free-model", "--provider", "openai",
        ])
        assert r.returncode != 0
        assert "openrouter" in r.stderr.lower()

    def test_pick_free_model_picks_one(self):
        # No env keys set; --pick-free-model forces openrouter, but we don't
        # have a key, so we expect the resolver to bail with a helpful error.
        # The key assertion is that pick_free_openrouter_model() ran and
        # printed a model id, _then_ failed at the provider-key check.
        r = _run_cli([
            "generate", "--type", "ai", "--repo", "test/m",
            "--pick-free-model",
        ])
        assert "Picked free OpenRouter model:" in r.stdout
