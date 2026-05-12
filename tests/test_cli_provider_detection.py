"""
Tests for CLI auto-detection of LLM provider based on environment variables.
"""
import os
import pytest
import subprocess
import sys
from unittest.mock import patch


@pytest.fixture
def clean_env():
    """Strip all provider env vars for the test."""
    keys = ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_BASE_URL"]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


def _run_cli(env_overrides, args, cwd=None):
    """Run the CLI as a subprocess with explicit environment, return CompletedProcess.

    Sets ``BOM_SKIP_DOTENV=1`` so the CLI does not load the repo's ``.env`` at
    import time — the file is found by walking up from ``aikaboom/cli.py``
    regardless of subprocess cwd, so the explicit ``env_overrides`` we pass
    in would otherwise be overridden by the developer's real keys.
    """
    env = os.environ.copy()
    for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_BASE_URL"):
        env.pop(k, None)
    env["BOM_SKIP_DOTENV"] = "1"
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "aikaboom.cli"] + args,
        capture_output=True, text=True, env=env, timeout=30,
        cwd=str(cwd) if cwd is not None else None,
    )


class TestProviderAutoDetection:
    """Subprocess tests use ``tmp_path`` as cwd so the CLI's ``load_dotenv()``
    call doesn't pick up the developer's real ``.env`` and contaminate the
    ``env_overrides`` we explicitly set."""

    def test_no_keys_set_errors_out(self, tmp_path):
        result = _run_cli({}, ["generate", "--type", "ai", "--repo", "test/m"], cwd=tmp_path)
        assert result.returncode != 0
        assert "no LLM provider credentials" in result.stderr

    def test_only_openrouter_uses_openrouter(self, tmp_path):
        # The processing itself will fail (no real key) but we check the
        # provider-detection log line was emitted before that point.
        result = _run_cli(
            {"OPENROUTER_API_KEY": "sk-or-v1-fake"},
            ["generate", "--type", "ai", "--repo", "test/m", "--yes"],
            cwd=tmp_path,
        )
        assert "Using openrouter" in result.stdout

    def test_only_ollama_uses_ollama(self, tmp_path):
        result = _run_cli(
            {"OLLAMA_BASE_URL": "http://localhost:11434/v1/"},
            ["generate", "--type", "ai", "--repo", "test/m", "--yes"],
            cwd=tmp_path,
        )
        assert "Using ollama" in result.stdout

    def test_explicit_provider_without_key_errors(self, tmp_path):
        result = _run_cli(
            {},
            ["generate", "--type", "ai", "--repo", "test/m", "--provider", "openai"],
            cwd=tmp_path,
        )
        assert result.returncode != 0
        assert "OPENAI_API_KEY" in result.stderr

    def test_yes_flag_skips_prompt_with_multiple_keys(self, tmp_path):
        # Both OpenAI and OpenRouter set; --yes should auto-pick (prefers
        # openrouter over openai per the resolver's preference order).
        result = _run_cli(
            {"OPENAI_API_KEY": "sk-fake", "OPENROUTER_API_KEY": "sk-or-fake"},
            ["generate", "--type", "ai", "--repo", "test/m", "--yes"],
            cwd=tmp_path,
        )
        # Should have picked openrouter (preferred over openai when both set)
        assert "openrouter" in result.stdout.lower()


class TestProviderResolverUnit:
    """Unit tests for the resolver helpers without subprocess overhead.

    These tests use ``monkeypatch.delenv()`` AFTER the ``aikaboom.cli``
    import so the import-time ``load_dotenv()`` cannot reintroduce the
    developer's real ``.env`` keys between fixture setup and the assertion.
    """

    def test_detect_available_only_openai(self, monkeypatch):
        from aikaboom import cli
        for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_BASE_URL"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        assert cli._detect_available_providers() == ["openai"]

    def test_detect_available_multiple(self, monkeypatch):
        from aikaboom import cli
        for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_BASE_URL"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.setenv("OPENROUTER_API_KEY", "y")
        avail = cli._detect_available_providers()
        assert "openai" in avail
        assert "openrouter" in avail

    def test_detect_available_none(self, monkeypatch):
        from aikaboom import cli
        for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_BASE_URL"):
            monkeypatch.delenv(k, raising=False)
        assert cli._detect_available_providers() == []
