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


def _run_cli(env_overrides, args):
    """Run the CLI as a subprocess with explicit environment, return CompletedProcess."""
    env = os.environ.copy()
    for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_BASE_URL"):
        env.pop(k, None)
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "aikaboom.cli"] + args,
        capture_output=True, text=True, env=env, timeout=30
    )


class TestProviderAutoDetection:

    def test_no_keys_set_errors_out(self):
        result = _run_cli({}, ["generate", "--type", "ai", "--repo", "test/m"])
        assert result.returncode != 0
        assert "no LLM provider credentials" in result.stderr

    def test_only_openrouter_uses_openrouter(self):
        # The processing itself will fail (no real key) but we check the
        # provider-detection log line was emitted before that point.
        result = _run_cli(
            {"OPENROUTER_API_KEY": "sk-or-v1-fake"},
            ["generate", "--type", "ai", "--repo", "test/m", "--yes"]
        )
        assert "Using openrouter" in result.stdout

    def test_only_ollama_uses_ollama(self):
        result = _run_cli(
            {"OLLAMA_BASE_URL": "http://localhost:11434/v1/"},
            ["generate", "--type", "ai", "--repo", "test/m", "--yes"]
        )
        assert "Using ollama" in result.stdout

    def test_explicit_provider_without_key_errors(self):
        result = _run_cli(
            {},
            ["generate", "--type", "ai", "--repo", "test/m", "--provider", "openai"]
        )
        assert result.returncode != 0
        assert "OPENAI_API_KEY" in result.stderr

    def test_yes_flag_skips_prompt_with_multiple_keys(self):
        # Both OpenAI and OpenRouter set; --yes should auto-pick (prefers
        # openrouter over openai per the resolver's preference order).
        result = _run_cli(
            {"OPENAI_API_KEY": "sk-fake", "OPENROUTER_API_KEY": "sk-or-fake"},
            ["generate", "--type", "ai", "--repo", "test/m", "--yes"]
        )
        # Should have picked openrouter (preferred over openai when both set)
        assert "openrouter" in result.stdout.lower()


class TestProviderResolverUnit:
    """Unit tests for the resolver helpers without subprocess overhead."""

    def test_detect_available_only_openai(self, clean_env):
        os.environ["OPENAI_API_KEY"] = "x"
        from aikaboom import cli
        assert cli._detect_available_providers() == ["openai"]

    def test_detect_available_multiple(self, clean_env):
        os.environ["OPENAI_API_KEY"] = "x"
        os.environ["OPENROUTER_API_KEY"] = "y"
        from aikaboom import cli
        avail = cli._detect_available_providers()
        assert "openai" in avail
        assert "openrouter" in avail

    def test_detect_available_none(self, clean_env):
        from aikaboom import cli
        assert cli._detect_available_providers() == []
