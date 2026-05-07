"""Tests for the sbom-utility-backed CycloneDX validator wrapper.

Three test surfaces:
  1. Binary-missing path returns a "skipped" result (no raise).
  2. Mocked subprocess.run handles success and failure cleanly.
  3. Real-binary integration test, guarded by ``pytest.skip`` when the
     binary isn't on PATH — runs against a known-good CycloneDX fixture.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from aikaboom.utils.cyclonedx_validator import (
    BINARY,
    INSTALL_HINT,
    is_available,
    validate_cyclonedx,
)


_KNOWN_GOOD_BOM = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.6",
    "serialNumber": "urn:uuid:12345678-1234-4234-8234-123456789012",
    "version": 1,
    "metadata": {
        "timestamp": "2026-01-01T00:00:00Z",
        "tools": [{"name": "AIkaBoOM", "version": "1.0.0"}],
    },
    "components": [
        {
            "type": "machine-learning-model",
            "bom-ref": "ai-model:test",
            "name": "test-model",
        }
    ],
}


# ---------------------------------------------------------------------------
# Binary-missing path
# ---------------------------------------------------------------------------


def test_returns_skipped_when_binary_absent():
    with patch("aikaboom.utils.cyclonedx_validator.shutil.which", return_value=None):
        result = validate_cyclonedx(_KNOWN_GOOD_BOM)
    assert result["valid"] is None
    assert result["validator"] == "skipped"
    assert "sbom-utility" in result["reason"]
    assert result["errors"] == []


def test_install_hint_mentions_install_script():
    assert "install_sbom_utility.sh" in INSTALL_HINT


# ---------------------------------------------------------------------------
# Mocked subprocess
# ---------------------------------------------------------------------------


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        args=[BINARY], returncode=returncode, stdout=stdout, stderr=stderr,
    )


def test_clean_run_returns_valid_true():
    with patch("aikaboom.utils.cyclonedx_validator.shutil.which", return_value="/usr/bin/sbom-utility"), \
         patch("aikaboom.utils.cyclonedx_validator.subprocess.run", return_value=_completed()):
        result = validate_cyclonedx(_KNOWN_GOOD_BOM)
    assert result == {"valid": True, "errors": [], "validator": "sbom-utility"}


def test_failed_run_parses_stderr_into_errors():
    stderr = "schema error: components[0].name is required\nschema error: foo bad\n"
    with patch("aikaboom.utils.cyclonedx_validator.shutil.which", return_value="/usr/bin/sbom-utility"), \
         patch("aikaboom.utils.cyclonedx_validator.subprocess.run",
               return_value=_completed(stderr=stderr, returncode=1)):
        result = validate_cyclonedx(_KNOWN_GOOD_BOM)
    assert result["valid"] is False
    assert result["validator"] == "sbom-utility"
    # Each non-empty stderr line becomes a discrete error
    assert any("components[0].name is required" in e for e in result["errors"])
    assert any("foo bad" in e for e in result["errors"])


def test_failed_run_with_silent_stderr_fabricates_an_error():
    """Defensive: if sbom-utility exits non-zero with no stderr, we still
    surface an error rather than silently reporting valid=False with no
    diagnostic."""
    with patch("aikaboom.utils.cyclonedx_validator.shutil.which", return_value="/usr/bin/sbom-utility"), \
         patch("aikaboom.utils.cyclonedx_validator.subprocess.run",
               return_value=_completed(returncode=2)):
        result = validate_cyclonedx(_KNOWN_GOOD_BOM)
    assert result["valid"] is False
    assert result["errors"] == ["sbom-utility exited with code 2"]


def test_timeout_returns_invalid_with_message():
    with patch("aikaboom.utils.cyclonedx_validator.shutil.which", return_value="/usr/bin/sbom-utility"), \
         patch("aikaboom.utils.cyclonedx_validator.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd=BINARY, timeout=30)):
        result = validate_cyclonedx(_KNOWN_GOOD_BOM)
    assert result["valid"] is False
    assert any("timed out" in e for e in result["errors"])


def test_path_input_is_passed_through():
    """When a Path/str is passed instead of a dict, it's used directly
    without re-serializing."""
    captured = {}

    def fake_run(cmd, **_kw):
        captured["cmd"] = cmd
        return _completed()

    with patch("aikaboom.utils.cyclonedx_validator.shutil.which", return_value="/usr/bin/sbom-utility"), \
         patch("aikaboom.utils.cyclonedx_validator.subprocess.run", side_effect=fake_run):
        validate_cyclonedx("/tmp/bom.json")
    assert captured["cmd"][1:4] == ["validate", "-i", "/tmp/bom.json"]


# ---------------------------------------------------------------------------
# Real-binary integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_available(), reason="sbom-utility binary not installed")
def test_real_binary_passes_known_good(tmp_path):
    """If sbom-utility is installed, a structurally valid CycloneDX 1.6
    BOM should pass. This locks the binary's expected behaviour against
    a fixture we control."""
    fixture = tmp_path / "bom.json"
    fixture.write_text(json.dumps(_KNOWN_GOOD_BOM))
    result = validate_cyclonedx(fixture)
    # Real binary must agree this fixture is valid; if not, the schema
    # version we picked here is wrong and the fixture needs updating.
    assert result["valid"] is True, result["errors"]
