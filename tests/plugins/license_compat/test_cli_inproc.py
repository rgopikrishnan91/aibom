"""In-process CLI coverage for license_compat.cli.

The subprocess-driven tests in ``test_cli.py`` validate end-to-end exit codes
and stdout, but because they fork a new interpreter, coverage.py records 0%
for ``cli.py``. These tests import the CLI handlers directly so the lines
get marked covered. They reuse the ``lineage_3node_store`` fixture and
``monkeypatch`` ``cli._open_store`` to point at it.
"""
from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout, redirect_stderr

import pytest

from aikaboom.plugins.license_compat import cli as cli_mod
from aikaboom.plugins.license_compat.plugin import LicenseCompatPlugin


@pytest.fixture
def plugin_with_tiny_matrix(tiny_matrix):
    return LicenseCompatPlugin(matrix=tiny_matrix)


@pytest.fixture
def patched_store(monkeypatch, lineage_3node_store):
    monkeypatch.setattr(cli_mod, "_open_store", lambda: lineage_3node_store)
    return lineage_3node_store


def test_register_cli_adds_two_subparsers(plugin_with_tiny_matrix):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    cli_mod.register_cli(sub, plugin_with_tiny_matrix)
    assert "license-check" in sub.choices
    assert "license-audit" in sub.choices


def test_cmd_check_text_output_runs_clean(
    patched_store, plugin_with_tiny_matrix
):
    args = argparse.Namespace(
        artifact="https://example.org/ModelA",
        depth=3,
        format="text",
        matrix=None,
        violations_only=False,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_mod._cmd_check(args, plugin_with_tiny_matrix)
    assert rc in (0, 2)
    out = buf.getvalue()
    assert "ModelA" in out
    assert "Summary" in out


def test_cmd_check_json_output_has_expected_keys(
    patched_store, plugin_with_tiny_matrix
):
    args = argparse.Namespace(
        artifact="https://example.org/ModelA",
        depth=3,
        format="json",
        matrix=None,
        violations_only=False,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli_mod._cmd_check(args, plugin_with_tiny_matrix)
    payload = json.loads(buf.getvalue())
    assert "findings" in payload
    assert "compatible_subchains" in payload
    assert "breaking_nodes" in payload


def test_cmd_check_jsonl_format(patched_store, plugin_with_tiny_matrix):
    args = argparse.Namespace(
        artifact="https://example.org/ModelA",
        depth=3,
        format="jsonl",
        matrix=None,
        violations_only=False,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli_mod._cmd_check(args, plugin_with_tiny_matrix)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    for ln in lines:
        json.loads(ln)


def test_cmd_check_unknown_artifact_exits_3(
    patched_store, plugin_with_tiny_matrix
):
    args = argparse.Namespace(
        artifact="some-non-iri-label-that-does-not-resolve",
        depth=3,
        format="text",
        matrix=None,
        violations_only=False,
    )
    err = io.StringIO()
    with redirect_stderr(err):
        rc = cli_mod._cmd_check(args, plugin_with_tiny_matrix)
    assert rc == 3
    assert "Artifact not found" in err.getvalue()


def test_cmd_audit_writes_jsonl_out_file(
    patched_store, plugin_with_tiny_matrix, tmp_path
):
    out = tmp_path / "audit.jsonl"
    args = argparse.Namespace(
        format="jsonl",
        matrix=None,
        out=out,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli_mod._cmd_audit(args, plugin_with_tiny_matrix)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    # File contains JSONL — each line valid JSON. Empty file is acceptable
    # when there are no findings.
    for line in text.splitlines():
        json.loads(line)


def test_cmd_check_violations_only_renders_violations(
    patched_store, plugin_with_tiny_matrix
):
    # ModelA -> DatasetB (apache-2.0 -> apache-2.0) is compatible, so the
    # violations-only path also exercises the "no items" branch (line 136).
    args = argparse.Namespace(
        artifact="https://example.org/ModelA",
        depth=3,
        format="text",
        matrix=None,
        violations_only=True,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli_mod._cmd_check(args, plugin_with_tiny_matrix)
    # Either "No findings." or a violation block — both exercise the render path.
    out = buf.getvalue()
    assert "Summary" in out


def test_cmd_check_with_matrix_override(
    patched_store, plugin_with_tiny_matrix, tiny_matrix_paths
):
    """Covers ``_override_matrix`` when ``args.matrix`` is non-None."""
    args = argparse.Namespace(
        artifact="https://example.org/ModelA",
        depth=3,
        format="text",
        matrix=tiny_matrix_paths["matrix"],
        violations_only=False,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_mod._cmd_check(args, plugin_with_tiny_matrix)
    assert rc in (0, 2)


def test_resolve_artifact_iri_returns_iri_when_passed_url(patched_store):
    assert (
        cli_mod._resolve_artifact_iri(patched_store, "https://example.org/X")
        == "https://example.org/X"
    )


def test_resolve_artifact_iri_resolves_canonical_label(patched_store):
    iri = cli_mod._resolve_artifact_iri(patched_store, "ModelA")
    assert iri == "https://example.org/ModelA"


def test_resolve_artifact_iri_returns_none_for_unknown_label(patched_store):
    assert cli_mod._resolve_artifact_iri(patched_store, "TotallyBogus") is None


def test_cmd_check_text_no_findings_branch(
    patched_store, plugin_with_tiny_matrix
):
    """PaperC is a leaf in the fixture — no upstreams — so violations_only
    enters the ``No findings.`` branch (cli.py:136)."""
    args = argparse.Namespace(
        artifact="https://example.org/PaperC",
        depth=3,
        format="text",
        matrix=None,
        violations_only=True,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli_mod._cmd_check(args, plugin_with_tiny_matrix)
    assert "No findings." in buf.getvalue()
