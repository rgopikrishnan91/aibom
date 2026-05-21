"""Plugin contract tests for AvidSecurityPlugin.

Tests that every hook exists, returns the documented type, and wires
through correctly to the core modules (snapshot, matcher, walker, emitter).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aikaboom.plugins import GraphOverlay, Plugin, Scope, TabSpec
from aikaboom.plugins.avid_security.engine import AvidFinding, AvidFindings
from aikaboom.plugins.avid_security.plugin import AvidSecurityPlugin

# ── fixtures ───────────────────────────────────────────────────────────────


FIXTURES = Path(__file__).parent / "fixtures"

_BERT_REPORT = {
    "metadata": {"report_id": "AVID-2022-R0001"},
    "affects": {
        "developer": [],
        "deployer": ["HuggingFace"],
        "artifacts": [{"type": "Model", "name": "bert-base-uncased"}],
    },
    "impact": {
        "avid": {
            "risk_domain": ["Ethics"],
            "sep_view": ["E0101: Group fairness"],
            "lifecycle_view": ["L05: Evaluation"],
        }
    },
    "reported_date": "2022-11-09",
    "description": {"value": "Sample AVID description for testing"},
    "problemtype": {"description": {"value": "Gender bias in BERT"}},
}


@pytest.fixture
def seeded_cache(tmp_path):
    """Seed a fake AVID snapshot cache so plugin.analyze() never hits the network.

    Writes:
      - <cache>/snapshot.json  (marker)
      - <cache>/avid-db/reports/2022/AVID-2022-R0001.json  (one report)
    Then builds the SQLite index from that repo dir.
    """
    cache_dir = tmp_path / "avid"
    repo_dir = cache_dir / "avid-db"
    report_dir = repo_dir / "reports" / "2022"
    report_dir.mkdir(parents=True)
    (report_dir / "AVID-2022-R0001.json").write_text(json.dumps(_BERT_REPORT))

    # Write the marker so ensure_fresh() sees a fresh snapshot and skips cloning.
    from datetime import datetime, timezone
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "snapshot.json").write_text(json.dumps({
        "sha": "abc123deadbeef",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "ttl_days": 10,
    }))

    # Pre-build the SQLite index.
    from aikaboom.plugins.avid_security.snapshot import AvidIndex
    idx = AvidIndex(db_path=cache_dir / "avid.sqlite")
    idx.build(repo_dir=repo_dir)

    return cache_dir


@pytest.fixture
def avid_corpus_store(tmp_path, monkeypatch):
    """BomStore populated from avid_bom_corpus.ttl (same fixture as test_walker_store.py)."""
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_path))

    from rdflib import Graph
    from aikaboom.store import BomStore

    store = BomStore.open()
    g = Graph()
    g.parse(str(FIXTURES / "avid_bom_corpus.ttl"), format="turtle")
    store._backend.add_quads([(s, p, o, None) for s, p, o in g])
    yield store
    store._backend.close()


def _make_finding(tier: int, report_id: str = "AVID-2022-R0001") -> AvidFinding:
    confidence = {1: "high", 2: "medium", 3: "low"}[tier]
    return AvidFinding(
        component_iri="https://example.org/BertBase",
        component_label="bert-base-uncased",
        avid_report_id=report_id,
        tier=tier,
        confidence=confidence,
        matched_via="exact_bare_name" if tier == 1 else "base_model_lineage",
        match=object(),
    )


# ── protocol conformance ───────────────────────────────────────────────────


def test_plugin_satisfies_protocol():
    assert isinstance(AvidSecurityPlugin(), Plugin)


# ── enabled() ─────────────────────────────────────────────────────────────


def test_enabled_default_is_true(monkeypatch):
    monkeypatch.delenv("AIKABOOM_AVID_DISABLED", raising=False)
    assert AvidSecurityPlugin().enabled() is True


def test_enabled_false_when_env_set(monkeypatch):
    monkeypatch.setenv("AIKABOOM_AVID_DISABLED", "1")
    assert AvidSecurityPlugin().enabled() is False


# ── bom_viewer_tab() ──────────────────────────────────────────────────────


def test_bom_viewer_tab_returns_tabspec():
    tab = AvidSecurityPlugin().bom_viewer_tab()
    assert isinstance(tab, TabSpec)
    assert tab.label == "AVID Security"
    assert "{artifact_id}" in tab.url_template


# ── register_cli() ────────────────────────────────────────────────────────


def test_register_cli_attaches_subcommands():
    plugin = AvidSecurityPlugin()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    plugin.register_cli(sub)
    assert "avid-status" in sub.choices
    assert "avid-refresh" in sub.choices
    assert "avid-scan" in sub.choices


# ── analyze() + full pipeline ─────────────────────────────────────────────


def test_analyze_returns_findings_with_matches(avid_corpus_store, seeded_cache):
    """analyze() wires snapshot→index→matcher→walker; bert-base-uncased should match."""
    plugin = AvidSecurityPlugin(cache_dir=seeded_cache)
    findings = plugin.analyze(avid_corpus_store, Scope.graph_wide())

    assert len(findings) > 0, "Expected at least one AVID finding"
    assert len(findings.matches()) > 0, "Expected at least one Match object"

    # bert-base-uncased (tier-1 exact match) must appear
    report_ids = {f.avid_report_id for f in findings}
    assert "AVID-2022-R0001" in report_ids

    # snapshot_sha should be propagated
    assert findings.snapshot_sha == "abc123deadbeef"


# ── spdx_elements() ───────────────────────────────────────────────────────


def test_spdx_elements_from_findings(avid_corpus_store, seeded_cache):
    plugin = AvidSecurityPlugin(cache_dir=seeded_cache)
    findings = plugin.analyze(avid_corpus_store, Scope.graph_wide())
    elements = plugin.spdx_elements("urn:claim:test", findings)

    assert isinstance(elements, list)
    assert len(elements) > 0
    types = {e["type"] for e in elements}
    assert "security_Vulnerability" in types


def test_spdx_elements_empty_for_empty_findings():
    plugin = AvidSecurityPlugin()
    empty = AvidFindings([], snapshot_sha="x")
    result = plugin.spdx_elements("urn:c", empty)
    assert result == []


# ── spdx_annotations() ────────────────────────────────────────────────────


def test_spdx_annotations_returns_empty_list():
    """avid_security uses spdx_elements, not Annotations."""
    plugin = AvidSecurityPlugin()
    empty = AvidFindings([], snapshot_sha="x")
    assert plugin.spdx_annotations("urn:c", empty) == []


# ── graph_overlay() ───────────────────────────────────────────────────────


def test_graph_overlay_returns_graphoverlay():
    plugin = AvidSecurityPlugin()
    findings = AvidFindings([_make_finding(1), _make_finding(2)], snapshot_sha="x")
    overlay = plugin.graph_overlay(findings)
    assert isinstance(overlay, GraphOverlay)


def test_graph_overlay_node_attrs_have_ring_colors():
    plugin = AvidSecurityPlugin()
    findings = AvidFindings([_make_finding(1), _make_finding(2)], snapshot_sha="x")
    overlay = plugin.graph_overlay(findings)
    assert len(overlay.node_attrs) > 0
    for iri, attrs in overlay.node_attrs.items():
        assert "ring_color" in attrs
        assert "badge" in attrs
        assert attrs["badge"] == "AVID"


def test_graph_overlay_worst_tier_wins():
    """Component with tier-1 and tier-2 findings should show red (tier-1 color)."""
    plugin = AvidSecurityPlugin()
    # Same IRI, tier 1 and tier 3 → worst is tier 1 → red
    f1 = AvidFinding(
        component_iri="https://example.org/X", component_label="x",
        avid_report_id="AVID-A", tier=1, confidence="high",
        matched_via="exact_bare_name", match=object(),
    )
    f3 = AvidFinding(
        component_iri="https://example.org/X", component_label="x",
        avid_report_id="AVID-B", tier=3, confidence="low",
        matched_via="family_prefix_developer", match=object(),
    )
    findings = AvidFindings([f1, f3], snapshot_sha="x")
    overlay = plugin.graph_overlay(findings)
    # tier-1 ring color is #ef4444
    assert overlay.node_attrs["https://example.org/X"]["ring_color"] == "#ef4444"


def test_graph_overlay_empty_findings():
    plugin = AvidSecurityPlugin()
    empty = AvidFindings([], snapshot_sha="x")
    overlay = plugin.graph_overlay(empty)
    assert isinstance(overlay, GraphOverlay)
    assert overlay.node_attrs == {}


# ── conflict_findings() ───────────────────────────────────────────────────


def test_conflict_findings_only_tier1():
    from aikaboom.plugins import ConflictRecord
    plugin = AvidSecurityPlugin()
    # Mix of tier 1, 2, 3 — only tier 1 should appear as ConflictRecord
    findings = AvidFindings(
        [_make_finding(1, "AVID-0001"), _make_finding(2, "AVID-0002"),
         _make_finding(3, "AVID-0003"), _make_finding(1, "AVID-0004")],
        snapshot_sha="x",
    )
    records = plugin.conflict_findings(findings)
    assert isinstance(records, list)
    assert len(records) == 2  # only tier-1 findings
    for r in records:
        assert isinstance(r, ConflictRecord)
        assert r.category == "avid-security"
        assert r.severity == "high"


def test_conflict_findings_empty_when_no_tier1():
    plugin = AvidSecurityPlugin()
    findings = AvidFindings([_make_finding(2), _make_finding(3)], snapshot_sha="x")
    assert plugin.conflict_findings(findings) == []


def test_conflict_findings_record_fields():
    from aikaboom.plugins import ConflictRecord
    plugin = AvidSecurityPlugin()
    findings = AvidFindings([_make_finding(1, "AVID-2022-R0001")], snapshot_sha="x")
    records = plugin.conflict_findings(findings)
    assert len(records) == 1
    r = records[0]
    assert "AVID-2022-R0001" in r.title
    assert r.data["avid_report_id"] == "AVID-2022-R0001"
    assert "matched_via" in r.data


# ── web_blueprint() ───────────────────────────────────────────────────────


def test_web_blueprint_returns_blueprint():
    from flask import Blueprint
    plugin = AvidSecurityPlugin()
    bp = plugin.web_blueprint()
    assert isinstance(bp, Blueprint)
    assert bp.name == "avid_security"
