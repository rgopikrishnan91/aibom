"""Real-store end-to-end test: analyze() → spdx_elements().

Exercises the full pipeline (walk_components → matcher → engine → emitter)
against a BomStore populated from the shared avid_bom_corpus.ttl fixture,
which contains:

  - bert-base-uncased  (standalone Model, Hugging Face supplier)
  - dslim/bert-base-NER  (fine-tune, trainedOn bert-base-uncased)
  - squad_v2  (Dataset, no supplier)

With AVID-2022-R0001 seeded (matches bert-base-uncased exactly):
  - bert-base-uncased  → Tier 1 (VexAffected)
  - dslim/bert-base-NER → Tier 2 (VexUnderInvestigation, via base_model_lineage)

The test confirms:
  * analyze() returns findings for both components
  * spdx_elements() emits ONE shared Vulnerability node
  * Both VEX types appear (Affected + UnderInvestigation)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aikaboom.plugins import Scope
from aikaboom.plugins.avid_security.plugin import AvidSecurityPlugin

FIXTURES = Path(__file__).parent / "fixtures"
AVID_FIXTURE_SRC = (
    Path(__file__).resolve().parents[3]
    / "Golden_Set" / "AVID_Security" / "avid_fixtures" / "AVID-2022-R0001.json"
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _load_ttl_into_store(store, ttl_path: Path) -> None:
    """Parse a Turtle file and bulk-add its triples to the BomStore backend."""
    from rdflib import Graph

    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    store._backend.add_quads([(s, p, o, None) for s, p, o in g])


@pytest.fixture
def avid_corpus_store(tmp_path, monkeypatch):
    """BomStore (RDFLib backend) populated from avid_bom_corpus.ttl."""
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_path))

    from aikaboom.store import BomStore

    store = BomStore.open()
    _load_ttl_into_store(store, FIXTURES / "avid_bom_corpus.ttl")
    yield store
    store._backend.close()


@pytest.fixture
def seeded_cache(tmp_path):
    """Seed the AVID cache with AVID-2022-R0001 so analyze() never hits the network.

    Layout mirrors test_plugin_contract.py:seeded_cache so the approach is
    consistent throughout the test suite.
    """
    cache_dir = tmp_path / "avid"
    repo_dir = cache_dir / "avid-db"
    report_dir = repo_dir / "reports" / "2022"
    report_dir.mkdir(parents=True)

    # Copy the Golden Set fixture rather than constructing a synthetic one.
    report_data = json.loads(AVID_FIXTURE_SRC.read_text())
    (report_dir / "AVID-2022-R0001.json").write_text(json.dumps(report_data))

    # Write the snapshot marker so ensure_fresh() considers the cache current.
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "snapshot.json").write_text(json.dumps({
        "sha": "goldenset-sha",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "ttl_days": 10,
    }))

    # Pre-build the SQLite index from the seeded repo dir.
    from aikaboom.plugins.avid_security.snapshot import AvidIndex

    idx = AvidIndex(db_path=cache_dir / "avid.sqlite")
    idx.build(repo_dir=repo_dir)

    return cache_dir


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------

def test_analyze_finds_both_tiers(avid_corpus_store, seeded_cache):
    """Both the base model (T1) and the fine-tune (T2) should have findings."""
    plugin = AvidSecurityPlugin(cache_dir=seeded_cache)
    findings = plugin.analyze(avid_corpus_store, Scope.graph_wide())

    assert len(findings) >= 2, (
        f"Expected ≥2 findings (T1 for bert-base-uncased, T2 for fine-tune); "
        f"got {len(findings)}"
    )

    tiers = {f.tier for f in findings}
    assert 1 in tiers, "Tier 1 (Affected) finding expected for bert-base-uncased"
    assert 2 in tiers, "Tier 2 (UnderInvestigation) finding expected for dslim/bert-base-NER"


def test_spdx_elements_shared_vulnerability(avid_corpus_store, seeded_cache):
    """One Vulnerability node shared across both components (dedup by report_id)."""
    plugin = AvidSecurityPlugin(cache_dir=seeded_cache)
    findings = plugin.analyze(avid_corpus_store, Scope.graph_wide())
    nodes = plugin.spdx_elements("urn:claim:1", findings)

    vulns = [n for n in nodes if n["type"] == "security_Vulnerability"]
    assert len(vulns) == 1, (
        f"Expected exactly 1 shared Vulnerability (AVID-2022-R0001); got {len(vulns)}"
    )
    assert vulns[0]["spdxId"] == "urn:aibom:vuln:avid-2022-r0001"


def test_spdx_elements_both_vex_types_present(avid_corpus_store, seeded_cache):
    """Both VexAffected (T1) and VexUnderInvestigation (T2) must appear."""
    plugin = AvidSecurityPlugin(cache_dir=seeded_cache)
    findings = plugin.analyze(avid_corpus_store, Scope.graph_wide())
    nodes = plugin.spdx_elements("urn:claim:1", findings)

    types = {n["type"] for n in nodes if n["type"].startswith("security_Vex")}
    assert "security_VexAffectedVulnAssessmentRelationship" in types, (
        "VexAffected missing — Tier 1 (bert-base-uncased) not producing Affected VEX"
    )
    assert "security_VexUnderInvestigationVulnAssessmentRelationship" in types, (
        "VexUnderInvestigation missing — Tier 2 (dslim/bert-base-NER) not producing advisory VEX"
    )


def test_spdx_elements_distinct_vex_nodes(avid_corpus_store, seeded_cache):
    """The two VEX nodes must point to different components."""
    plugin = AvidSecurityPlugin(cache_dir=seeded_cache)
    findings = plugin.analyze(avid_corpus_store, Scope.graph_wide())
    nodes = plugin.spdx_elements("urn:claim:1", findings)

    vex_nodes = [n for n in nodes if n["type"].startswith("security_Vex")]
    # Each VEX points to a different component IRI in its `to` list.
    to_sets = [frozenset(n.get("to", [])) for n in vex_nodes]
    assert len(vex_nodes) >= 2, "Expected at least 2 VEX nodes (one per tier)"
    # The two VEX targets should differ.
    all_targets = set()
    for s in to_sets:
        all_targets |= s
    assert len(all_targets) >= 2, (
        "Expected VEX nodes to target different component IRIs; "
        "got same target for both tiers"
    )


def test_affected_vex_carries_action_statement(avid_corpus_store, seeded_cache):
    """Tier 1 VEX (Affected) must include security_actionStatement."""
    plugin = AvidSecurityPlugin(cache_dir=seeded_cache)
    findings = plugin.analyze(avid_corpus_store, Scope.graph_wide())
    nodes = plugin.spdx_elements("urn:claim:1", findings)

    affected = [
        n for n in nodes
        if n["type"] == "security_VexAffectedVulnAssessmentRelationship"
    ]
    assert len(affected) == 1
    assert "security_actionStatement" in affected[0], (
        "Tier 1 VexAffected must carry security_actionStatement"
    )


def test_under_investigation_vex_carries_status_notes(avid_corpus_store, seeded_cache):
    """Tier 2 VEX (UnderInvestigation) must include security_statusNotes."""
    plugin = AvidSecurityPlugin(cache_dir=seeded_cache)
    findings = plugin.analyze(avid_corpus_store, Scope.graph_wide())
    nodes = plugin.spdx_elements("urn:claim:1", findings)

    under_inv = [
        n for n in nodes
        if n["type"] == "security_VexUnderInvestigationVulnAssessmentRelationship"
    ]
    assert len(under_inv) == 1
    assert "security_statusNotes" in under_inv[0], (
        "Tier 2 VexUnderInvestigation must carry security_statusNotes"
    )
