import json
import pytest
from aikaboom.plugins.avid_security.spdx import emit_security_elements
from aikaboom.plugins.avid_security.matcher import Match
from aikaboom.plugins.avid_security.walker import Component


def _match(tier, hf_path="bert-base-uncased", report_id="AVID-2022-R0001",
           base_models=()):
    component = Component(
        kind="Model", hf_path=hf_path,
        developer="Hugging Face", base_models=base_models,
        scope_in_bom="principal",
        spdx_id=f"urn:aibom:pkg:{hf_path.replace('/', '__').lower()}",
    )
    confidence = {1: "high", 2: "medium", 3: "low"}[tier]
    report = {
        "report_id": report_id,
        "bare_name": hf_path.split("/")[-1].lower(),
        "sep_view": '["E0101: Group fairness"]',
        "risk_domain": '["Ethics"]',
        "published_date": "2022-11-09",
        "source_path": "reports/2022/AVID-2022-R0001.json",
        "raw_json": json.dumps({
            "description": {"value": "sample description"},
            "problemtype": {"description": {"value": "Gender bias"}},
        }),
    }
    evidence = {"matched_via": "exact_bare_name"} if tier == 1 else (
        {"matched_via": "base_model_lineage", "base_model": "bert-base-uncased"}
        if tier == 2 else
        {"matched_via": "family_prefix_developer",
         "family_prefix": "gemma-3n", "developer": "Hugging Face"}
    )
    return Match(component=component, avid_report=report,
                 tier=tier, confidence=confidence, evidence=evidence)


def test_tier1_emits_vulnerability_link_and_vexaffected():
    nodes = emit_security_elements([_match(tier=1)], snapshot_sha="3f2a91c")
    types = {n["type"] for n in nodes}
    assert "security_Vulnerability" in types
    assert "Relationship" in types
    assert "security_VexAffectedVulnAssessmentRelationship" in types

    vex = next(n for n in nodes
               if n["type"] == "security_VexAffectedVulnAssessmentRelationship")
    assert vex["relationshipType"] == "affects"
    assert "security_actionStatement" in vex
    assert "AVID-2022-R0001" in vex["security_actionStatement"]
    assert "E0101: Group fairness" in vex["security_actionStatement"]


def test_tier2_emits_underinvestigation_with_statusnotes():
    nodes = emit_security_elements(
        [_match(tier=2, hf_path="dslim/bert-base-NER",
                base_models=("bert-base-uncased",))],
        snapshot_sha="3f2a91c",
    )
    vex = next(n for n in nodes
               if n["type"] == "security_VexUnderInvestigationVulnAssessmentRelationship")
    assert vex["relationshipType"] == "underInvestigationFor"
    assert "Inherited from base model `bert-base-uncased`" in vex["security_statusNotes"]
    assert "security_actionStatement" not in vex


def test_shared_vulnerability_across_components():
    # Same AVID report matches two different components — expect ONE Vulnerability,
    # TWO hasAssociatedVulnerability Relationships, TWO Vex relationships.
    m1 = _match(tier=1, hf_path="bert-base-uncased")
    m2 = _match(tier=2, hf_path="dslim/bert-base-NER", base_models=("bert-base-uncased",))
    nodes = emit_security_elements([m1, m2], snapshot_sha="3f2a91c")
    vulns = [n for n in nodes if n["type"] == "security_Vulnerability"]
    assert len(vulns) == 1
    rels = [n for n in nodes if n["type"] == "Relationship"]
    assert len(rels) == 2
    vexes = [n for n in nodes if n["type"].startswith("security_Vex")]
    assert len(vexes) == 2
