"""Tests for AvidFinding and AvidFindings (pure, no store)."""
from __future__ import annotations

import pytest
from aikaboom.plugins.avid_security.engine import AvidFinding, AvidFindings
from aikaboom.plugins.avid_security.walker import Component


def _component(hf_path: str = "bert-base-uncased") -> Component:
    return Component(
        kind="Model",
        hf_path=hf_path,
        developer="Hugging Face",
        base_models=(),
        scope_in_bom="principal",
        spdx_id=f"https://example.org/{hf_path}",
    )


def _finding(tier: int, report_id: str = "AVID-2022-R0001") -> AvidFinding:
    confidence = {1: "high", 2: "medium", 3: "low"}[tier]
    return AvidFinding(
        component_iri=f"https://example.org/bert-base-uncased",
        component_label="bert-base-uncased",
        avid_report_id=report_id,
        tier=tier,
        confidence=confidence,
        matched_via="exact_bare_name" if tier == 1 else "base_model_lineage",
        match=object(),  # opaque; we just carry it
    )


class TestAvidFinding:
    def test_is_affected_tier1(self):
        f = _finding(tier=1)
        assert f.is_affected is True

    def test_is_affected_tier2(self):
        f = _finding(tier=2)
        assert f.is_affected is False

    def test_is_affected_tier3(self):
        f = _finding(tier=3)
        assert f.is_affected is False

    def test_fields_accessible(self):
        f = _finding(tier=1, report_id="AVID-2022-R0001")
        assert f.avid_report_id == "AVID-2022-R0001"
        assert f.tier == 1
        assert f.confidence == "high"
        assert f.matched_via == "exact_bare_name"


class TestAvidFindings:
    def _make(self, tiers: list[int]) -> AvidFindings:
        items = [_finding(t, report_id=f"AVID-{i:04d}") for i, t in enumerate(tiers, 1)]
        return AvidFindings(items, snapshot_sha="abc123")

    def test_len(self):
        findings = self._make([1, 2, 3])
        assert len(findings) == 3

    def test_iter_yields_all(self):
        findings = self._make([1, 2, 3])
        assert len(list(findings)) == 3

    def test_violations_returns_only_tier1(self):
        findings = self._make([1, 2, 3, 1])
        v = findings.violations()
        assert all(f.tier == 1 for f in v)
        assert len(v) == 2

    def test_violations_empty_when_no_tier1(self):
        findings = self._make([2, 3])
        assert findings.violations() == []

    def test_matches_returns_match_objects(self):
        findings = self._make([1, 2])
        m = findings.matches()
        assert len(m) == 2  # one per finding

    def test_to_dict_structure(self):
        findings = self._make([1, 2])
        d = findings.to_dict()
        assert d["snapshot_sha"] == "abc123"
        assert len(d["findings"]) == 2
        first = d["findings"][0]
        assert "component_iri" in first
        assert "component" in first
        assert "avid_report_id" in first
        assert "tier" in first
        assert "confidence" in first
        assert "matched_via" in first

    def test_to_dict_tier_values(self):
        findings = self._make([1, 3])
        d = findings.to_dict()
        tiers = {f["tier"] for f in d["findings"]}
        assert tiers == {1, 3}

    def test_snapshot_sha_attribute(self):
        findings = AvidFindings([], snapshot_sha="deadbeef")
        assert findings.snapshot_sha == "deadbeef"

    def test_empty_findings(self):
        findings = AvidFindings([])
        assert len(findings) == 0
        assert findings.violations() == []
        assert findings.matches() == []
        assert findings.to_dict()["findings"] == []
