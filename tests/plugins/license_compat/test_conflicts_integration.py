"""Plugin-contributed entries appear in the Conflicts-tab feed."""
from __future__ import annotations

from aikaboom.plugins import get
from aikaboom.plugins.license_compat.engine import (
    CompatVerdict,
    Finding,
    Findings,
)


def test_license_compat_emits_conflict_records_for_violations():
    plugin = get("license-compat")
    findings = Findings([Finding(
        downstream_iri="https://example.org/Down",
        downstream_label="Down",
        upstream_iri="https://example.org/Up",
        upstream_label="Up",
        predicate="https://aikaboom.dev/aibom#trainedOn",
        downstream_license="gpl-3.0",
        upstream_licenses=frozenset({"apache-2.0"}),
        verdict=CompatVerdict(
            downstream="gpl-3.0",
            upstreams=frozenset({"apache-2.0"}),
            status="violation",
            incompatible_with=frozenset({"apache-2.0"}),
        ),
        recommendation=None,
    )])
    entries = plugin.conflict_findings(findings)
    assert len(entries) == 1
    assert entries[0].category == "license-compat"
    assert entries[0].severity == "high"
    assert entries[0].subject_iri == "https://example.org/Down"


def test_license_compat_returns_empty_when_no_violations():
    plugin = get("license-compat")
    findings = Findings([])
    assert plugin.conflict_findings(findings) == []
