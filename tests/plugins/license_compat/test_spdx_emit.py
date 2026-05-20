"""SPDX Annotation Element emitter tests."""
from __future__ import annotations

import json

from aikaboom.plugins.license_compat.engine import (
    CompatVerdict,
    Finding,
    Findings,
)
from aikaboom.plugins.license_compat.spdx import emit_annotations


def _violation_finding() -> Finding:
    return Finding(
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
    )


def test_emit_one_annotation_per_violation(tiny_matrix):
    findings = Findings([_violation_finding()])
    out = emit_annotations("https://example.org/Claim", findings, matrix=tiny_matrix)
    assert len(out) == 1
    a = out[0]
    assert a["type"] == "Annotation"
    assert a["annotationType"] == "review"
    assert a["subject"] == "https://example.org/Down"
    body = json.loads(a["comment"])
    assert body["plugin"] == "license-compat"
    assert body["verdict"] == "violation"
    assert body["upstream"] == "https://example.org/Up"


def test_emit_includes_breaking_node_annotation(tiny_matrix):
    findings = Findings([_violation_finding(), _violation_finding()])
    out = emit_annotations("https://example.org/Claim", findings, matrix=tiny_matrix)
    breaking_anns = [a for a in out if json.loads(a["comment"]).get("kind") == "breaking-node"]
    assert len(breaking_anns) >= 1


def test_emit_empty_findings_returns_empty_list(tiny_matrix):
    out = emit_annotations("https://example.org/Claim", Findings([]), matrix=tiny_matrix)
    assert out == []
