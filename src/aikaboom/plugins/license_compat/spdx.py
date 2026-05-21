"""SPDX 3.0.1 Annotation Element emitter for license-compat findings.

Reuses the conflict-annotation pattern: one Annotation Element per finding,
annotationType="review", structured JSON in `comment`. Other SPDX tools
can ignore the body; aibom round-trips it.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from aikaboom.plugins.license_compat.engine import (
    Findings,
    find_breaking_nodes,
)
from aikaboom.plugins.license_compat.matrix import LicenseMatrix

_TOOL = "Tool:aikaboom-license-compat/0.1"


def _ann_id(*parts: str) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"spdx:annotation/license-compat/{h}"


def emit_annotations(claim_iri: str, findings: Findings, matrix: LicenseMatrix) -> list[dict]:
    out: list[dict] = []
    for f in findings.violations():
        body = {
            "plugin": "license-compat",
            "kind": "violation",
            "verdict": f.verdict.status,
            "predicate": f.predicate,
            "upstream": f.upstream_iri,
            "downstream_license": f.downstream_license,
            "upstream_licenses": sorted(f.upstream_licenses),
            "incompatible_with": sorted(f.verdict.incompatible_with),
        }
        if f.recommendation is not None:
            body["recommendation"] = {
                "by_category": f.recommendation.by_category,
                "is_solvable": f.recommendation.is_solvable,
            }
        out.append({
            "type": "Annotation",
            "spdxId": _ann_id("violation", f.downstream_iri, f.upstream_iri, f.predicate),
            "annotationType": "review",
            "subject": f.downstream_iri,
            "creationInfo": {"createdBy": [_TOOL]},
            "statement": (
                f"License {f.downstream_license} incompatible with upstream "
                f"{sorted(f.verdict.incompatible_with)} via "
                f"{f.predicate.rsplit('#', 1)[-1]}"
            ),
            "contentType": "application/json",
            "comment": json.dumps(body),
        })

    # One annotation per breaking node, attached to the upstream artifact.
    # Only emit when the node blocks more than one downstream — a singleton
    # violation is already covered by the per-violation annotation above and
    # would duplicate the same information.
    breaking = find_breaking_nodes(findings, matrix, Counter())
    for n in breaking:
        if n.blamed_in < 2:
            continue
        body = {
            "plugin": "license-compat",
            "kind": "breaking-node",
            "blamed_in": n.blamed_in,
            "affected_downstream": sorted(n.affected_downstream),
            "license": n.license,
            "fix_recommendations": {
                "by_category": n.fix_recommendations.by_category,
                "is_solvable": n.fix_recommendations.is_solvable,
            },
        }
        out.append({
            "type": "Annotation",
            "spdxId": _ann_id("breaking", n.artifact_iri),
            "annotationType": "review",
            "subject": n.artifact_iri,
            "creationInfo": {"createdBy": [_TOOL]},
            "statement": (
                f"Breaking node: {n.label} ({n.license}) blocks {n.blamed_in} "
                f"downstream artifact(s)"
            ),
            "contentType": "application/json",
            "comment": json.dumps(body),
        })

    return out
