from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from typing import Iterable

from aikaboom.plugins.avid_security.matcher import Match
from aikaboom.plugins.avid_security.engine import (
    tier_to_vex_status, build_action_statement, build_status_notes,
)

AGENT_IRI = "urn:aibom:agent:aibom-avid-plugin"
AVID_AGENT_IRI = "urn:aibom:agent:avid-ml"


def _slug(hf_path: str) -> str:
    s = hf_path.lower().replace("/", "__")
    return re.sub(r"[^a-z0-9._-]", "-", s)


def _avid_id_lower(report_id: str) -> str:
    return report_id.lower()


def _vulnerability_node(report: dict, snapshot_sha: str) -> dict:
    raw = json.loads(report["raw_json"])
    description = raw.get("description", {}).get("value", "")
    summary = raw.get("problemtype", {}).get("description", {}).get(
        "value", report["report_id"]
    )
    published_date = report.get("published_date") or "1970-01-01"
    published_iso = f"{published_date}T00:00:00Z"
    rid = report["report_id"]
    return {
        "type": "security_Vulnerability",
        "spdxId": f"urn:aibom:vuln:{_avid_id_lower(rid)}",
        "creationInfo": "_:creationinfo",
        "summary": summary,
        "description": description,
        "publishedTime": published_iso,
        "externalIdentifier": [{
            "type": "ExternalIdentifier",
            "externalIdentifierType": "securityOther",
            "identifier": rid,
            "identifierLocator": [f"https://avidml.org/database/{rid}"],
            "issuingAuthority": AVID_AGENT_IRI,
        }],
        "externalRef": [{
            "type": "ExternalRef",
            "externalRefType": "securityAdvisory",
            "locator": (
                f"https://github.com/avidml/avid-db/blob/{snapshot_sha}/"
                f"{report['source_path']}"
            ),
        }],
    }


def _has_associated_node(match: Match) -> dict:
    rid_lower = _avid_id_lower(match.avid_report["report_id"])
    slug = _slug(match.component.hf_path)
    return {
        "type": "Relationship",
        "spdxId": f"urn:aibom:rel:vuln-link-{slug}-{rid_lower}",
        "creationInfo": "_:creationinfo",
        "relationshipType": "hasAssociatedVulnerability",
        "from": match.component.spdx_id,
        "to": [f"urn:aibom:vuln:{rid_lower}"],
    }


def _vex_node(match: Match, generation_time: str) -> dict:
    rid_lower = _avid_id_lower(match.avid_report["report_id"])
    slug = _slug(match.component.hf_path)
    status = tier_to_vex_status(match.tier)
    base = {
        "spdxId": f"urn:aibom:vex:{slug}-{rid_lower}",
        "creationInfo": "_:creationinfo",
        "relationshipType": status,
        "from": f"urn:aibom:vuln:{rid_lower}",
        "to": [match.component.spdx_id],
        "security_assessedElement": match.component.spdx_id,
        "suppliedBy": [AGENT_IRI],
        "publishedTime": generation_time,
    }
    if match.tier == 1:
        base["type"] = "security_VexAffectedVulnAssessmentRelationship"
        base["security_actionStatement"] = build_action_statement(match.avid_report)
    else:
        base["type"] = "security_VexUnderInvestigationVulnAssessmentRelationship"
        base["security_statusNotes"] = build_status_notes(
            tier=match.tier,
            base_model=match.evidence.get("base_model"),
            avid_bare_name=match.avid_report["bare_name"],
            developer=match.component.developer,
        )
    return base


def emit_security_elements(
    matches: Iterable[Match], *, snapshot_sha: str,
    generation_time: str | None = None,
) -> list[dict]:
    matches = list(matches)
    if not matches:
        return []
    generation_time = generation_time or datetime.now(timezone.utc).isoformat()
    # One Vulnerability per unique report_id; many Relationship/Vex per match.
    vulns_by_id: dict[str, dict] = {}
    rels: list[dict] = []
    vexes: list[dict] = []
    for m in matches:
        rid = m.avid_report["report_id"]
        if rid not in vulns_by_id:
            vulns_by_id[rid] = _vulnerability_node(m.avid_report, snapshot_sha)
        rels.append(_has_associated_node(m))
        vexes.append(_vex_node(m, generation_time))
    return list(vulns_by_id.values()) + rels + vexes
