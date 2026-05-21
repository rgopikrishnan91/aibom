from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal

VexStatus = Literal["affects", "underInvestigationFor"]


def tier_to_vex_status(tier: int) -> VexStatus:
    return "affects" if tier == 1 else "underInvestigationFor"


def _first(json_list_str: str | None) -> str:
    if not json_list_str:
        return ""
    try:
        arr = json.loads(json_list_str)
        return arr[0] if arr else ""
    except (json.JSONDecodeError, IndexError):
        return ""


def build_action_statement(report: dict) -> str:
    sep = _first(report.get("sep_view"))
    risk = _first(report.get("risk_domain"))
    return (
        f"Mitigation: review AVID report {report['report_id']} ({sep}). "
        f"No upstream fix recorded; apply {risk}-category guardrails "
        f"appropriate to the deployment context."
    )


def build_status_notes(*, tier: int, base_model: str | None,
                        avid_bare_name: str, developer: str | None) -> str:
    if tier == 2:
        return (
            f"Inherited from base model `{base_model}`; "
            f"downstream fine-tune may preserve or mask the issue. "
            f"Re-evaluate against the AVID metric."
        )
    if tier == 3:
        return (
            f"Same family as AVID artifact `{avid_bare_name}` "
            f"(developer `{developer}`); could impact this component — "
            f"manual review needed to confirm applicability."
        )
    return ""


@dataclass(frozen=True)
class AvidFinding:
    """One AVID match against a BOM component."""
    component_iri: str
    component_label: str
    avid_report_id: str
    tier: int            # 1, 2, or 3
    confidence: str      # high / medium / low
    matched_via: str
    match: Any           # the matcher.Match object, carried for SPDX emission

    @property
    def is_affected(self) -> bool:
        return self.tier == 1


class AvidFindings:
    """Findings wrapper conforming to the plugin Findings protocol
    (to_dict + violations), plus helpers for SPDX emission."""

    def __init__(self, items: Iterable[AvidFinding], snapshot_sha: str = "unknown"):
        self._items: list[AvidFinding] = list(items)
        self.snapshot_sha = snapshot_sha

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def violations(self) -> list[AvidFinding]:
        # Tier-1 (Affected) findings are the actionable ones surfaced in the
        # Conflicts tab; tier 2/3 are advisory (UnderInvestigation).
        return [f for f in self._items if f.is_affected]

    def matches(self) -> list:
        """The underlying matcher.Match objects, for the SPDX emitter."""
        return [f.match for f in self._items]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_sha": self.snapshot_sha,
            "findings": [
                {
                    "component_iri": f.component_iri,
                    "component": f.component_label,
                    "avid_report_id": f.avid_report_id,
                    "tier": f.tier,
                    "confidence": f.confidence,
                    "matched_via": f.matched_via,
                }
                for f in self._items
            ],
        }
