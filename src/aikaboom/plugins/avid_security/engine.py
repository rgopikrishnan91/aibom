from __future__ import annotations
import json
from typing import Literal

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
