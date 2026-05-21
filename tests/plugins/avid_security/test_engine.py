from aikaboom.plugins.avid_security.engine import (
    tier_to_vex_status, build_action_statement, build_status_notes,
)


def test_tier_to_vex_status():
    assert tier_to_vex_status(1) == "affects"
    assert tier_to_vex_status(2) == "underInvestigationFor"
    assert tier_to_vex_status(3) == "underInvestigationFor"


def test_build_action_statement_uses_sep_view_and_risk_domain():
    report = {
        "report_id": "AVID-2022-R0001",
        "sep_view": '["E0101: Group fairness"]',
        "risk_domain": '["Ethics"]',
    }
    s = build_action_statement(report)
    assert "AVID-2022-R0001" in s
    assert "E0101: Group fairness" in s
    assert "Ethics" in s


def test_build_status_notes_tier2():
    s = build_status_notes(tier=2, base_model="bert-base-uncased", avid_bare_name="bert-base-uncased", developer=None)
    assert "Inherited from base model `bert-base-uncased`" in s


def test_build_status_notes_tier3():
    s = build_status_notes(tier=3, base_model=None, avid_bare_name="gemma-3n-E4B-it", developer="Google")
    assert "Same family as AVID artifact `gemma-3n-E4B-it`" in s
    assert "developer `Google`" in s
    assert "manual review needed" in s
