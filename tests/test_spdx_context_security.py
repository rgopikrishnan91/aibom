import json
from pathlib import Path

CTX = Path("src/aikaboom/schemas/spdx-context.jsonld")


def test_context_contains_security_namespace():
    ctx = json.loads(CTX.read_text())["@context"]
    required = [
        "security_Vulnerability",
        "security_VexAffectedVulnAssessmentRelationship",
        "security_VexUnderInvestigationVulnAssessmentRelationship",
        "security_actionStatement",
        "security_assessedElement",
        "security_statusNotes",
    ]
    for term in required:
        assert term in ctx, f"missing JSON-LD context term: {term}"


def test_assessed_element_uses_id_type():
    """security_assessedElement must use @type: @id (not @vocab) so it resolves IRIs."""
    ctx = json.loads(CTX.read_text())["@context"]
    entry = ctx.get("security_assessedElement")
    assert isinstance(entry, dict), "security_assessedElement must be an object with @id and @type"
    assert entry.get("@type") == "@id", (
        f"security_assessedElement @type must be '@id', got: {entry.get('@type')!r}"
    )
