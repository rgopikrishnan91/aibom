"""
Integration tests: SHACL + JSON-LD validation for the AVID/Security plugin.

test_jsonld_expands_without_unresolved_iris
    Parses a security-bearing SPDX 3.0.1 JSON-LD document using the bundled
    spdx-context.jsonld and verifies that all security IRIs survive expansion
    (i.e. no node is silently dropped or becomes a relative IRI).

test_shacl_validates_security_doc
    Runs pyshacl against the bundled spdx-model.ttl shapes using Turtle data
    (the same format and approach as tests/test_spdx_shacl_security.py) to
    confirm that a VexAffectedVulnAssessmentRelationship with actionStatement
    passes and one without it fails.  Turtle is used here rather than JSON-LD
    because the full SPDX SHACL shapes require creationInfo / name on every
    Element; Turtle lets us test only the security-specific constraints in
    isolation, which is the same pattern the project already uses.
"""

import json
from pathlib import Path

import pytest
from pyshacl import validate
from rdflib import Graph

# Repo root: tests/plugins/avid_security/test_shacl_and_jsonld.py → parents[3]
ROOT = Path(__file__).resolve().parents[3]
CTX_PATH = ROOT / "src" / "aikaboom" / "schemas" / "spdx-context.jsonld"
TTL_PATH = ROOT / "src" / "aikaboom" / "schemas" / "spdx-model.ttl"


def _security_doc():
    """Return a minimal security-bearing SPDX 3.0.1 JSON-LD document.

    Uses ``software_Package`` (not plain ``Package``) because the bundled
    context maps ``software_Package`` → the full SPDX Software/Package IRI,
    while bare ``Package`` has no context entry and becomes a local relative
    IRI.  The ``"type": "@type"`` alias (context line 802) lets us use
    ``"type"`` instead of ``"@type"``.
    """
    ctx = json.loads(CTX_PATH.read_text())["@context"]
    return {
        "@context": ctx,
        "@graph": [
            {"type": "software_Package", "spdxId": "urn:aibom:pkg:bert"},
            {
                "type": "security_Vulnerability",
                "spdxId": "urn:aibom:vuln:avid-2022-r0001",
                "summary": "Group fairness",
            },
            {
                "type": "Relationship",
                "spdxId": "urn:aibom:rel:1",
                "relationshipType": "hasAssociatedVulnerability",
                "from": "urn:aibom:pkg:bert",
                "to": ["urn:aibom:vuln:avid-2022-r0001"],
            },
            {
                "type": "security_VexAffectedVulnAssessmentRelationship",
                "spdxId": "urn:aibom:vex:1",
                "relationshipType": "affects",
                "from": "urn:aibom:vuln:avid-2022-r0001",
                "to": ["urn:aibom:pkg:bert"],
                "security_assessedElement": "urn:aibom:pkg:bert",
                "security_actionStatement": "Mitigation: review.",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Turtle fixture for SHACL tests — mirrors tests/test_spdx_shacl_security.py
# ---------------------------------------------------------------------------
_SECURITY_TTL = """\
@prefix spdx:     <https://spdx.org/rdf/3.0.1/terms/> .
@prefix security: <https://spdx.org/rdf/3.0.1/terms/Security/> .
@prefix xsd:      <http://www.w3.org/2001/XMLSchema#> .

<urn:aibom:vuln:avid-2022-r0001> a security:Vulnerability ;
    spdx:summary "Group fairness" .

<urn:aibom:pkg:bert> a spdx:Package .

<urn:aibom:rel:1> a spdx:Relationship ;
    spdx:relationshipType "hasAssociatedVulnerability" ;
    spdx:from <urn:aibom:pkg:bert> ;
    spdx:to   <urn:aibom:vuln:avid-2022-r0001> .

<urn:aibom:vex:1> a security:VexAffectedVulnAssessmentRelationship ;
    spdx:relationshipType    "affects" ;
    spdx:from                <urn:aibom:vuln:avid-2022-r0001> ;
    spdx:to                  <urn:aibom:pkg:bert> ;
    security:assessedElement <urn:aibom:pkg:bert> ;
    security:actionStatement "Mitigation: review." .
"""


# ===========================================================================
# Test 1 — JSON-LD expansion
# ===========================================================================

def test_jsonld_expands_without_unresolved_iris():
    """Security IRIs must survive JSON-LD expansion with the bundled context."""
    doc = _security_doc()
    g = Graph().parse(data=json.dumps(doc), format="json-ld")

    subjects = {str(s) for s in g.subjects()}

    # Vulnerability node must appear as a typed RDF subject.
    assert any("urn:aibom:vuln" in s for s in subjects), (
        "Vulnerability IRI did not survive JSON-LD expansion — "
        "check 'security_Vulnerability' mapping in spdx-context.jsonld"
    )

    # VEX relationship must also expand.
    vex_subjects = {s for s in subjects if "urn:aibom:vex" in s}
    assert vex_subjects, (
        "VEX relationship IRI did not survive JSON-LD expansion"
    )

    # security_Vulnerability must expand to the full SPDX Security IRI.
    type_objects = {str(o) for _, _, o in g.triples(
        (None,
         next(p for p in g.predicates()
              if "rdf-syntax-ns#type" in str(p)),
         None)
    )}
    assert any("Security/Vulnerability" in t for t in type_objects), (
        "security_Vulnerability did not expand to the Security/Vulnerability IRI"
    )

    # security_actionStatement must expand (predicate must appear in graph).
    predicates = {str(p) for p in g.predicates()}
    assert any("Security/actionStatement" in p for p in predicates), (
        "security_actionStatement predicate was not expanded — "
        "context mapping missing or broken"
    )


# ===========================================================================
# Test 2 — SHACL validation (Turtle data, same pattern as project convention)
# ===========================================================================

def test_shacl_validates_security_doc():
    """A well-formed security doc must pass the bundled SPDX SHACL shapes."""
    data = Graph().parse(data=_SECURITY_TTL, format="turtle")
    shapes = Graph().parse(location=str(TTL_PATH), format="turtle")
    conforms, _, report_text = validate(data, shacl_graph=shapes)
    assert conforms, f"SHACL validation failed:\n{report_text}"


def test_shacl_rejects_vex_without_action_statement():
    """VexAffectedVulnAssessmentRelationship without actionStatement must fail."""
    ttl_no_action = """\
@prefix spdx:     <https://spdx.org/rdf/3.0.1/terms/> .
@prefix security: <https://spdx.org/rdf/3.0.1/terms/Security/> .

<urn:aibom:vuln:x> a security:Vulnerability ;
    spdx:summary "test" .

<urn:aibom:pkg:z> a spdx:Package .

<urn:aibom:vex:y> a security:VexAffectedVulnAssessmentRelationship ;
    spdx:relationshipType "affects" ;
    spdx:from <urn:aibom:vuln:x> ;
    spdx:to   <urn:aibom:pkg:z> .
"""
    data = Graph().parse(data=ttl_no_action, format="turtle")
    shapes = Graph().parse(location=str(TTL_PATH), format="turtle")
    conforms, _, report_text = validate(data, shacl_graph=shapes)
    assert not conforms, (
        "SHACL should reject VexAffectedVulnAssessmentRelationship "
        "missing security_actionStatement"
    )
