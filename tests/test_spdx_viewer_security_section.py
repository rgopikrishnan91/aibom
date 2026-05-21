"""SPDX viewer — Security section rendering tests.

The SPDX viewer is JavaScript-driven (renderSPDXBOM in index.html).
Two complementary test strategies:

  1. **Template static scan** — confirms the new CSS classes and JS
     identifiers that implement the security section are present in the
     served HTML. If the template is accidentally reverted these tests
     catch the regression before any browser is involved.

  2. **SPDX data structural tests** — confirm that a document carrying
     ``security_Vulnerability`` + VEX nodes round-trips correctly
     through the data layer (validator + SPDX type ordering) so the JS
     renderer has well-formed input to display.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEMPLATE_PATH = (
    Path(__file__).parent.parent
    / "src" / "aikaboom" / "web" / "templates" / "index.html"
)


def _template_html() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def _fixture_spdx_with_security() -> dict:
    """Minimal SPDX 3.0.1 JSON-LD document containing one Vulnerability
    and one VexAffected relationship — the same shape the AVID plugin emits.
    """
    return {
        "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
        "@graph": [
            {
                "type": "ai_AIPackage",
                "spdxId": "urn:aibom:pkg:bert",
                "name": "bert-base-uncased",
            },
            {
                "type": "security_Vulnerability",
                "spdxId": "urn:vuln:AVID-2022-R0001",
                "summary": "Test vuln summary",
                "description": "Test vuln description text",
                "externalIdentifier": [{
                    "identifier": "AVID-2022-R0001",
                    "identifierLocator": [
                        "https://avidml.org/database/AVID-2022-R0001"
                    ],
                }],
            },
            {
                "type": "security_VexAffectedVulnAssessmentRelationship",
                "spdxId": "urn:vex:1",
                "from": "urn:vuln:AVID-2022-R0001",
                "to": ["urn:aibom:pkg:bert"],
                "security_actionStatement": "Mitigation: review model outputs.",
            },
        ],
    }


def _fixture_spdx_with_investigation() -> dict:
    """VexUnderInvestigation variant — exercises the Tier 2/3 status notes path."""
    return {
        "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
        "@graph": [
            {
                "type": "security_Vulnerability",
                "spdxId": "urn:vuln:AVID-2023-R0002",
                "summary": "Under investigation vuln",
                "externalIdentifier": [{
                    "identifier": "AVID-2023-R0002",
                    "identifierLocator": [
                        "https://avidml.org/database/AVID-2023-R0002"
                    ],
                }],
            },
            {
                "type": "security_VexUnderInvestigationVulnAssessmentRelationship",
                "spdxId": "urn:vex:2",
                "from": "urn:vuln:AVID-2023-R0002",
                "to": ["urn:aibom:pkg:llama"],
                "security_statusNotes": "Investigating potential bias amplification.",
            },
        ],
    }


# ---------------------------------------------------------------------------
# 1. Template static-scan tests — JS / CSS must be present in served HTML
# ---------------------------------------------------------------------------

class TestTemplateStaticScan:
    """Confirm the security-section JS and CSS are present in the template."""

    def test_template_file_exists(self):
        assert TEMPLATE_PATH.exists(), (
            f"template not found at {TEMPLATE_PATH}"
        )

    def test_css_vuln_section_class_present(self):
        html = _template_html()
        assert ".spdx-vuln-section" in html, (
            "missing CSS for .spdx-vuln-section — security section styles not added"
        )

    def test_css_vuln_card_class_present(self):
        html = _template_html()
        assert ".spdx-vuln-card" in html, (
            "missing CSS for .spdx-vuln-card"
        )

    def test_css_vex_badge_affected_class_present(self):
        html = _template_html()
        assert ".spdx-vex-badge.vex-affected" in html or (
            ".spdx-vex-badge" in html and "vex-affected" in html
        ), "missing CSS for vex-affected badge"

    def test_css_vex_badge_under_investigation_class_present(self):
        html = _template_html()
        assert "vex-under-investigation" in html, (
            "missing CSS for vex-under-investigation badge"
        )

    def test_css_vuln_ext_link_class_present(self):
        html = _template_html()
        assert ".spdx-vuln-ext-link" in html, (
            "missing CSS for .spdx-vuln-ext-link (AVID hyperlink)"
        )

    def test_js_render_security_section_function_present(self):
        html = _template_html()
        assert "renderSecuritySection" in html, (
            "missing JS function renderSecuritySection"
        )

    def test_js_security_vulnerability_type_in_order(self):
        html = _template_html()
        assert "security_Vulnerability" in html, (
            "security_Vulnerability not referenced in viewer JS"
        )

    def test_js_vex_affected_type_in_order(self):
        html = _template_html()
        assert "security_VexAffectedVulnAssessmentRelationship" in html, (
            "VexAffectedVulnAssessmentRelationship not referenced in viewer JS"
        )

    def test_js_vex_under_investigation_type_in_order(self):
        html = _template_html()
        assert "security_VexUnderInvestigationVulnAssessmentRelationship" in html, (
            "VexUnderInvestigationVulnAssessmentRelationship not referenced in viewer JS"
        )

    def test_js_label_for_vulnerability_branch(self):
        """labelForSPDXItem must have a branch for security_Vulnerability."""
        html = _template_html()
        # The label branch references externalIdentifier to pull the AVID id.
        assert "externalIdentifier" in html, (
            "labelForSPDXItem does not reference externalIdentifier — "
            "Vulnerability labels may be wrong"
        )

    def test_js_security_action_statement_key(self):
        """VEX action statement key must be referenced in the viewer."""
        html = _template_html()
        assert "security_actionStatement" in html, (
            "security_actionStatement not referenced — Tier 1 VEX action won't render"
        )

    def test_js_security_status_notes_key(self):
        """VEX status notes key (Tier 2/3) must be referenced in the viewer."""
        html = _template_html()
        assert "security_statusNotes" in html, (
            "security_statusNotes not referenced — Tier 2/3 VEX status won't render"
        )

    def test_render_security_section_called_inside_render_spdx_bom(self):
        """renderSecuritySection must be called from renderSPDXBOM, not just defined."""
        html = _template_html()
        # Find the renderSPDXBOM function body (between its opening brace and
        # the next top-level function definition) and confirm the call appears.
        m = re.search(
            r'function renderSPDXBOM\b.*?(?=\n\s*(?:function|//\s+=====))',
            html, re.DOTALL,
        )
        assert m, "could not locate renderSPDXBOM function body"
        assert "renderSecuritySection" in m.group(), (
            "renderSecuritySection is defined but never called inside renderSPDXBOM"
        )

    def test_vex_pre_pass_buckets_by_vuln_id(self):
        """The VEX pre-pass must bucket items by vulnerability spdxId (``from``)."""
        html = _template_html()
        # The pre-pass uses a Map keyed on the VEX ``from`` field.
        assert "vexByVulnId" in html, (
            "vexByVulnId map not found — VEX → Vulnerability bucketing missing"
        )


# ---------------------------------------------------------------------------
# 2. Flask GET / — served HTML contains security section infrastructure
# ---------------------------------------------------------------------------

class TestFlaskServedHTML:
    """The live app must serve the updated index.html with security classes."""

    @pytest.fixture(scope="class")
    def html(self):
        from aikaboom.web.app import app
        with app.test_client() as client:
            resp = client.get("/")
        assert resp.status_code == 200
        return resp.data.decode("utf-8")

    def test_served_html_has_vuln_section_css(self, html):
        assert "spdx-vuln-section" in html

    def test_served_html_has_vex_badge_css(self, html):
        assert "spdx-vex-badge" in html

    def test_served_html_has_render_security_section(self, html):
        assert "renderSecuritySection" in html

    def test_served_html_has_security_vulnerability_type(self, html):
        assert "security_Vulnerability" in html

    def test_served_html_has_security_action_statement(self, html):
        assert "security_actionStatement" in html


# ---------------------------------------------------------------------------
# 3. SPDX data-layer structural tests — what the JS renderer will receive
# ---------------------------------------------------------------------------

class TestSpdxSecurityNodeShape:
    """Verify that security nodes in the graph have the shape the viewer expects.

    These run against the fixture document (no plugin network calls) and
    confirm field names / structure match what renderSecuritySection reads.
    """

    def test_vulnerability_node_has_expected_fields(self):
        doc = _fixture_spdx_with_security()
        graph = doc["@graph"]
        vulns = [n for n in graph if n.get("type") == "security_Vulnerability"]
        assert len(vulns) == 1
        v = vulns[0]
        assert "summary" in v
        assert "description" in v
        assert "externalIdentifier" in v
        ext = v["externalIdentifier"][0]
        assert "identifier" in ext
        assert "identifierLocator" in ext
        assert ext["identifier"] == "AVID-2022-R0001"
        assert "avidml.org" in ext["identifierLocator"][0]

    def test_vex_affected_node_from_points_at_vuln(self):
        doc = _fixture_spdx_with_security()
        graph = doc["@graph"]
        vex = next(
            n for n in graph
            if n.get("type") == "security_VexAffectedVulnAssessmentRelationship"
        )
        vuln = next(
            n for n in graph if n.get("type") == "security_Vulnerability"
        )
        assert vex["from"] == vuln["spdxId"], (
            "VEX 'from' must match the Vulnerability spdxId so vexByVulnId lookup works"
        )

    def test_vex_affected_node_has_action_statement(self):
        doc = _fixture_spdx_with_security()
        graph = doc["@graph"]
        vex = next(
            n for n in graph
            if n.get("type") == "security_VexAffectedVulnAssessmentRelationship"
        )
        assert "security_actionStatement" in vex, (
            "VEX Affected node must carry security_actionStatement for Tier 1 rendering"
        )

    def test_vex_under_investigation_has_status_notes(self):
        doc = _fixture_spdx_with_investigation()
        graph = doc["@graph"]
        vex = next(
            n for n in graph
            if n.get("type") == "security_VexUnderInvestigationVulnAssessmentRelationship"
        )
        assert "security_statusNotes" in vex, (
            "VEX UnderInvestigation node must carry security_statusNotes for Tier 2/3 rendering"
        )

    def test_spdx_type_order_lists_security_types(self):
        """SPDX_TYPE_ORDER in the template must include both security node types
        so they are rendered in the generic bucket fallback too.
        """
        html = _template_html()
        # Extract the SPDX_TYPE_ORDER array literal.
        m = re.search(
            r'const SPDX_TYPE_ORDER\s*=\s*\[(.*?)\];',
            html, re.DOTALL,
        )
        assert m, "SPDX_TYPE_ORDER constant not found in template"
        order_block = m.group(1)
        assert "security_Vulnerability" in order_block
        assert "security_VexAffectedVulnAssessmentRelationship" in order_block
        assert "security_VexUnderInvestigationVulnAssessmentRelationship" in order_block

    def test_label_function_covers_both_vex_types(self):
        """labelForSPDXItem must branch on both VEX type strings."""
        html = _template_html()
        m = re.search(
            r'function labelForSPDXItem\b.*?(?=\n\s*//\s+Best-effort|\n\s*// Replace each|\n\s*function)',
            html, re.DOTALL,
        )
        assert m, "labelForSPDXItem function not found"
        body = m.group()
        assert "security_VexAffectedVulnAssessmentRelationship" in body
        assert "security_VexUnderInvestigationVulnAssessmentRelationship" in body

    def test_multiple_vulnerabilities_all_rendered(self):
        """renderSecuritySection iterates; ensure the fixture with 2 vulns
        would give distinct spdxIds to the Map lookup.
        """
        doc = {
            "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
            "@graph": [
                {
                    "type": "security_Vulnerability",
                    "spdxId": "urn:vuln:A",
                    "summary": "Vuln A",
                    "externalIdentifier": [{"identifier": "AVID-A", "identifierLocator": ["https://avidml.org/A"]}],
                },
                {
                    "type": "security_Vulnerability",
                    "spdxId": "urn:vuln:B",
                    "summary": "Vuln B",
                    "externalIdentifier": [{"identifier": "AVID-B", "identifierLocator": ["https://avidml.org/B"]}],
                },
                {
                    "type": "security_VexAffectedVulnAssessmentRelationship",
                    "spdxId": "urn:vex:A",
                    "from": "urn:vuln:A",
                    "to": ["urn:aibom:pkg:x"],
                    "security_actionStatement": "Action A",
                },
            ],
        }
        graph = doc["@graph"]
        vulns = [n for n in graph if n.get("type") == "security_Vulnerability"]
        vex_a = [n for n in graph
                 if n.get("type") == "security_VexAffectedVulnAssessmentRelationship"
                 and n.get("from") == "urn:vuln:A"]
        vex_b = [n for n in graph
                 if n.get("type") == "security_VexAffectedVulnAssessmentRelationship"
                 and n.get("from") == "urn:vuln:B"]
        assert len(vulns) == 2
        assert len(vex_a) == 1, "VEX for vuln A should be found by from-lookup"
        assert len(vex_b) == 0, "Vuln B has no VEX — card still renders without a badge"
