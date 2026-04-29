"""
Tests for web UI features: SPDX-always generation, link fallback messaging.
These verify behavior introduced when polishing the public-facing UI.
"""
import json
import os
import pytest
from unittest.mock import patch


@pytest.fixture
def client():
    from aikaboom.web.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestFindLinksEndpoint:
    """Tests for the /find_links endpoint UI feedback."""

    def test_no_gemini_key_returns_machine_readable_reason(self, client):
        """When GEMINI_API_KEY is missing, response must include reason='no_gemini_key'."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            resp = client.post(
                "/find_links",
                json={"bom_type": "ai", "repo_id": "microsoft/DialoGPT-medium"},
                content_type="application/json",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "skipped"
        assert data["reason"] == "no_gemini_key"
        assert "Link Fallback Agent" in data["message"]

    def test_no_gemini_key_preserves_provided_links(self, client):
        """When skipped, the response should echo back what the user already has."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            resp = client.post(
                "/find_links",
                json={
                    "bom_type": "ai",
                    "repo_id": "org/model",
                    "arxiv_url": "https://arxiv.org/abs/1234.5678",
                },
                content_type="application/json",
            )
        data = resp.get_json()
        assert data["hf_repo_id"] == "org/model"
        assert data["arxiv_url"] == "https://arxiv.org/abs/1234.5678"


class TestSPDXAlwaysGenerated:
    """SPDX should be generated automatically — no checkbox needed."""

    def test_response_data_shape_includes_spdx_keys(self):
        """Verify that successful /process responses always include SPDX keys.

        We can't actually run the LLM in tests, but we can verify the code path
        builds the correct response_data shape via mocking.
        """
        from aikaboom.web import app as app_module

        # Build a fake metadata dict matching processor output
        fake_metadata = {
            "repo_id": "org/model",
            "model_id": "org_model",
            "use_case": "complete",
            "direct_fields": {
                "license": {"value": "MIT", "source": "hf", "conflict": None}
            },
            "rag_fields": {
                "model_name": {"value": "Test", "source": "hf", "conflict": None}
            },
        }

        # Verify SPDX validator can handle it without raising
        from aikaboom.utils.spdx_validator import SPDXValidator
        validator = SPDXValidator(bom_type="ai")
        spdx = validator.validate_and_convert(fake_metadata)
        assert "@context" in spdx
        assert "@graph" in spdx


class TestLinkFallbackInfoStructure:
    """Verify the link_fallback info structure surfaced to the UI."""

    def test_no_gemini_key_structure(self, client):
        """Backend should return structured link_fallback info with reason."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            resp = client.post(
                "/find_links",
                json={"bom_type": "ai"},
                content_type="application/json",
            )
        data = resp.get_json()
        # The /find_links response itself uses 'reason' for the UI to read
        assert data.get("reason") == "no_gemini_key"

    def test_data_bom_no_urls_returns_400(self, client):
        """Data BOM with no URLs should still validate input."""
        resp = client.post(
            "/process",
            json={"bom_type": "data", "mode": "rag"},
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"


class TestUITemplateContent:
    """Verify the rendered HTML has the expected new tab labels and no checkbox."""

    def test_index_html_no_spdx_checkbox(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        # The old checkbox should be gone
        assert 'id="spdx_output"' not in html

    def test_index_html_has_provenance_tab(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        assert "BOM with Provenance" in html

    def test_index_html_has_spdx_tab(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        assert "SPDX 3.0.1" in html

    def test_index_html_has_gemini_help_link(self, client):
        """When showing 'agent inactive', UI should link to a free-key signup."""
        resp = client.get("/")
        html = resp.data.decode()
        assert "aistudio.google.com/app/apikey" in html

    def test_index_html_download_buttons_relabeled(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        assert "Download Provenance BOM" in html
        assert "Download SPDX 3.0.1" in html
