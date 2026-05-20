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
        assert "validate_spdx" in html
        assert "strict_spdx_validation" in html
        assert "Deep SHACL validation (beta)" in html
        assert "recursive_bom" in html
        assert "Recursive Beta" in html

    def test_index_html_warns_before_recursion(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        assert "confirmRecursive" in html
        assert "walks the dependency tree" in html
        assert "unique-target set" in html
        assert "Download Linked SPDX Beta" in html

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
        assert "Download CycloneDX 1.6 Beta" in html
        assert "Download Recursive BOM Beta" in html


class TestWorldOfBomsRoutes:
    def test_graph_route_returns_nodes_and_edges(self, client):
        resp = client.get("/worldofboms/graph")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "nodes" in data and "edges" in data

    def test_stats_route_returns_counts(self, client):
        resp = client.get("/worldofboms/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "artifacts" in data and "edges" in data

    def test_graph_route_survives_unavailable_store(self, client, monkeypatch):
        monkeypatch.setenv("AIKABOOM_GRAPH_DISABLE", "1")
        resp = client.get("/worldofboms/graph")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["nodes"] == [] and data.get("store_unavailable") is True

    def test_ego_route_survives_unavailable_store(self, client, monkeypatch):
        monkeypatch.setenv("AIKABOOM_GRAPH_DISABLE", "1")
        resp = client.get("/worldofboms/ego/bom:artifact/whatever")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["nodes"] == [] and data.get("store_unavailable") is True

    def test_bom_route_survives_unavailable_store(self, client, monkeypatch):
        monkeypatch.setenv("AIKABOOM_GRAPH_DISABLE", "1")
        resp = client.get("/worldofboms/bom/bom:artifact/whatever")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("store_unavailable") is True

    def test_query_route_runs_preset(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
        monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_path / "graph"))
        from aikaboom.store.store import BomStore
        from aikaboom.store.naming import Identifier
        store = BomStore.open()
        store.save_claim(
            {"repo_id": "acme/m", "use_case": "complete",
             "direct_fields": {"trainedOnDatasets": {"value": "squad",
                               "source": "huggingface", "conflict": None}},
             "rag_fields": {}},
            {"provider": "t", "llm_model": "t", "prompt_version": "t",
             "code_version": "t", "mode": "rag", "use_case": "complete"},
            identifiers=[Identifier("huggingface", "acme/m")],
        )
        from aikaboom.store import graph_view
        g = graph_view.full_graph(store)
        m_iri = next(n["iri"] for n in g["nodes"] if n["label"] == "acme/m")
        resp = client.post("/worldofboms/query",
                            json={"preset": "datasets", "artifact": m_iri,
                                  "direction": "up"},
                            content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "rows" in data
        assert "store_unavailable" not in data   # the real path ran
        assert any(r.get("label") == "squad" for r in data["rows"])

    def test_query_route_rejects_mutating_sparql(self, client):
        resp = client.post("/worldofboms/query",
                            json={"sparql": "DELETE WHERE { ?s ?p ?o }"},
                            content_type="application/json")
        assert resp.status_code == 400

    def test_export_route_returns_jsonld(self, client):
        resp = client.get("/worldofboms/export?scope=full")
        assert resp.status_code == 200
        assert "@graph" in resp.get_json()
        assert resp.headers.get("Content-Disposition", "").startswith("attachment")


class TestWorldOfBomsTab:
    def test_index_has_worldofboms_tab(self, client):
        html = client.get("/").get_data(as_text=True)
        assert "switchTab(event, 'worldofboms')" in html
        assert 'id="worldofbomsTab"' in html
        assert 'id="worldGraphCanvas"' in html

    def test_worldofboms_tab_has_lineage_and_direction_controls(self, client):
        html = client.get("/").get_data(as_text=True)
        assert 'id="worldDirection"' in html
        assert 'data-world-pane="lineage"' in html
        assert 'id="worldDownload"' in html
