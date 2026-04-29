"""
Integration tests — verify components work together correctly.
Tests the SPDX pipeline, web app processing flow, CLI argument handling,
and cross-module data flow. No real LLM calls (uses mocks where needed).
"""
import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock


class TestSPDXPipeline:
    """Test the full flow: BOM data → SPDX conversion → validation."""

    def test_ai_bom_to_spdx_roundtrip(self):
        """Complete AI BOM → SPDX conversion produces valid SPDX."""
        from aikaboom.utils.spdx_validator import SPDXValidator

        bom_data = {
            "repo_id": "test/model",
            "direct_fields": {
                "suppliedBy": {"value": "TestOrg", "source": "hf", "conflict": None},
                "license": {"value": "MIT", "source": "hf", "conflict": {
                    "value": "Apache-2.0", "source": "github", "type": "inter"
                }},
                "downloadLocation": {"value": "https://example.com", "source": "hf", "conflict": None},
                "releaseTime": {"value": "2024-01-01", "source": "hf", "conflict": None},
            },
            "rag_fields": {
                "model_name": {"value": "TestModel", "source": "arxiv", "conflict": None},
                "domain": {"value": "NLP", "source": "hf", "conflict": None},
                "model_type": {"value": "transformer", "source": "arxiv", "conflict": None},
            },
        }

        validator = SPDXValidator(bom_type="ai")
        spdx = validator.validate_and_convert(bom_data)

        # Validate structure
        is_valid, errors = validator.validate_spdx_bom(spdx)
        assert is_valid, f"SPDX validation failed: {errors}"

        # Check AI-specific elements exist
        types = [e["type"] for e in spdx["@graph"]]
        assert "ai_AIPackage" in types
        assert "SpdxDocument" in types
        assert "Bom" in types
        assert "simplelicensing_LicenseExpression" in types

        # Check license was mapped
        license_elem = next(e for e in spdx["@graph"] if e["type"] == "simplelicensing_LicenseExpression")
        assert license_elem["simplelicensing_licenseExpression"] == "MIT"

        # Check AI package has correct fields
        ai_pkg = next(e for e in spdx["@graph"] if e["type"] == "ai_AIPackage")
        assert ai_pkg["name"] == "TestModel"
        assert ai_pkg["ai_domain"] == "NLP"
        assert ai_pkg["ai_typeOfModel"] == "transformer"
        assert ai_pkg["suppliedBy"] == "TestOrg"

    def test_dataset_bom_to_spdx_roundtrip(self):
        """Complete Dataset BOM → SPDX conversion produces valid SPDX."""
        from aikaboom.utils.spdx_validator import SPDXValidator

        bom_data = {
            "dataset_id": "test/dataset",
            "direct_metadata": {
                "name": "TestDataset",
                "license": "Apache-2.0",
                "originatedBy": "TestLab",
            },
            "rag_metadata": {
                "intendedUse": "Research",
                "datasetSize": 50000,
                "datasetType": ["text", "qa"],
                "knownBias": "Some demographic bias in training data",
            },
            "urls": {"huggingface": "https://huggingface.co/datasets/test/dataset"},
        }

        validator = SPDXValidator(bom_type="data")
        spdx = validator.validate_and_convert(bom_data)

        is_valid, errors = validator.validate_spdx_bom(spdx)
        assert is_valid, f"SPDX validation failed: {errors}"

        types = [e["type"] for e in spdx["@graph"]]
        assert "dataset_DatasetPackage" in types

        ds_pkg = next(e for e in spdx["@graph"] if e["type"] == "dataset_DatasetPackage")
        assert ds_pkg["name"] == "TestDataset"
        assert ds_pkg["dataset_datasetSize"] == 50000
        assert ds_pkg["dataset_intendedUse"] == "Research"

    def test_spdx_save_to_file(self):
        """SPDX output can be saved and re-read as valid JSON."""
        from aikaboom.utils.spdx_validator import validate_bom_to_spdx

        bom_data = {
            "repo_id": "org/model",
            "direct_fields": {"license": "MIT"},
            "rag_fields": {"model_name": "Test"},
        }

        with tempfile.NamedTemporaryFile(suffix=".spdx.json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            spdx = validate_bom_to_spdx(bom_data, bom_type="ai", output_path=tmp_path)
            assert os.path.exists(tmp_path)

            with open(tmp_path) as f:
                loaded = json.load(f)
            assert loaded["@context"] == spdx["@context"]
            assert len(loaded["@graph"]) == len(spdx["@graph"])
        finally:
            os.unlink(tmp_path)


class TestWebAppIntegration:
    """Integration tests for web app endpoints with valid payloads."""

    @pytest.fixture
    def client(self):
        from aikaboom.web.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_process_valid_json_no_urls(self, client):
        """POST /process with valid JSON but no URLs returns 400."""
        resp = client.post("/process",
            json={"bom_type": "ai", "mode": "rag", "llm_provider": "openai"},
            content_type="application/json")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert "at least" in data["message"].lower() or "provide" in data["message"].lower()

    def test_process_data_bom_no_urls(self, client):
        """POST /process for data BOM with no URLs returns 400."""
        resp = client.post("/process",
            json={"bom_type": "data", "mode": "rag", "llm_provider": "openai"},
            content_type="application/json")
        assert resp.status_code == 400

    def test_find_links_without_gemini_key(self, client):
        """POST /find_links without GEMINI_API_KEY should return skipped."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            resp = client.post("/find_links",
                json={"bom_type": "ai", "repo_id": "test/model"},
                content_type="application/json")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["status"] == "skipped"

    def test_process_format_parameter_accepted(self, client):
        """POST /process with format=spdx should be accepted (not crash on format param)."""
        resp = client.post("/process",
            json={"bom_type": "ai", "mode": "rag", "format": "spdx"},
            content_type="application/json")
        # Will fail with 400 (no URLs) or 500 (no API key), not a crash from format param
        assert resp.status_code in (400, 500)

    def test_download_valid_file(self, client):
        """Write a file to results dir and verify download works."""
        from aikaboom.web.app import app
        test_content = {"test": True}
        test_file = os.path.join(app.config["UPLOAD_FOLDER"], "test_download.json")
        with open(test_file, "w") as f:
            json.dump(test_content, f)
        try:
            resp = client.get("/download/test_download.json")
            assert resp.status_code == 200
            assert json.loads(resp.data) == test_content
        finally:
            os.unlink(test_file)


class TestCLIIntegration:
    """Test CLI argument parsing and error handling (no real LLM calls)."""

    def test_generate_ai_missing_urls(self):
        """CLI generate --type ai with no URLs should exit with error."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "aikaboom.cli", "generate", "--type", "ai"],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode != 0
        assert "provide" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_generate_data_missing_urls(self):
        """CLI generate --type data with no URLs should exit with error."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "aikaboom.cli", "generate", "--type", "data"],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode != 0


class TestConflictDetectionIntegration:
    """Test the full conflict detection pipeline across modules."""

    def test_inter_source_conflict_flows_through_triplet(self):
        """SourceHandler conflict → _parse_conflict_string → triplet output."""
        from aikaboom.core.source_handler import SourceHandler
        from aikaboom.core.processors import _parse_conflict_string

        val, src, conflict = SourceHandler.get_field_conflict(
            "license",
            ("hf", {"license": "MIT"}),
            ("gh", {"license": "Apache-2.0"}),
        )
        assert val == "MIT"
        assert conflict is not None

        # Parse the conflict string into structured format
        parsed = _parse_conflict_string(conflict)
        assert parsed is not None
        assert parsed["type"] == "inter"
        assert "Apache-2.0" in parsed["value"]

    def test_intra_source_conflict_flows_through_merge(self):
        """LicenseConflictChecker → _merge_license_intra_conflict → triplet output."""
        from aikaboom.core.internal_conflict import LicenseConflictChecker
        from aikaboom.core.processors import _merge_license_intra_conflict

        conflict_result = LicenseConflictChecker.check_all_sources(
            structured_license="MIT",
            readme_texts={"github_readme": "This software is released under the Apache License 2.0."},
        )
        assert conflict_result["has_conflict"] is True

        direct_fields = {
            "license": {"value": "MIT", "source": "hf", "conflict": None}
        }
        merged = _merge_license_intra_conflict(direct_fields, conflict_result)
        assert merged["license"]["conflict"] is not None
        assert merged["license"]["conflict"]["type"] == "intra"

    def test_triplet_payload_preserves_conflict_chain(self):
        """_build_triplet_payload correctly wires conflict through."""
        from aikaboom.core.processors import _build_triplet_payload

        raw = {
            "license": "MIT",
            "license_conflict": "github: Apache-2.0",
            "license_source": "huggingface",
            "domain": "NLP",
            "domain_conflict": None,
            "domain_source": "arxiv",
        }
        result = _build_triplet_payload(raw, conflict_suffix="_conflict", source_suffix="_source")

        assert result["license"]["value"] == "MIT"
        assert result["license"]["source"] == "huggingface"
        assert result["license"]["conflict"] is not None
        assert result["license"]["conflict"]["type"] == "inter"

        assert result["domain"]["value"] == "NLP"
        assert result["domain"]["conflict"] is None


class TestSPDXFieldMapping:
    """Verify that BOM fields correctly map to SPDX elements."""

    def test_ai_rag_fields_map_to_spdx(self):
        """RAG field names correctly map to SPDX AI Package properties."""
        from aikaboom.utils.spdx_validator import SPDXValidator

        bom = {
            "repo_id": "t/m",
            "direct_fields": {},
            "rag_fields": {
                "model_name": {"value": "TestModel", "source": "hf", "conflict": None},
                "domain": {"value": "CV", "source": "arxiv", "conflict": None},
                "intended_use": {"value": "classification", "source": "hf", "conflict": None},
                "model_type": {"value": "CNN", "source": "arxiv", "conflict": None},
                "limitations": {"value": "bias in data", "source": "arxiv", "conflict": None},
            },
        }

        spdx = SPDXValidator(bom_type="ai").validate_and_convert(bom)
        ai_pkg = next(e for e in spdx["@graph"] if e["type"] == "ai_AIPackage")

        assert ai_pkg["name"] == "TestModel"
        assert ai_pkg["ai_domain"] == "CV"
        assert ai_pkg["ai_informationAboutApplication"] == "classification"
        assert ai_pkg["ai_typeOfModel"] == "CNN"
        assert ai_pkg["ai_limitation"] == "bias in data"
