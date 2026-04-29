"""
Tests for CycloneDX 1.7 exporter.
"""
import json
import os
import tempfile
import pytest
from aikaboom.utils.cyclonedx_exporter import CycloneDXExporter, bom_to_cyclonedx


def _ai_bom_data():
    return {
        "repo_id": "microsoft/DialoGPT-medium",
        "model_id": "microsoft_DialoGPT-medium",
        "direct_fields": {
            "suppliedBy": {"value": "microsoft", "source": "hf", "conflict": None},
            "license": {"value": "MIT", "source": "hf", "conflict": {
                "value": "Apache-2.0", "source": "github", "type": "inter"
            }},
            "downloadLocation": {"value": "https://hf.co/microsoft/DialoGPT-medium", "source": "hf", "conflict": None},
            "primaryPurpose": {"value": "text-generation", "source": "hf", "conflict": None},
        },
        "rag_fields": {
            "model_name": {"value": "DialoGPT-medium", "source": "hf", "conflict": None},
            "model_type": {"value": "GPT-2 transformer", "source": "arxiv", "conflict": None},
            "limitations": {"value": "May generate biased content", "source": "hf", "conflict": None},
            "performance_metrics": {"value": "BLEU: 0.34, SSA: 0.72", "source": "arxiv", "conflict": None},
            "trainedOnDatasets": {"value": "Reddit dialogues, WebText", "source": "arxiv", "conflict": None},
            "testedOnDatasets": {"value": "DSTC-7, PersonaChat", "source": "arxiv", "conflict": None},
            "modelLineage": {"value": "GPT-2 medium", "source": "hf", "conflict": None},
        },
    }


def _dataset_bom_data():
    return {
        "dataset_id": "squad",
        "direct_metadata": {
            "name": "SQuAD 2.0",
            "license": "CC-BY-4.0",
            "originatedBy": "Stanford NLP",
        },
        "rag_metadata": {
            "intendedUse": "Question answering research",
            "datasetSize": 150000,
        },
        "urls": {"huggingface": "https://hf.co/datasets/squad"},
    }


class TestAIBOMConversion:

    def test_top_level_shape(self):
        cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(_ai_bom_data())
        assert cdx["bomFormat"] == "CycloneDX"
        assert cdx["specVersion"] == "1.7"
        assert cdx["serialNumber"].startswith("urn:uuid:")
        assert len(cdx["components"]) == 1

    def test_component_type(self):
        cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(_ai_bom_data())
        comp = cdx["components"][0]
        assert comp["type"] == "machine-learning-model"
        assert comp["name"] == "DialoGPT-medium"

    def test_license_mapped(self):
        cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(_ai_bom_data())
        licenses = cdx["components"][0].get("licenses", [])
        assert any(l.get("license", {}).get("id") == "MIT" for l in licenses)

    def test_model_card_task(self):
        cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(_ai_bom_data())
        mc = cdx["components"][0].get("modelCard", {})
        assert mc.get("modelParameters", {}).get("task") == "text-generation"

    def test_model_card_architecture(self):
        cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(_ai_bom_data())
        mc = cdx["components"][0].get("modelCard", {})
        assert "GPT-2" in mc.get("modelParameters", {}).get("architectureFamily", "")

    def test_training_datasets(self):
        cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(_ai_bom_data())
        mc = cdx["components"][0].get("modelCard", {})
        datasets = mc.get("modelParameters", {}).get("datasets", [])
        training = [d for d in datasets if d.get("type") == "training"]
        assert len(training) >= 1
        assert any("Reddit" in d["name"] for d in training)

    def test_evaluation_datasets(self):
        cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(_ai_bom_data())
        mc = cdx["components"][0].get("modelCard", {})
        datasets = mc.get("modelParameters", {}).get("datasets", [])
        evaluation = [d for d in datasets if d.get("type") == "evaluation"]
        assert len(evaluation) >= 1

    def test_pedigree_ancestors(self):
        cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(_ai_bom_data())
        comp = cdx["components"][0]
        assert "pedigree" in comp
        ancestors = comp["pedigree"]["ancestors"]
        assert any("GPT-2" in a["name"] for a in ancestors)

    def test_considerations_limitations(self):
        cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(_ai_bom_data())
        mc = cdx["components"][0].get("modelCard", {})
        lims = mc.get("considerations", {}).get("technicalLimitations", [])
        assert len(lims) >= 1
        assert "biased" in lims[0].lower()

    def test_conflict_properties(self):
        cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(_ai_bom_data())
        props = cdx["components"][0].get("properties", [])
        conflict_props = [p for p in props if p["name"].startswith("aikaboom:conflict:")]
        assert len(conflict_props) >= 1
        assert any("license" in p["name"] for p in conflict_props)


class TestDatasetBOMConversion:

    def test_dataset_component_type(self):
        cdx = CycloneDXExporter(bom_type="data").validate_and_convert(_dataset_bom_data())
        comp = cdx["components"][0]
        assert comp["type"] == "data"
        assert comp["name"] == "SQuAD 2.0"

    def test_dataset_license(self):
        cdx = CycloneDXExporter(bom_type="data").validate_and_convert(_dataset_bom_data())
        licenses = cdx["components"][0].get("licenses", [])
        assert any("CC-BY-4.0" in l.get("license", {}).get("id", "") for l in licenses)


class TestValidation:

    def test_valid_ai_bom(self):
        cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(_ai_bom_data())
        ok, errs = CycloneDXExporter().validate_cyclonedx(cdx)
        assert ok, f"Validation errors: {errs}"

    def test_wrong_format(self):
        ok, errs = CycloneDXExporter().validate_cyclonedx({"bomFormat": "SPDX"})
        assert not ok
        assert any("bomFormat" in e for e in errs)

    def test_invalid_bom_type_raises(self):
        with pytest.raises(ValueError):
            CycloneDXExporter(bom_type="invalid").validate_and_convert({})


class TestConvenienceFunction:

    def test_bom_to_cyclonedx_ai(self):
        result = bom_to_cyclonedx(_ai_bom_data(), bom_type="ai")
        assert result["bomFormat"] == "CycloneDX"

    def test_save_to_file(self):
        with tempfile.NamedTemporaryFile(suffix=".cdx.json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = bom_to_cyclonedx(_ai_bom_data(), bom_type="ai", output_path=tmp_path)
            assert os.path.exists(tmp_path)
            with open(tmp_path) as f:
                loaded = json.load(f)
            assert loaded["bomFormat"] == "CycloneDX"
        finally:
            os.unlink(tmp_path)


class TestPublicImports:

    def test_import_from_package(self):
        from aikaboom import CycloneDXExporter as CDX, bom_to_cyclonedx as b2c
        assert callable(CDX)
        assert callable(b2c)
