"""
Baseline regression tests for SPDXValidator.
Captures current behavior BEFORE any code changes.
"""
import pytest
from aikaboom.utils.spdx_validator import SPDXValidator, validate_bom_to_spdx, validate_spdx_export


class TestExtractValue:
    """Tests for SPDXValidator._extract_value."""

    def setup_method(self):
        self.validator = SPDXValidator(bom_type='ai')

    def test_triplet_structure(self):
        assert self.validator._extract_value({"value": "MIT", "source": "hf", "conflict": None}) == "MIT"

    def test_plain_value(self):
        assert self.validator._extract_value("MIT") == "MIT"

    def test_none_value(self):
        assert self.validator._extract_value(None) is None

    def test_dict_without_value_key(self):
        result = self.validator._extract_value({"name": "test"})
        assert result == {"name": "test"}

    def test_nested_value_none(self):
        assert self.validator._extract_value({"value": None}) is None


class TestValidateAndConvert:
    """Tests for SPDXValidator.validate_and_convert."""

    def test_ai_bom_basic_structure(self):
        bom_data = {
            "repo_id": "test/model",
            "direct_fields": {"suppliedBy": "TestOrg", "license": "MIT"},
            "rag_fields": {"model_name": "TestModel"},
        }
        validator = SPDXValidator(bom_type='ai')
        result = validator.validate_and_convert(bom_data)

        assert "@context" in result
        assert "@graph" in result
        assert isinstance(result["@graph"], list)
        assert len(result["@graph"]) > 0

        # Check that AI Package exists in graph
        types = [elem.get("type") for elem in result["@graph"]]
        assert "ai_AIPackage" in types
        assert "SpdxDocument" in types
        assert "Bom" in types

    def test_dataset_bom_basic_structure(self):
        bom_data = {
            "dataset_id": "test/dataset",
            "direct_metadata": {"name": "TestDataset", "license": "Apache-2.0"},
            "rag_metadata": {"intendedUse": "Research"},
            "urls": {"huggingface": "https://huggingface.co/datasets/test/dataset"},
        }
        validator = SPDXValidator(bom_type='data')
        result = validator.validate_and_convert(bom_data)

        assert "@context" in result
        assert "@graph" in result
        types = [elem.get("type") for elem in result["@graph"]]
        assert "dataset_DatasetPackage" in types

    def test_invalid_bom_type_raises(self):
        validator = SPDXValidator(bom_type='ai')
        with pytest.raises(ValueError, match="Invalid bom_type"):
            validator.validate_and_convert({}, bom_type="invalid")

    def test_ai_bom_triplet_input(self):
        """Ensure triplet-structured fields are properly extracted."""
        bom_data = {
            "repo_id": "test/model",
            "direct_fields": {
                "suppliedBy": {"value": "DeepOrg", "source": "hf", "conflict": None},
                "license": {"value": "MIT", "source": "gh", "conflict": None},
                "downloadLocation": {"value": "https://example.com", "source": "hf", "conflict": None},
            },
            "rag_fields": {
                "model_name": {"value": "MyModel", "source": "hf", "conflict": None},
                "model_type": {"value": "transformer", "source": "arxiv", "conflict": None},
            },
        }
        validator = SPDXValidator(bom_type='ai')
        result = validator.validate_and_convert(bom_data)

        # Find the AI package
        ai_pkg = next(e for e in result["@graph"] if e["type"] == "ai_AIPackage")
        assert ai_pkg["suppliedBy"].startswith("urn:spdx:Organization-")
        assert ai_pkg["software_downloadLocation"] == "https://example.com"
        assert ai_pkg["name"] == "MyModel"
        assert ai_pkg["ai_typeOfModel"] == ["transformer"]


class TestValidateSpdxBom:
    """Tests for SPDXValidator.validate_spdx_bom."""

    def setup_method(self):
        self.validator = SPDXValidator(bom_type='ai')

    def test_valid_bom(self):
        """A structurally complete AI BOM must pass the hardened validator."""
        bom_data = {
            "repo_id": "test/model",
            "direct_fields": {"suppliedBy": "Org", "license": "MIT"},
            "rag_fields": {"model_name": "Test"},
        }
        spdx = self.validator.validate_and_convert(bom_data)
        is_valid, errors = self.validator.validate_spdx_bom(spdx)
        assert is_valid is True, f"Validator errors: {errors}"
        assert len(errors) == 0

    def test_missing_graph(self):
        spdx = {"@context": "https://spdx.org/..."}
        is_valid, errors = self.validator.validate_spdx_bom(spdx)
        assert is_valid is False
        assert any("@graph" in e for e in errors)

    def test_missing_context(self):
        spdx = {"@graph": []}
        is_valid, errors = self.validator.validate_spdx_bom(spdx)
        assert is_valid is False
        assert any("@context" in e for e in errors)

    def test_empty_dict(self):
        is_valid, errors = self.validator.validate_spdx_bom({})
        assert is_valid is False
        assert any("@context" in e for e in errors)
        assert any("@graph" in e for e in errors)

    def test_official_schema_catches_bad_field_type(self):
        spdx = self.validator.validate_and_convert({
            "repo_id": "test/model",
            "direct_fields": {"license": "MIT"},
            "rag_fields": {"model_name": "Test"},
        })
        ai_pkg = next(e for e in spdx["@graph"] if e["type"] == "ai_AIPackage")
        ai_pkg["ai_domain"] = "not-an-array"
        is_valid, errors = self.validator.validate_spdx_bom(spdx)
        assert is_valid is False
        assert any("JSON Schema" in e and "ai_domain" in e for e in errors)

    def test_structured_validation_helper(self):
        spdx = self.validator.validate_and_convert({
            "repo_id": "test/model",
            "direct_fields": {"license": "MIT"},
            "rag_fields": {"model_name": "Test"},
        })
        result = validate_spdx_export(spdx)
        assert result["valid"] is True
        assert result["validator"] == "jsonschema"
        assert result["errors"] == []

    def test_structured_validation_helper_accepts_dataset_exports(self):
        validator = SPDXValidator(bom_type='data')
        spdx = validator.validate_and_convert({
            "dataset_id": "test/dataset",
            "direct_metadata": {"license": "CC-BY-4.0"},
            "rag_metadata": {"intendedUse": "Research"},
            "urls": {"huggingface": "https://huggingface.co/datasets/test/dataset"},
        })
        result = validate_spdx_export(spdx, bom_type='data')
        assert result["valid"] is True
        assert result["errors"] == []


class TestConvenienceFunction:
    """Tests for validate_bom_to_spdx."""

    def test_ai_bom(self):
        bom_data = {
            "repo_id": "org/model",
            "direct_fields": {"license": "MIT"},
            "rag_fields": {},
        }
        result = validate_bom_to_spdx(bom_data, bom_type='ai')
        assert "@context" in result
        assert "@graph" in result

    def test_dataset_bom(self):
        bom_data = {
            "dataset_id": "org/data",
            "direct_metadata": {"license": "CC-BY-4.0"},
            "rag_metadata": {},
            "urls": {},
        }
        result = validate_bom_to_spdx(bom_data, bom_type='data')
        assert "@context" in result
