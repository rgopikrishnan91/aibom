"""
Baseline regression tests for processor utility functions.
Tests pure logic helpers from processors.py — no LLM or API calls.
"""
import pytest
import math
from bom_tools.core.processors import (
    _clean_value,
    _parse_conflict_string,
    _build_triplet_payload,
    _merge_license_intra_conflict,
)


class TestCleanValue:
    """Tests for _clean_value."""

    def test_normal_string(self):
        assert _clean_value("MIT") == "MIT"

    def test_none(self):
        assert _clean_value(None) is None

    def test_nan(self):
        assert _clean_value(float('nan')) is None

    def test_number(self):
        assert _clean_value(42) == 42

    def test_empty_string(self):
        assert _clean_value("") == ""


class TestParseConflictString:
    """Tests for _parse_conflict_string."""

    def test_none_input(self):
        assert _parse_conflict_string(None) is None

    def test_dict_passthrough(self):
        d = {"value": "v", "type": "inter"}
        assert _parse_conflict_string(d) is d

    def test_no_conflict_string(self):
        assert _parse_conflict_string("No conflict detected") is None
        assert _parse_conflict_string("no cross-source issues") is None
        assert _parse_conflict_string("none") is None
        assert _parse_conflict_string("") is None

    def test_single_source_conflict(self):
        result = _parse_conflict_string("github: Apache-2.0")
        assert result is not None
        assert result["value"] == "Apache-2.0"
        assert result["source"] == "github"
        assert result["type"] == "inter"

    def test_multi_source_conflict(self):
        result = _parse_conflict_string("github: Apache-2.0, arxiv: GPL-3.0")
        assert result is not None
        assert "Apache-2.0" in result["value"]
        assert "GPL-3.0" in result["value"]
        assert "github" in result["source"]
        assert "arxiv" in result["source"]
        assert result["type"] == "inter"

    def test_plain_string_fallback(self):
        result = _parse_conflict_string("some conflict info")
        assert result is not None
        assert result["value"] == "some conflict info"
        assert result["source"] is None
        assert result["type"] == "inter"

    def test_non_string_non_dict(self):
        assert _parse_conflict_string(42) is None


class TestBuildTripletPayload:
    """Tests for _build_triplet_payload."""

    def test_basic_payload(self):
        mapping = {
            "license": "MIT",
            "license_conflict": None,
            "model_name": "TestModel",
            "model_name_conflict": "github: OtherModel",
        }
        result = _build_triplet_payload(mapping)
        assert "license" in result
        assert result["license"]["value"] == "MIT"
        assert result["license"]["conflict"] is None
        assert "model_name" in result
        assert result["model_name"]["value"] == "TestModel"
        assert result["model_name"]["conflict"] is not None

    def test_skip_keys(self):
        mapping = {"model_id": "123", "license": "MIT", "license_conflict": None}
        result = _build_triplet_payload(mapping, skip_keys={"model_id"})
        assert "model_id" not in result
        assert "license" in result

    def test_conflict_keys_excluded_from_output(self):
        mapping = {"license": "MIT", "license_conflict": "github: Apache"}
        result = _build_triplet_payload(mapping)
        assert "license_conflict" not in result
        assert "license" in result

    def test_source_suffix(self):
        mapping = {
            "license": "MIT",
            "license_conflict": None,
            "license_source": "hf",
        }
        result = _build_triplet_payload(mapping, source_suffix="_source")
        assert result["license"]["source"] == "hf"
        assert "license_source" not in result


class TestMergeLicenseIntraConflict:
    """Tests for _merge_license_intra_conflict."""

    def test_no_conflict(self):
        direct = {"license": {"value": "MIT", "source": "hf", "conflict": None}}
        result = _merge_license_intra_conflict(direct, None)
        assert result["license"]["conflict"] is None

    def test_no_has_conflict_flag(self):
        direct = {"license": {"value": "MIT", "source": "hf", "conflict": None}}
        result = _merge_license_intra_conflict(direct, {"has_conflict": False})
        assert result["license"]["conflict"] is None

    def test_with_conflict(self):
        direct = {"license": {"value": "MIT", "source": "hf", "conflict": None}}
        conflict_info = {
            "has_conflict": True,
            "conflict_description": "License mismatch",
            "per_source": {
                "github_readme": {"has_conflict": True, "conflict_description": "MIT vs Apache"},
            },
        }
        result = _merge_license_intra_conflict(direct, conflict_info)
        assert result["license"]["conflict"] is not None
        assert result["license"]["conflict"]["type"] == "intra"
        assert "License mismatch" in result["license"]["conflict"]["value"]

    def test_conflict_without_description_builds_from_per_source(self):
        direct = {"license": {"value": "MIT", "source": "hf", "conflict": None}}
        conflict_info = {
            "has_conflict": True,
            "conflict_description": None,
            "per_source": {
                "readme": {"has_conflict": True, "conflict_description": "MIT vs GPL"},
            },
        }
        result = _merge_license_intra_conflict(direct, conflict_info)
        assert result["license"]["conflict"] is not None
        assert "MIT vs GPL" in result["license"]["conflict"]["value"]
