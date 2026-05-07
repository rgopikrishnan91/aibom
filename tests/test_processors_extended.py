"""
Baseline regression tests for processor utility functions.
Tests pure logic helpers from processors.py — no LLM or API calls.
"""
import pytest
import math
from aikaboom.core.processors import (
    _clean_value,
    _parse_conflict_string,
    _build_triplet_payload,
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

    def test_trace_block_attached_when_claims_present(self):
        """Phase 4 — when the wide row carries the per-source trace keys,
        the triplet gets a ``trace`` block alongside value/source/conflict."""
        mapping = {
            "license": "Apache-2.0",
            "license_conflict": None,
            "license_source": "huggingface, arxiv",
            "license_claims": {
                "huggingface": "Apache-2.0",
                "arxiv": "Apache-2.0",
                "github": "MIT",
            },
            "license_internal_conflicts": {},
            "license_external_conflicts": [
                {"sources": ["huggingface", "github"],
                 "description": "huggingface says Apache-2.0 vs github says MIT"},
                {"sources": ["arxiv", "github"],
                 "description": "arxiv says Apache-2.0 vs github says MIT"},
            ],
            "license_selected_sources": ["huggingface", "arxiv"],
        }
        result = _build_triplet_payload(mapping, source_suffix="_source")
        # Legacy slots untouched
        assert result["license"]["value"] == "Apache-2.0"
        assert result["license"]["source"] == "huggingface, arxiv"
        assert result["license"]["conflict"] is None
        # Trace block present and well-shaped
        trace = result["license"]["trace"]
        assert trace["claims"] == {
            "huggingface": "Apache-2.0",
            "arxiv": "Apache-2.0",
            "github": "MIT",
        }
        assert trace["selected_sources"] == ["huggingface", "arxiv"]
        assert trace["internal_conflicts"] == {}
        assert len(trace["external_conflicts"]) == 2
        assert trace["external_conflicts"][0]["sources"] == ["huggingface", "github"]
        # Trace propagation keys themselves are not promoted to triplets
        for k in ("license_claims", "license_internal_conflicts",
                  "license_external_conflicts", "license_selected_sources"):
            assert k not in result

    def test_no_trace_block_when_claims_absent(self):
        """Direct-API fields with no Phase-4 trace data have no trace key."""
        mapping = {
            "license": "MIT",
            "license_conflict": None,
        }
        result = _build_triplet_payload(mapping)
        assert "trace" not in result["license"]
        assert set(result["license"].keys()) == {"value", "source", "conflict"}


# NOTE: TestMergeLicenseIntraConflict was removed because the
# _merge_license_intra_conflict function it referenced does not exist
# in processors.py — the test was collection-failing before this
# refactor touched anything. If/when that helper is added, restore
# the test class.
