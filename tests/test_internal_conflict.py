"""
Baseline regression tests for LicenseConflictChecker.
Captures current behavior BEFORE any code changes.
"""
import pytest
from aikaboom.core.internal_conflict import LicenseConflictChecker


class TestNormalizeLicense:
    """Tests for LicenseConflictChecker.normalize_license."""

    def test_mit_variants(self):
        assert LicenseConflictChecker.normalize_license("mit") == "MIT"
        assert LicenseConflictChecker.normalize_license("MIT License") == "MIT"
        assert LicenseConflictChecker.normalize_license("the mit license") == "MIT"

    def test_apache_variants(self):
        assert LicenseConflictChecker.normalize_license("apache-2.0") == "Apache-2.0"
        assert LicenseConflictChecker.normalize_license("Apache 2.0") == "Apache-2.0"
        assert LicenseConflictChecker.normalize_license("apache license 2.0") == "Apache-2.0"

    def test_gpl_variants(self):
        assert LicenseConflictChecker.normalize_license("gpl-3.0") == "GPL-3.0"
        assert LicenseConflictChecker.normalize_license("gplv3") == "GPL-3.0"

    def test_cc_variants(self):
        assert LicenseConflictChecker.normalize_license("cc-by-4.0") == "CC-BY-4.0"
        assert LicenseConflictChecker.normalize_license("cc0") == "CC0-1.0"

    def test_empty_string(self):
        assert LicenseConflictChecker.normalize_license("") == ""

    def test_none(self):
        assert LicenseConflictChecker.normalize_license(None) == ""

    def test_unknown_license(self):
        result = LicenseConflictChecker.normalize_license("some-unknown-license")
        assert result == "some-unknown-license"

    def test_whitespace_handling(self):
        assert LicenseConflictChecker.normalize_license("  MIT  ") == "MIT"

    def test_license_suffix_stripping(self):
        # "bsd license" → strip "license" → "bsd" → lookup → "BSD-3-Clause"
        assert LicenseConflictChecker.normalize_license("bsd license") == "BSD-3-Clause"


class TestExtractLicenseFromText:
    """Tests for LicenseConflictChecker.extract_license_from_text."""

    def test_licensed_under_pattern(self):
        text = "This software is licensed under the MIT License."
        result = LicenseConflictChecker.extract_license_from_text(text)
        assert result is not None
        assert "MIT" in result

    def test_released_under_pattern(self):
        text = "This model is released under the Apache License 2.0."
        result = LicenseConflictChecker.extract_license_from_text(text)
        assert result is not None
        assert "Apache" in result

    def test_yaml_header_pattern(self):
        text = "license: MIT\ntags:\n- nlp"
        result = LicenseConflictChecker.extract_license_from_text(text)
        assert result is not None
        assert "MIT" in result

    def test_no_license_found(self):
        text = "This is a simple text with no license information."
        result = LicenseConflictChecker.extract_license_from_text(text)
        assert result is None

    def test_empty_text(self):
        assert LicenseConflictChecker.extract_license_from_text("") is None

    def test_none_text(self):
        assert LicenseConflictChecker.extract_license_from_text(None) is None


class TestComputeSimilarity:
    """Tests for LicenseConflictChecker.compute_similarity."""

    def test_identical_licenses(self):
        assert LicenseConflictChecker.compute_similarity("MIT", "MIT") == 1.0

    def test_same_after_normalization(self):
        assert LicenseConflictChecker.compute_similarity("mit", "MIT License") == 1.0

    def test_completely_different(self):
        score = LicenseConflictChecker.compute_similarity("MIT", "GPL-3.0")
        assert score < 0.5

    def test_empty_string_returns_zero(self):
        assert LicenseConflictChecker.compute_similarity("", "MIT") == 0.0
        assert LicenseConflictChecker.compute_similarity("MIT", "") == 0.0


class TestCheck:
    """Tests for LicenseConflictChecker.check."""

    def test_matching_licenses(self):
        result = LicenseConflictChecker.check("MIT", "This software is licensed under the MIT License.")
        assert result["has_conflict"] is False
        assert result["structured_license"] == "MIT"

    def test_conflicting_licenses(self):
        result = LicenseConflictChecker.check(
            "MIT",
            "This software is licensed under the Apache License 2.0.",
        )
        assert result["has_conflict"] is True
        assert result["similarity_score"] is not None
        assert result["conflict_description"] is not None

    def test_missing_structured_license(self):
        result = LicenseConflictChecker.check(None, "Licensed under MIT.")
        assert result["has_conflict"] is False
        assert "Cannot compare" in result["conflict_description"]

    def test_no_license_in_text(self):
        result = LicenseConflictChecker.check("MIT", "No license info here.")
        assert result["has_conflict"] is False
        assert "Cannot compare" in result["conflict_description"]


class TestCheckAllSources:
    """Tests for LicenseConflictChecker.check_all_sources."""

    def test_all_matching(self):
        result = LicenseConflictChecker.check_all_sources(
            "MIT",
            {"github_readme": "Licensed under the MIT License.", "hf_readme": "license: MIT"},
        )
        assert result["has_conflict"] is False
        assert len(result["per_source"]) == 2

    def test_one_conflicts(self):
        result = LicenseConflictChecker.check_all_sources(
            "MIT",
            {"github_readme": "Licensed under the MIT License.", "hf_readme": "Licensed under Apache License 2.0."},
        )
        assert result["has_conflict"] is True
        assert result["conflict_description"] is not None

    def test_empty_sources(self):
        result = LicenseConflictChecker.check_all_sources("MIT", {})
        assert result["has_conflict"] is False
        assert len(result["per_source"]) == 0
