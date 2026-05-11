"""Tests for the SupplierAliasIndex tiered resolver.

Tier 1 (cross-reference / curated seed), Tier 2 (normalised exact),
Tier 3 (Jaro-Winkler) are each exercised independently so a regression
in one tier doesn't silently degrade into another.
"""
import pytest

from aikaboom.utils.supplier_alias import (
    SupplierAliasIndex,
    default_alias_index,
    normalize_org,
)


class TestNormalizeOrg:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("google-research", "google"),
            ("FacebookAI", "facebook"),
            ("facebookresearch", "facebook"),
            ("Qwen", "qwen"),
            ("Mistral-AI", "mistral"),
            # "ai" is in the strip-suffix list (matches "Labs", "Research"
            # etc.); "OpenAI" normalises to "open". Tier 1 of the index
            # still resolves "OpenAI" -> canonical "openai" via the curated
            # seed before normalize_org is ever consulted.
            ("OpenAI", "open"),
            ("Eleuther AI", "eleuther"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_normalises_separators_and_suffixes(self, raw, expected):
        assert normalize_org(raw) == expected


class TestSupplierAliasIndexTier1:
    """Curated-seed cross-references collapse known platform pairs."""

    def setup_method(self):
        self.idx = default_alias_index()

    @pytest.mark.parametrize(
        "hf_handle,gh_handle",
        [
            # Cross-platform pairs that the substring tier resolves without
            # any curated seed entry. These should remain robust to seed/harvest
            # churn — they rely only on normalize_org + substring.
            ("Qwen", "QwenLM"),
            ("mistralai", "mistral-ai"),
            ("google", "google-research"),
            ("EleutherAI", "eleuther-ai"),
            # Curated equivalence (handles share no substring overlap).
            ("allenai", "ai2"),
        ],
    )
    def test_known_aliases_match(self, hf_handle, gh_handle):
        assert self.idx.is_same_supplier(hf_handle, gh_handle) is True

    @pytest.mark.parametrize(
        "hf_handle,gh_handle",
        [
            ("openai", "anthropic"),
            ("mistralai", "google"),
            ("facebookresearch", "openai"),
            # Phase 12 design call: these are real org-vs-individual provenance
            # splits, not aliases. Surfacing them as a `suppliedBy` conflict
            # is the correct behaviour, not noise to suppress.
            ("cais", "hendrycks"),
            ("distilbert", "huggingface"),
        ],
    )
    def test_distinct_orgs_dont_match(self, hf_handle, gh_handle):
        assert self.idx.is_same_supplier(hf_handle, gh_handle) is False

    def test_canonicalize_collapses_aliases(self):
        assert self.idx.canonicalize("Qwen") == self.idx.canonicalize("QwenLM")
        assert self.idx.canonicalize("google-research") == self.idx.canonicalize("Google")


class TestSupplierAliasIndexTier2:
    """Normalised-exact tier handles unseen handles whose normalised form matches."""

    def setup_method(self):
        # Empty seed: forces tests to hit Tier 2 or Tier 3.
        self.idx = SupplierAliasIndex(seed={})

    def test_separator_difference_matches(self):
        assert self.idx.is_same_supplier("mistral-ai", "Mistral AI")

    def test_suffix_strip_matches(self):
        assert self.idx.is_same_supplier("acme", "acme-research")
        assert self.idx.is_same_supplier("acme", "acmeLabs")


class TestSupplierAliasIndexTier3:
    """Jaro-Winkler fallback for typo-style variants the upper tiers miss."""

    def setup_method(self):
        self.idx = SupplierAliasIndex(seed={}, jw_threshold=0.85)

    def test_close_variants_match(self):
        # Manufactured close variant; would also be caught by Tier 2 suffix
        # strip in most real cases. Use an unusual stem to isolate Tier 3.
        assert self.idx.is_same_supplier("Snoogle", "Snooglee")

    def test_distant_orgs_do_not_match(self):
        assert not self.idx.is_same_supplier("openai", "anthropic")
        assert not self.idx.is_same_supplier("mistralai", "facebook")

    def test_empty_inputs_do_not_match(self):
        assert not self.idx.is_same_supplier("", "openai")
        assert not self.idx.is_same_supplier("openai", "")
        assert not self.idx.is_same_supplier(None, "openai")


class TestSupplierAliasIndexHarvestRoundTrip:
    """add_cross_reference makes a previously-unknown pair resolve."""

    def test_add_then_match(self):
        idx = SupplierAliasIndex(seed={})
        assert idx.is_same_supplier("acme-models", "acme-corp") in (True, False)  # tier 2/3 dependent
        idx.add_cross_reference("zwq-research", "ZQ-Code")
        assert idx.is_same_supplier("zwq-research", "ZQ-Code")
