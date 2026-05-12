"""Integration tests for Phase 12 conflict-detection rebuild.

Covers Findings #34 (per-platform direct-field schema, SAIL
packageVersion, alias-aware suppliedBy) and the supporting
SourceHandler helper. The full agentic-RAG NLI path is exercised by
``tests/test_conflict_detector.py``.
"""
import pytest

from aikaboom.core.processors import _normalize_supplier
from aikaboom.core.source_handler import SourceHandler


# ---------------------------------------------------------------------------
# get_field_per_platform: artefact-platform fields should not flag conflicts
# ---------------------------------------------------------------------------

class TestGetFieldPerPlatform:
    def test_picks_priority_winner(self):
        hf = {"downloadLocation": "https://huggingface.co/mistralai/Mistral-7B"}
        gh = {"downloadLocation": "https://github.com/mistralai/mistral-src"}
        v, src, alts = SourceHandler.get_field_per_platform(
            "downloadLocation", {"huggingface": hf, "github": gh},
            priority=["huggingface", "github"],
        )
        assert src == "huggingface"
        assert v.startswith("https://huggingface.co/")
        assert alts == {
            "huggingface": "https://huggingface.co/mistralai/Mistral-7B",
            "github": "https://github.com/mistralai/mistral-src",
        }

    def test_falls_back_when_priority_source_empty(self):
        hf = {}
        gh = {"downloadLocation": "https://github.com/foo/bar"}
        v, src, alts = SourceHandler.get_field_per_platform(
            "downloadLocation", {"huggingface": hf, "github": gh},
            priority=["huggingface", "github"],
        )
        assert src == "github"
        assert v == "https://github.com/foo/bar"
        assert alts == {"github": "https://github.com/foo/bar"}

    def test_empty_sources_return_none(self):
        v, src, alts = SourceHandler.get_field_per_platform(
            "downloadLocation", {"huggingface": {}, "github": {}},
            priority=["huggingface", "github"],
        )
        assert (v, src, alts) == (None, None, {})

    def test_never_flags_cross_platform_conflict(self):
        # Two genuinely different values — no conflict slot in the return shape.
        hf = {"releaseTime": "2024-08-15"}
        gh = {"releaseTime": "2023-01-01"}
        result = SourceHandler.get_field_per_platform(
            "releaseTime", {"huggingface": hf, "github": gh},
            priority=["huggingface", "github"],
        )
        # Tuple is (value, source, alternates); structurally cannot carry a conflict.
        assert len(result) == 3
        assert "2024-08-15" in result
        assert "2023-01-01" in result[2].values()


# ---------------------------------------------------------------------------
# _normalize_supplier (processors): alias-index canonicalisation collapses
# known cross-platform identities so they don't register as suppliedBy
# conflicts.
# ---------------------------------------------------------------------------

class TestNormalizeSupplier:
    @pytest.mark.parametrize(
        "hf_handle,gh_handle",
        [
            ("Qwen", "QwenLM"),
            ("mistralai", "mistral-ai"),
            ("google", "google-research"),
        ],
    )
    def test_known_aliases_normalise_to_same_canonical(self, hf_handle, gh_handle):
        assert _normalize_supplier(hf_handle) == _normalize_supplier(gh_handle)

    @pytest.mark.parametrize(
        "hf_handle,gh_handle",
        [
            ("openai", "anthropic"),
            ("mistralai", "google"),
            # Phase 12 design call: real org-vs-individual / org-vs-platform-org
            # provenance splits. The supplier resolver should keep them distinct
            # so the BOM surfaces them as conflicts.
            ("cais", "hendrycks"),
            ("distilbert", "huggingface"),
        ],
    )
    def test_distinct_orgs_stay_distinct(self, hf_handle, gh_handle):
        assert _normalize_supplier(hf_handle) != _normalize_supplier(gh_handle)

    def test_unknown_handle_passes_through(self):
        # An org we've never seen should still return a deterministic string,
        # not crash.
        result = _normalize_supplier("brand-new-org-2026")
        assert isinstance(result, str)
        assert result  # non-empty


# ---------------------------------------------------------------------------
# packageVersion SAIL cascade: AI BOM resolver picks SAIL name-derived
# version over HF git SHA / GH release tag whenever it can.
# ---------------------------------------------------------------------------

class TestPackageVersionCascade:
    def setup_method(self):
        # Construct a minimal AIBOMProcessor without doing any network calls.
        from aikaboom.core.processors import AIBOMProcessor

        # mode='direct' avoids spinning up the AgenticRAG vectorstore path
        self.proc = AIBOMProcessor.__new__(AIBOMProcessor)  # bypass __init__

    def test_sail_version_wins_over_git_sha(self):
        # `_resolve_direct_fields_ai` only consults `huggingface_metadata['name']`
        # plus the existing source dicts. Feed it synthetic dicts so the test
        # stays offline.
        from aikaboom.core.processors import AIBOMProcessor

        hf = {"name": "mistralai/Mistral-7B-v0.1", "packageVersion": "27d67f1b"}
        gh = {"packageVersion": "v0.1.0"}
        result = AIBOMProcessor._resolve_direct_fields_ai(
            self.proc, huggingface_metadata=hf, github_metadata=gh,
        )
        assert result["packageVersion"] == "7B-v0.1"
        assert result["packageVersion_source"] == "huggingface_name"
        assert result["packageVersion_conflicts"] is None

    def test_falls_back_to_existing_cascade_when_sail_returns_none(self):
        # google-bert/bert-base-uncased -> SAIL extracts nothing
        from aikaboom.core.processors import AIBOMProcessor

        hf = {"name": "google-bert/bert-base-uncased", "packageVersion": "1a2b3c4d"}
        gh = {"packageVersion": "v1.0"}
        result = AIBOMProcessor._resolve_direct_fields_ai(
            self.proc, huggingface_metadata=hf, github_metadata=gh,
        )
        # priority is ["huggingface_name", "huggingface", "github"]; SAIL is silent
        # so huggingface (1a2b3c4d) wins.
        assert result["packageVersion"] == "1a2b3c4d"
        assert result["packageVersion_source"] == "huggingface"
