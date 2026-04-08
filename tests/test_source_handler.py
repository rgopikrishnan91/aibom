"""
Baseline regression tests for SourceHandler.
Captures current behavior of pure logic functions BEFORE any code changes.
"""
import pytest
from bom_tools.core.source_handler import _tag_similarity, SourceHandler


class TestTagSimilarity:
    """Tests for _tag_similarity Jaccard similarity function."""

    def test_identical_strings(self):
        assert _tag_similarity("nlp, text", "nlp, text") == 1.0

    def test_completely_different(self):
        assert _tag_similarity("alpha", "beta") == 0.0

    def test_partial_overlap(self):
        result = _tag_similarity("nlp, text, generation", "nlp, text, classification")
        # 2 shared (nlp, text), 4 total union → 0.5
        assert result == 0.5

    def test_both_empty(self):
        assert _tag_similarity("", "") == 1.0

    def test_one_empty(self):
        assert _tag_similarity("nlp", "") == 0.0

    def test_case_insensitive(self):
        assert _tag_similarity("NLP, Text", "nlp, text") == 1.0

    def test_semicolon_separator(self):
        assert _tag_similarity("nlp; text", "nlp, text") == 1.0

    def test_whitespace_separator(self):
        result = _tag_similarity("nlp text generation", "nlp text classification")
        assert result == 0.5


class TestSourceHandlerGetFieldConflict:
    """Tests for SourceHandler.get_field_conflict using explicit (name, dict) tuples."""

    def test_single_source(self):
        val, src, conflict = SourceHandler.get_field_conflict(
            "license", ("hf", {"license": "MIT"})
        )
        assert val == "MIT"
        assert src == "hf"
        assert conflict is None

    def test_no_sources_have_key(self):
        val, src, conflict = SourceHandler.get_field_conflict(
            "license", ("hf", {}), ("gh", {})
        )
        assert val is None
        assert src is None
        assert conflict is None

    def test_all_sources_agree(self):
        val, src, conflict = SourceHandler.get_field_conflict(
            "license",
            ("hf", {"license": "MIT"}),
            ("gh", {"license": "MIT"}),
        )
        assert val == "MIT"
        assert conflict is None

    def test_two_agree_one_disagrees(self):
        val, src, conflict = SourceHandler.get_field_conflict(
            "license",
            ("hf", {"license": "MIT"}),
            ("gh", {"license": "MIT"}),
            ("arxiv", {"license": "Apache-2.0"}),
        )
        assert val == "MIT"
        assert conflict is not None
        assert "Apache-2.0" in conflict

    def test_all_different_uses_priority(self):
        val, src, conflict = SourceHandler.get_field_conflict(
            "license",
            ("hf", {"license": "MIT"}),
            ("gh", {"license": "Apache-2.0"}),
            ("arxiv", {"license": "GPL-3.0"}),
        )
        # Priority = first source
        assert val == "MIT"
        assert src == "hf"
        assert "Apache-2.0" in conflict
        assert "GPL-3.0" in conflict

    def test_none_values_ignored(self):
        val, src, conflict = SourceHandler.get_field_conflict(
            "license",
            ("hf", {"license": None}),
            ("gh", {"license": "MIT"}),
        )
        assert val == "MIT"
        assert src == "gh"
        assert conflict is None

    def test_empty_string_ignored(self):
        val, src, conflict = SourceHandler.get_field_conflict(
            "license",
            ("hf", {"license": ""}),
            ("gh", {"license": "MIT"}),
        )
        assert val == "MIT"
        assert src == "gh"

    def test_fuzzy_mode_similar_tags(self):
        val, src, conflict = SourceHandler.get_field_conflict(
            "tags",
            ("hf", {"tags": "nlp, text, generation"}),
            ("gh", {"tags": "nlp, text, classification"}),
            fuzzy=True,
            fuzzy_threshold=0.4,
        )
        # Similarity=0.5 >= 0.4, so they're in the same group → no conflict
        assert val is not None
        assert conflict is None

    def test_fuzzy_mode_dissimilar_tags(self):
        val, src, conflict = SourceHandler.get_field_conflict(
            "tags",
            ("hf", {"tags": "alpha"}),
            ("gh", {"tags": "beta"}),
            fuzzy=True,
            fuzzy_threshold=0.4,
        )
        # Similarity=0.0 < 0.4, so different groups → conflict
        assert conflict is not None


class TestSourceHandlerGetField:
    """Tests for SourceHandler.get_field using explicit (name, dict) tuples."""

    def test_priority_mode_first_wins(self):
        val, src = SourceHandler.get_field(
            "license",
            ("hf", {"license": "MIT"}),
            ("gh", {"license": "Apache-2.0"}),
            mode='priority',
        )
        assert val == "MIT"
        assert src == "hf"

    def test_priority_mode_skips_none(self):
        val, src = SourceHandler.get_field(
            "license",
            ("hf", {"license": None}),
            ("gh", {"license": "MIT"}),
            mode='priority',
        )
        assert val == "MIT"
        assert src == "gh"

    def test_priority_mode_no_values(self):
        val, src = SourceHandler.get_field(
            "license",
            ("hf", {}),
            ("gh", {}),
            mode='priority',
        )
        assert val is None
        assert src is None

    def test_earliest_mode(self):
        val, src = SourceHandler.get_field(
            "releaseTime",
            ("hf", {"releaseTime": "2024-06-01"}),
            ("gh", {"releaseTime": "2024-01-15"}),
            mode='earliest',
        )
        assert val == "2024-01-15"
        assert src == "gh"

    def test_latest_mode(self):
        val, src = SourceHandler.get_field(
            "releaseTime",
            ("hf", {"releaseTime": "2024-06-01"}),
            ("gh", {"releaseTime": "2024-01-15"}),
            mode='latest',
        )
        assert val == "2024-06-01"
        assert src == "hf"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            SourceHandler.get_field(
                "x", ("hf", {"x": "val"}), mode='invalid'
            )

    def test_llm_source_renamed_to_paper(self):
        val, src = SourceHandler.get_field(
            "license",
            ("LLM_Result", {"license": "MIT"}),
            mode='priority',
        )
        assert val == "MIT"
        assert src == "paper"
