"""Unit tests for the retrieval chunk-quality filter.

Replaces the previous ``len < 100 chars`` cutoff that was throwing out
genuinely useful short content (one-liners like ``License: MIT``).
The new filter only drops chunks with no substantive prose.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aikaboom.utils.chunk_filter import is_useful_chunk  # noqa: E402


class TestKeptAsUseful:
    """Short-but-useful content the old filter dropped now passes."""

    def test_one_line_license(self):
        assert is_useful_chunk("License: MIT")

    def test_one_line_metric(self):
        assert is_useful_chunk("120 kWh")

    def test_short_descriptive_phrase(self):
        assert is_useful_chunk("Apache-2.0")

    def test_header_plus_prose(self):
        assert is_useful_chunk("## Bias\n\nThis dataset reflects Western bias.")

    def test_yaml_frontmatter_and_text(self):
        assert is_useful_chunk(
            "---\nlicense: apache-2.0\n---\n\n# License\n\nReleased under Apache 2.0."
        )

    def test_table_with_data(self):
        assert is_useful_chunk(
            "| Metric | Value |\n| --- | --- |\n| Accuracy | 0.94 |"
        )


class TestDroppedAsGarbage:
    """Genuine noise the filter must reject."""

    def test_empty(self):
        assert not is_useful_chunk("")

    def test_whitespace_only(self):
        assert not is_useful_chunk("   \n\n  \t  ")

    def test_below_min_floor(self):
        assert not is_useful_chunk("#")
        assert not is_useful_chunk(" - ")

    def test_header_only(self):
        assert not is_useful_chunk("## Bias")

    def test_multiple_headers_no_prose(self):
        assert not is_useful_chunk("# Title\n## Subtitle\n### Sub-subtitle")

    def test_horizontal_rule_only(self):
        assert not is_useful_chunk("---\n---")

    def test_code_fence_marker_only(self):
        assert not is_useful_chunk("```python\n```")

    def test_table_separator_only(self):
        assert not is_useful_chunk("| --- | --- |")


class TestEdgeCases:
    def test_none_input_is_safe(self):
        assert not is_useful_chunk(None)  # type: ignore[arg-type]

    def test_unicode_content(self):
        assert is_useful_chunk("Trained on 中文 corpus")

    def test_mixed_structural_and_prose(self):
        # First non-structural line wins
        text = "## Header\n\n---\n\nThis is prose.\n\n## Another header"
        assert is_useful_chunk(text)


class TestPlaceholderFilter:
    """Finding #37: HF dataset card placeholders dominate retrieval if
    not filtered. These chunks should drop out of the retrieval pool
    entirely.
    """

    def test_bare_placeholder(self):
        assert not is_useful_chunk("[More Information Needed]")

    def test_bracketed_variants(self):
        for s in [
            "[More Information Needed]",
            "[Needs More Information]",
            "[More Info Needed]",
            "[Optional]",
            "[TBD]",
            "[Coming Soon]",
            "[Placeholder]",
            "[Not Provided]",
        ]:
            assert not is_useful_chunk(s), s

    def test_unbracketed_short_markers(self):
        for s in ("TBD", "T.B.D", "N/A", "n/a", "None", "Not provided", "Not specified"):
            assert not is_useful_chunk(s), s

    def test_bulleted_placeholder_lines(self):
        text = (
            "- [More Information Needed]\n"
            "- [More Information Needed]\n"
            "* [More Information Needed]"
        )
        assert not is_useful_chunk(text)

    def test_section_header_plus_placeholder_only(self):
        """The MMLU/intendedUse failure mode: HF chunk that looked like
        ``### Personal and Sensitive Information [More Information Needed]
        ## Considerations for Using the Data`` outscored real arxiv prose
        in retrieval. After the filter, this chunk should never enter the
        retrieval pool."""
        text = (
            "### Personal and Sensitive Information\n"
            "\n"
            "[More Information Needed]\n"
            "\n"
            "## Considerations for Using the Data"
        )
        assert not is_useful_chunk(text)

    def test_placeholder_with_real_content_keeps_chunk(self):
        """A chunk that mixes a placeholder with real content stays.
        Only chunks dominated by placeholders are dropped."""
        text = (
            "### Bias\n"
            "[More Information Needed]\n"
            "\n"
            "### License\n"
            "Released under Apache-2.0."
        )
        assert is_useful_chunk(text)

    def test_numbered_list_placeholder(self):
        assert not is_useful_chunk("1. [More Information Needed]\n2. TBD")
