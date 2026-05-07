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
