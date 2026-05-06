"""Tests for the auditor-output parser ``_parse_detector_output``.

The parser must be robust against LLM output noise (markdown emphasis,
whitespace, missing labels, malformed lines) and never raise — partial
parses fall back to safe defaults.
"""

import pytest

from aikaboom.core.conflict_routing import _parse_detector_output


GROUP_TO_SOURCE_3 = {"A": "huggingface", "B": "arxiv", "C": "github"}
GROUP_TO_SOURCE_2 = {"A": "huggingface", "B": "arxiv"}
GROUP_TO_SOURCE_1 = {"A": "huggingface"}


def test_three_groups_well_formed():
    text = """\
CLAIM_A: encoder-decoder transformer with 7B params
CLAIM_B: encoder-decoder transformer
CLAIM_C: decoder-only architecture

CONFLICT_WITHIN_A: No
CONFLICT_WITHIN_B: No
CONFLICT_WITHIN_C: No

CONFLICT_A_VS_B: No
CONFLICT_A_VS_C: Yes: A says "encoder-decoder" vs C says "decoder-only"
CONFLICT_B_VS_C: Yes: B says "encoder-decoder" vs C says "decoder-only"
"""
    claims, internal, external = _parse_detector_output(text, GROUP_TO_SOURCE_3)

    assert claims == {
        "huggingface": "encoder-decoder transformer with 7B params",
        "arxiv": "encoder-decoder transformer",
        "github": "decoder-only architecture",
    }
    assert internal == {}
    assert len(external) == 2
    assert {tuple(c["sources"]) for c in external} == {
        ("huggingface", "github"),
        ("arxiv", "github"),
    }
    assert all("A says" in c["description"] or "B says" in c["description"]
               for c in external)


def test_two_groups_well_formed():
    text = """\
CLAIM_A: Apache-2.0
CLAIM_B: MIT

CONFLICT_WITHIN_A: No
CONFLICT_WITHIN_B: No

CONFLICT_A_VS_B: Yes: A says "Apache-2.0" vs B says "MIT"
"""
    claims, internal, external = _parse_detector_output(text, GROUP_TO_SOURCE_2)
    assert claims == {"huggingface": "Apache-2.0", "arxiv": "MIT"}
    assert internal == {}
    assert len(external) == 1
    assert external[0]["sources"] == ["huggingface", "arxiv"]


def test_one_group_no_pairwise():
    text = """\
CLAIM_A: a single claim about the field

CONFLICT_WITHIN_A: No
"""
    claims, internal, external = _parse_detector_output(text, GROUP_TO_SOURCE_1)
    assert claims == {"huggingface": "a single claim about the field"}
    assert internal == {}
    assert external == []


def test_no_relevant_information_treated_as_silent():
    text = """\
CLAIM_A: encoder-decoder
CLAIM_B: No relevant information
CLAIM_C: No relevant information

CONFLICT_WITHIN_A: No
CONFLICT_WITHIN_B: No
CONFLICT_WITHIN_C: No

CONFLICT_A_VS_B: No
CONFLICT_A_VS_C: No
CONFLICT_B_VS_C: No
"""
    claims, _, _ = _parse_detector_output(text, GROUP_TO_SOURCE_3)
    assert claims["huggingface"] == "encoder-decoder"
    assert claims["arxiv"] is None
    assert claims["github"] is None


def test_markdown_emphasis_around_labels():
    text = """\
**CLAIM_A:** value-a
*CLAIM_B:* value-b

**CONFLICT_WITHIN_A:** No
**CONFLICT_WITHIN_B:** Yes: "x" vs "y"

**CONFLICT_A_VS_B:** No
"""
    claims, internal, external = _parse_detector_output(text, GROUP_TO_SOURCE_2)
    assert claims == {"huggingface": "value-a", "arxiv": "value-b"}
    assert "arxiv" in internal
    assert external == []


def test_whitespace_and_quote_stripping():
    text = '''
CLAIM_A:    "spacey claim with quotes"
CLAIM_B:   value-b.

CONFLICT_WITHIN_A: No
CONFLICT_WITHIN_B: No

CONFLICT_A_VS_B: No
'''
    claims, _, _ = _parse_detector_output(text, GROUP_TO_SOURCE_2)
    assert claims["huggingface"] == "spacey claim with quotes"
    assert claims["arxiv"] == "value-b"


def test_missing_claim_line_falls_back_to_silent():
    text = """\
CLAIM_A: only-a-said-something

CONFLICT_WITHIN_A: No
CONFLICT_WITHIN_B: No

CONFLICT_A_VS_B: No
"""
    claims, _, _ = _parse_detector_output(text, GROUP_TO_SOURCE_2)
    assert claims == {"huggingface": "only-a-said-something", "arxiv": None}


def test_malformed_conflict_line_falls_back_to_no_conflict():
    text = """\
CLAIM_A: a
CLAIM_B: b

CONFLICT_WITHIN_A: maybe?
CONFLICT_WITHIN_B: definitely no

CONFLICT_A_VS_B: indeterminate
"""
    claims, internal, external = _parse_detector_output(text, GROUP_TO_SOURCE_2)
    # Lines that don't start with "yes" → no conflict recorded
    assert claims == {"huggingface": "a", "arxiv": "b"}
    assert internal == {}
    assert external == []


def test_empty_string_returns_safe_defaults():
    claims, internal, external = _parse_detector_output("", GROUP_TO_SOURCE_3)
    assert claims == {"huggingface": None, "arxiv": None, "github": None}
    assert internal == {}
    assert external == []


def test_unknown_group_letter_in_output_is_ignored():
    """LLM hallucinates a CLAIM_D when only A/B were given — must not crash."""
    text = """\
CLAIM_A: real-a
CLAIM_B: real-b
CLAIM_D: hallucinated

CONFLICT_WITHIN_A: No
CONFLICT_WITHIN_B: No
CONFLICT_WITHIN_D: Yes

CONFLICT_A_VS_B: No
CONFLICT_A_VS_D: Yes: invented
"""
    claims, internal, external = _parse_detector_output(text, GROUP_TO_SOURCE_2)
    assert set(claims.keys()) == {"huggingface", "arxiv"}
    assert "huggingface" not in internal
    assert "arxiv" not in internal
    assert external == []


def test_internal_conflict_narrative_preserved():
    text = """\
CLAIM_A: claim-a
CLAIM_B: claim-b

CONFLICT_WITHIN_A: Yes: "first stmt" vs "second stmt"
CONFLICT_WITHIN_B: No

CONFLICT_A_VS_B: No
"""
    _, internal, _ = _parse_detector_output(text, GROUP_TO_SOURCE_2)
    assert "huggingface" in internal
    assert '"first stmt"' in internal["huggingface"]
    assert '"second stmt"' in internal["huggingface"]


def test_none_input_does_not_raise():
    claims, internal, external = _parse_detector_output(None, GROUP_TO_SOURCE_2)
    assert claims == {"huggingface": None, "arxiv": None}
    assert internal == {}
    assert external == []
