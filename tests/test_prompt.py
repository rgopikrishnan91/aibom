"""Regression tests for the LLM prompt templates.

``prompt_generate_answer`` and ``prompt_direct_llm`` take the five-arg
signature ``(field_name, instruction, field_spec, output_guidance,
context)``. ``prompt_no_documents`` takes ``(field_name, instruction)``.

After Phase 4, ``prompt_detect_conflicts`` is a group-anonymized
auditor prompt with signature ``(field_spec, group_chunks)`` where
``group_chunks`` is an ``OrderedDict[letter -> list[chunk_text]]``.
"""
from collections import OrderedDict

from aikaboom.core.prompt import (
    prompt_detect_conflicts,
    prompt_generate_answer,
    prompt_no_documents,
    prompt_direct_llm,
)


def _groups(*pairs):
    """Build an OrderedDict from positional (letter, chunks) pairs."""
    return OrderedDict(pairs)


class TestPromptDetectConflicts:
    def test_three_groups_emits_full_schema(self):
        groups = _groups(
            ("A", ["hf chunk one", "hf chunk two"]),
            ("B", ["arxiv chunk"]),
            ("C", ["github chunk"]),
        )
        result = prompt_detect_conflicts("SPDX license expression", groups)

        # Field spec appears under the audit header
        assert "FIELD BEING AUDITED:" in result
        assert "SPDX license expression" in result

        # Group headers + chunk text appear verbatim
        assert "GROUP A:" in result
        assert "GROUP B:" in result
        assert "GROUP C:" in result
        assert "hf chunk one" in result
        assert "hf chunk two" in result
        assert "arxiv chunk" in result
        assert "github chunk" in result

        # All step labels present
        for marker in ("CLAIM_A:", "CLAIM_B:", "CLAIM_C:"):
            assert marker in result
        for marker in ("CONFLICT_WITHIN_A:", "CONFLICT_WITHIN_B:", "CONFLICT_WITHIN_C:"):
            assert marker in result
        for marker in ("CONFLICT_A_VS_B:", "CONFLICT_A_VS_C:", "CONFLICT_B_VS_C:"):
            assert marker in result

        # Few-shot example is present
        assert "EXAMPLE OUTPUT" in result
        assert "decoder-only transformer" in result

        # Legacy singular markers gone
        assert "INTERNAL_CONFLICT:" not in result
        assert "EXTERNAL_CONFLICT:" not in result

    def test_two_groups_skips_C_and_pairwise_with_C(self):
        groups = _groups(
            ("A", ["a chunk"]),
            ("B", ["b chunk"]),
        )
        result = prompt_detect_conflicts("FS", groups)

        # Group section excludes GROUP C
        assert "GROUP A:" in result
        assert "GROUP B:" in result
        assert "GROUP C:" not in result

        # The fixed 3-group few-shot example always mentions CLAIM_C and
        # the C-pairs, so audit-section assertions must look only at the
        # text AFTER the "NOW AUDIT THE FIELD ABOVE." sentinel.
        _, _, audit = result.partition("NOW AUDIT THE FIELD ABOVE.")
        assert "CLAIM_A:" in audit
        assert "CLAIM_B:" in audit
        assert "CLAIM_C:" not in audit
        assert "CONFLICT_A_VS_B:" in audit
        assert "CONFLICT_A_VS_C:" not in audit
        assert "CONFLICT_B_VS_C:" not in audit

    def test_one_group_has_no_pairwise_section(self):
        groups = _groups(("A", ["lone chunk"]))
        result = prompt_detect_conflicts("FS", groups)

        assert "GROUP A:" in result
        assert "lone chunk" in result
        # Audit section: STEP 3 has no real pairwise lines, only the
        # sentinel placeholder.
        _, _, audit = result.partition("NOW AUDIT THE FIELD ABOVE.")
        assert "CLAIM_A:" in audit
        assert "CONFLICT_WITHIN_A:" in audit
        assert "CONFLICT_A_VS_" not in audit
        assert "(only one group present — no pairwise comparison)" in audit

    def test_field_spec_substituted(self):
        groups = _groups(("A", ["chunk"]), ("B", ["chunk"]))
        result = prompt_detect_conflicts("UNIQUE_FIELD_SPEC_TOKEN_42", groups)
        assert "UNIQUE_FIELD_SPEC_TOKEN_42" in result


class TestPromptGenerateAnswer:
    def test_contains_field_and_context(self):
        result = prompt_generate_answer(
            "model_type",
            "Extract the model architecture.",
            "Free-form text",
            "Return the value verbatim.",
            "transformer info",
        )
        assert "model_type" in result
        assert "transformer info" in result
        assert "ANSWER:" in result

    def test_contains_three_part_extraction_slots(self):
        result = prompt_generate_answer("f", "I", "FS", "OG", "c")
        assert "INSTRUCTION: I" in result
        assert "FIELD SPEC: FS" in result
        assert "OUTPUT GUIDANCE: OG" in result

    def test_contains_not_found_fallback(self):
        result = prompt_generate_answer("f", "i", "fs", "og", "c")
        assert "Not found" in result


class TestPromptNoDocuments:
    def test_contains_field_and_instruction(self):
        result = prompt_no_documents("license", "Extract the SPDX license expression.")
        assert "license" in result
        assert "Extract the SPDX license expression." in result
        assert "Not found" in result


class TestPromptDirectLLM:
    def test_contains_all_fields(self):
        result = prompt_direct_llm(
            "license",
            "Extract the license.",
            "SPDX simplelicensing_LicenseExpression",
            "Return verbatim if found.",
            "source1\nsource2",
        )
        assert "FIELD_NAME: license" in result
        assert "INSTRUCTION: Extract the license." in result
        assert "FIELD_SPEC:" in result
        assert "OUTPUT_GUIDANCE:" in result
        assert "source1" in result
        assert "CONFLICT_STATUS" in result
        assert "CONFLICT DETECTED" in result
