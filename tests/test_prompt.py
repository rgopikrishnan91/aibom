"""Regression tests for the LLM prompt templates.

After Phase 3, each prompt takes a five-arg signature:
``(field_name, instruction, field_spec, output_guidance, context)`` —
the three middle args are the per-field ``extraction.*`` triplet from
the question bank. ``prompt_no_documents`` takes ``(field_name, instruction)``.
"""
from aikaboom.core.prompt import (
    prompt_detect_conflicts,
    prompt_generate_answer,
    prompt_no_documents,
    prompt_direct_llm,
)


class TestPromptDetectConflicts:
    def test_contains_field_name(self):
        result = prompt_detect_conflicts(
            "license",
            "Extract the license expression.",
            "SPDX simplelicensing_LicenseExpression",
            "Use noAssertion if missing.",
            "chunk1\nchunk2",
        )
        assert "license" in result
        assert "FIELD NAME: license" in result

    def test_contains_three_part_extraction_slots(self):
        result = prompt_detect_conflicts("domain", "I", "FS", "OG", "C")
        assert "INSTRUCTION: I" in result
        assert "FIELD SPEC: FS" in result
        assert "OUTPUT GUIDANCE: OG" in result

    def test_contains_conflict_markers(self):
        result = prompt_detect_conflicts("domain", "I", "FS", "OG", "C")
        assert "INTERNAL_CONFLICT" in result
        assert "EXTERNAL_CONFLICT" in result

    def test_contains_context(self):
        result = prompt_detect_conflicts("f", "i", "fs", "og", "my_context_data_here")
        assert "my_context_data_here" in result


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
