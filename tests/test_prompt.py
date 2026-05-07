"""Regression tests for the LLM prompt templates.

After Phase 5, ``prompt_generate_answer`` takes the four-arg CTF
signature ``(instruction, field_spec, output_guidance, context)`` —
``field_name`` was dropped because the answerer doesn't need it once
``field_spec`` is in scope. The companion helper
``format_chunks_for_answer`` renders a list of documents as plain
``---``-separated blocks (no source labels).

``prompt_direct_llm`` retains the five-arg signature.
``prompt_no_documents`` takes ``(field_name, instruction)``.

``prompt_detect_conflicts`` (Phase 4) is a group-anonymized auditor
prompt with signature ``(field_spec, group_chunks)`` where
``group_chunks`` is an ``OrderedDict[letter -> list[chunk_text]]``.
"""
from collections import OrderedDict

from aikaboom.core.prompt import (
    prompt_detect_conflicts,
    prompt_generate_answer,
    prompt_no_documents,
    prompt_direct_llm,
    format_chunks_for_answer,
)


class _Doc:
    """Minimal stand-in for langchain Documents."""

    def __init__(self, source, content):
        self.metadata = {"source": source}
        self.page_content = content


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
    def test_contains_ctf_section_markers(self):
        result = prompt_generate_answer(
            "Extract the model architecture.",
            "Free-form text",
            "Return the value verbatim.",
            "transformer info",
        )
        # New CTF section headers
        for marker in ("FIELD:", "TASK:", "RULES:",
                       "FIELD-SPECIFIC GUIDANCE:", "CONTEXT:", "ANSWER:"):
            assert marker in result, f"missing marker: {marker!r}"
        assert "transformer info" in result

    def test_each_slot_substituted(self):
        result = prompt_generate_answer("INSTR_X", "SPEC_Y", "GUIDE_Z", "CTX_W")
        assert "INSTR_X" in result
        assert "SPEC_Y" in result
        assert "GUIDE_Z" in result
        assert "CTX_W" in result

    def test_legacy_markers_absent(self):
        result = prompt_generate_answer("i", "fs", "og", "c")
        # Old labels and the numbered INSTRUCTIONS list are gone
        assert "FIELD NAME:" not in result
        assert "INSTRUCTION:" not in result
        assert "FIELD SPEC:" not in result
        assert "OUTPUT GUIDANCE:" not in result
        # The "Not found." sentinel was retired in favour of noAssertion
        assert "Not found" not in result
        # The new template uses noAssertion in Rule 3
        assert "noAssertion" in result

    def test_empty_guidance_falls_back_gracefully(self):
        result = prompt_generate_answer("i", "fs", "", "c")
        assert "(No additional guidance.)" in result


class TestFormatChunksForAnswer:
    def test_strips_source_labels(self):
        docs = [
            _Doc("huggingface", "model is a transformer"),
            _Doc("github", "license: apache-2.0"),
        ]
        result = format_chunks_for_answer(docs)
        # Plain --- separators only, no source attribution
        assert "(Source:" not in result
        assert "huggingface" not in result
        assert "github" not in result
        # Chunk content preserved verbatim
        assert "model is a transformer" in result
        assert "license: apache-2.0" in result
        # Separator structure
        assert result.startswith("---")
        assert result.endswith("---")

    def test_empty_chunk_list_yields_just_separators(self):
        result = format_chunks_for_answer([])
        assert result == "---"


class TestPromptNoDocuments:
    def test_contains_field_and_instruction(self):
        result = prompt_no_documents("license", "Extract the SPDX license expression.")
        assert "license" in result
        assert "Extract the SPDX license expression." in result
        # The legacy "Not found." sentinel was retired in Phase 6 in favour of
        # noAssertion to match the question-bank standardization.
        assert "Not found" not in result
        assert "noAssertion" in result


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
