"""
Baseline regression tests for prompt templates.
Captures current behavior BEFORE any code changes.
"""
from bom_tools.core.prompt import (
    prompt_detect_conflicts,
    prompt_generate_answer,
    prompt_no_documents,
    prompt_direct_llm,
)


class TestPromptDetectConflicts:
    def test_contains_field_name(self):
        result = prompt_detect_conflicts("license", "What is the license?", "The license type", "chunk1\nchunk2")
        assert "license" in result
        assert "FIELD NAME: license" in result

    def test_contains_instructions(self):
        result = prompt_detect_conflicts("domain", "Q", "D", "C")
        assert "INTERNAL_CONFLICT" in result
        assert "EXTERNAL_CONFLICT" in result

    def test_contains_context(self):
        result = prompt_detect_conflicts("f", "q", "d", "my_context_data_here")
        assert "my_context_data_here" in result


class TestPromptGenerateAnswer:
    def test_contains_field_and_context(self):
        result = prompt_generate_answer("model_type", "What type?", "The model type", "transformer info")
        assert "model_type" in result
        assert "transformer info" in result
        assert "ANSWER:" in result

    def test_contains_instructions(self):
        result = prompt_generate_answer("f", "q", "d", "c")
        assert "Not found" in result


class TestPromptNoDocuments:
    def test_contains_question(self):
        result = prompt_no_documents("What is the license?")
        assert "What is the license?" in result
        assert "Not found" in result


class TestPromptDirectLLM:
    def test_contains_all_fields(self):
        result = prompt_direct_llm("license", "What license?", "License type", "source1\nsource2")
        assert "FIELD_NAME: license" in result
        assert "QUESTION_SUMMARY: What license?" in result
        assert "source1" in result
        assert "CONFLICT_STATUS" in result
        assert "CONFLICT DETECTED" in result
