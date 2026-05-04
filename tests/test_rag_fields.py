"""Tests for the RAG-pipeline post-processors and the new question-bank
entries (license, primaryPurpose, datasetAvailability, description,
sourceInfo).

These tests don't spin up the full LangGraph workflow — they exercise the
post-processor dispatch and assert that each new question entry has the
expected metadata so the wiring stays sound.
"""
import importlib
import re

import pytest


def _read_question_bank_source():
    """Pull the FIXED_QUESTIONS_AI / DATA dict bodies from the source file
    without importing :mod:`aikaboom.core.agentic_rag` (which transitively
    requires langgraph)."""
    import os
    src_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aikaboom", "core", "agentic_rag.py",
    )
    with open(src_path, encoding="utf-8") as f:
        return f.read()


def _question_priorities(block_name: str):
    src = _read_question_bank_source()
    m = re.search(rf"{block_name}\s*=\s*(\{{.*?^\}})", src, re.S | re.M)
    if not m:
        return {}
    out = {}
    for entry in re.finditer(r"^    '([\w]+)': \{(.*?)^    \}", m.group(1), re.S | re.M):
        field, body = entry.group(1), entry.group(2)
        prio = re.search(r"'priority':\s*(\[[^\]]+\])", body)
        post = re.search(r"'post_process':\s*'([^']+)'", body)
        if prio:
            import ast
            out[field] = {
                "priority": ast.literal_eval(prio.group(1)),
                "post_process": post.group(1) if post else None,
            }
    return out


class TestNewAIQuestions:
    def test_license_present_with_post_process(self):
        ai = _question_priorities("FIXED_QUESTIONS_AI")
        assert "license" in ai, "license RAG question must be defined for AI BOMs"
        assert ai["license"]["post_process"] == "normalize_license"

    def test_primary_purpose_has_no_rag_post_process(self):
        """primaryPurpose stays human-readable in the Provenance BOM; SPDX
        enum coercion happens only at export time."""
        ai = _question_priorities("FIXED_QUESTIONS_AI")
        assert "primaryPurpose" in ai
        assert ai["primaryPurpose"]["post_process"] is None

    def test_priorities_match_doc(self):
        ai = _question_priorities("FIXED_QUESTIONS_AI")
        # docs/FIELD_STRATEGIES.md says license: HF > GH > arXiv;
        # primaryPurpose: HF > arXiv > GH.
        assert ai["license"]["priority"] == ["huggingface", "github", "arxiv"]
        assert ai["primaryPurpose"]["priority"] == ["huggingface", "arxiv", "github"]


class TestNewDatasetQuestions:
    def test_required_questions_present(self):
        data = _question_priorities("FIXED_QUESTIONS_DATA")
        for field in ("license", "primaryPurpose", "datasetAvailability", "description", "sourceInfo"):
            assert field in data, f"{field} RAG question must be defined for Dataset BOMs"

    def test_post_processors_set(self):
        """Post-processors only fire when they aid conflict identification or
        readability. primaryPurpose / datasetAvailability stay human-readable;
        SPDX enum coercion happens at export time."""
        data = _question_priorities("FIXED_QUESTIONS_DATA")
        assert data["license"]["post_process"] == "normalize_license"
        assert data["primaryPurpose"]["post_process"] is None
        assert data["datasetAvailability"]["post_process"] is None
        assert data["description"]["post_process"] == "collapse_whitespace"
        assert data["sourceInfo"]["post_process"] == "dedupe_named_entities"

    def test_priorities_match_doc(self):
        data = _question_priorities("FIXED_QUESTIONS_DATA")
        # docs/FIELD_STRATEGIES.md priorities for the new dataset entries
        assert data["license"]["priority"] == ["huggingface", "github", "arxiv"]
        assert data["primaryPurpose"]["priority"] == ["huggingface", "arxiv", "github"]
        assert data["datasetAvailability"]["priority"] == ["huggingface", "github", "arxiv"]
        assert data["description"]["priority"] == ["arxiv", "huggingface", "github"]
        assert data["sourceInfo"]["priority"] == ["arxiv", "huggingface", "github"]


class TestPostProcessorRoundTrip:
    """The post-process callable named in each question's ``post_process``
    key must be resolvable via ``utils.normalise.get_post_processor``."""

    def test_every_named_post_processor_resolves(self):
        from aikaboom.utils.normalise import get_post_processor

        ai = _question_priorities("FIXED_QUESTIONS_AI")
        data = _question_priorities("FIXED_QUESTIONS_DATA")
        for field, cfg in {**ai, **data}.items():
            name = cfg.get("post_process")
            if not name:
                continue
            fn = get_post_processor(name)
            assert callable(fn), f"post_process '{name}' for {field} is not resolvable"


class TestProvenanceBOMHumanReadable:
    """Regression lock: the Provenance BOM must keep the raw human-readable
    LLM answer for fields where SPDX shape coercion happens only at export.
    """

    def test_primary_purpose_has_no_rag_post_processor(self):
        """primaryPurpose has no RAG-stage post-processor, so a string like
        'text generation system' lands in the Provenance BOM as-is."""
        ai = _question_priorities("FIXED_QUESTIONS_AI")
        data = _question_priorities("FIXED_QUESTIONS_DATA")
        assert ai["primaryPurpose"]["post_process"] is None
        assert data["primaryPurpose"]["post_process"] is None

    def test_dataset_availability_has_no_rag_post_processor(self):
        """datasetAvailability stays raw too; SPDX enum coercion happens at
        emission time inside spdx_validator._normalize_enum."""
        data = _question_priorities("FIXED_QUESTIONS_DATA")
        assert data["datasetAvailability"]["post_process"] is None


class TestPostProcessorBehaviour:
    """Quick sanity that each active canonicaliser produces the right shape."""

    def test_license_post_processor(self):
        from aikaboom.utils.normalise import normalize_license
        assert normalize_license("Apache 2.0") == "Apache-2.0"
        assert normalize_license("mit license") == "MIT"

    def test_purpose_helper_is_still_available_for_emitter(self):
        """normalize_purpose_enum is no longer a RAG post-processor but the
        SPDX emitter uses the underlying _normalize_enum helper, so the
        callable must still behave correctly when invoked directly."""
        from aikaboom.utils.normalise import normalize_purpose_enum
        assert normalize_purpose_enum("model") == "model"
        assert normalize_purpose_enum("text generation system") == "other"

    def test_availability_helper_is_still_available_for_emitter(self):
        from aikaboom.utils.normalise import normalize_availability_enum
        assert normalize_availability_enum("clickthrough") == "clickthrough"
        assert normalize_availability_enum("public download") == "directDownload"

    def test_description_post_processor(self):
        from aikaboom.utils.normalise import collapse_whitespace
        assert collapse_whitespace("  hello\n  world  ") == "hello world"

    def test_source_info_post_processor(self):
        from aikaboom.utils.normalise import dedupe_named_entities
        assert dedupe_named_entities("squad, mmlu; squad") == ["squad", "mmlu"]
