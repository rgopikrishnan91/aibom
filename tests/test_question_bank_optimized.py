"""Regression test: every question-bank entry has the optimized
``retrieval`` and ``extraction`` blocks with the right shape.

Phase 3 added these blocks to drive HyDE dense retrieval, BM25 sparse
retrieval, and the three-part LLM extraction prompt. The structural
checks below lock the shape so a future codegen run, manual edit, or
new field can't ship without them.

Re-run ``python tools/optimize_question_bank.py`` (with
``ANTHROPIC_API_KEY`` set) to refresh the optimized blocks.
"""
import json
import os

import pytest


_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
_QB_ROOT = os.path.join(_REPO_ROOT, "src", "aikaboom", "question_bank")


def _entries():
    """Yield (bom_type, field, entry_dict) for every question-bank JSON."""
    for bom_type in ("ai", "data"):
        folder = os.path.join(_QB_ROOT, bom_type)
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8") as f:
                yield bom_type, name[:-5], json.load(f)


def _est_tokens(text: str) -> float:
    """Conservative WordPiece estimate: ~3.5 chars per token. The real
    ratio for English with technical identifiers is closer to 4.5, so
    this overcounts by ~30%. We accept the overcount because we're far
    enough below the embedder's 512-token window that an exact figure
    isn't needed — the test only flags egregiously long passages.
    """
    return len(text) / 3.5


class TestRetrievalBlockShape:
    """Every entry must carry ``retrieval`` with both required keys."""

    def test_retrieval_block_present(self):
        failures = []
        for bom_type, field, entry in _entries():
            retrieval = entry.get("retrieval")
            if not isinstance(retrieval, dict):
                failures.append(f"{bom_type}/{field}: missing or non-dict 'retrieval' block")
                continue
            if "hypothetical_passage" not in retrieval:
                failures.append(f"{bom_type}/{field}: retrieval.hypothetical_passage missing")
            if "bm25_terms" not in retrieval:
                failures.append(f"{bom_type}/{field}: retrieval.bm25_terms missing")
        assert not failures, "\n".join(failures)

    def test_hypothetical_passage_non_empty_and_within_token_budget(self):
        """HyDE passage must be non-empty. Conservative-estimate cap is
        200 tokens (real BPE for English-with-technical-terms is ~30%
        lower, so 200 estimated ≈ 140 real). The embedder window is
        512 tokens; we leave 300+ tokens of headroom for the question
        and any prefix."""
        TOKEN_CAP = 200
        failures = []
        for bom_type, field, entry in _entries():
            passage = (entry.get("retrieval") or {}).get("hypothetical_passage")
            if not isinstance(passage, str) or not passage.strip():
                failures.append(f"{bom_type}/{field}: hypothetical_passage empty/non-string")
                continue
            est = _est_tokens(passage)
            if est > TOKEN_CAP:
                failures.append(
                    f"{bom_type}/{field}: hypothetical_passage ~{est:.0f} tokens (>{TOKEN_CAP} cap)"
                )
        assert not failures, "\n".join(failures)

    def test_bm25_terms_is_list_of_strings_with_min_count(self):
        failures = []
        for bom_type, field, entry in _entries():
            terms = (entry.get("retrieval") or {}).get("bm25_terms")
            if not isinstance(terms, list):
                failures.append(f"{bom_type}/{field}: bm25_terms not a list")
                continue
            if len(terms) < 5:
                failures.append(
                    f"{bom_type}/{field}: bm25_terms has only {len(terms)} entries (need ≥5)"
                )
            non_string = [t for t in terms if not isinstance(t, str) or not t.strip()]
            if non_string:
                failures.append(
                    f"{bom_type}/{field}: bm25_terms contains {len(non_string)} non-string/empty entries"
                )
        assert not failures, "\n".join(failures)


class TestExtractionBlockShape:
    """Every entry must carry ``extraction`` with all three role slots."""

    REQUIRED_KEYS = ("instruction", "field_spec", "output_guidance")

    def test_extraction_block_present(self):
        failures = []
        for bom_type, field, entry in _entries():
            extraction = entry.get("extraction")
            if not isinstance(extraction, dict):
                failures.append(f"{bom_type}/{field}: missing or non-dict 'extraction' block")
                continue
            for key in self.REQUIRED_KEYS:
                val = extraction.get(key)
                if not isinstance(val, str) or not val.strip():
                    failures.append(f"{bom_type}/{field}: extraction.{key} missing/empty")
        assert not failures, "\n".join(failures)


class TestLoaderHelpers:
    """The runtime helpers in ``aikaboom.utils.question_bank`` should
    return the optimized blocks for every entry."""

    def test_dense_query_returns_hypothetical_passage(self):
        from aikaboom.utils.question_bank import dense_query, load_question_bank
        for bom_type in ("ai", "data"):
            for field, entry in load_question_bank(bom_type).items():
                q = dense_query(entry)
                assert q and q == entry["retrieval"]["hypothetical_passage"], (
                    f"{bom_type}/{field}: dense_query did not return the HyDE passage"
                )

    def test_sparse_query_joins_bm25_terms(self):
        from aikaboom.utils.question_bank import sparse_query, load_question_bank
        for bom_type in ("ai", "data"):
            for field, entry in load_question_bank(bom_type).items():
                q = sparse_query(entry)
                expected = " ".join(entry["retrieval"]["bm25_terms"])
                assert q == expected, f"{bom_type}/{field}: sparse_query mismatch"

    def test_extraction_prompt_parts_returns_three_keys(self):
        from aikaboom.utils.question_bank import extraction_prompt_parts, load_question_bank
        for bom_type in ("ai", "data"):
            for field, entry in load_question_bank(bom_type).items():
                parts = extraction_prompt_parts(entry)
                for key in ("instruction", "field_spec", "output_guidance"):
                    assert parts.get(key), f"{bom_type}/{field}: extraction_prompt_parts.{key} empty"
