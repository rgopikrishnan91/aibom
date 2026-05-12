"""Phase 12 — table-aware chunking + confidence-scored conflict grounding.

12A — markdown tables stay together in chunks (no off-by-one row→value
      misalignment when the LLM sees them).
12B — every auditor-detected conflict carries a cosine-similarity
      grounding score against the chunks it's supposed to come from.
12C — score + high/low breakdown surface in BOM JSON, web UI conflicts
      tab, HF Space (same template), and CLI end-of-run summary.
"""
from __future__ import annotations

from langchain_core.embeddings import FakeEmbeddings
import pytest


# ---------------------------------------------------------------------------
# 12A — table-aware chunking
# ---------------------------------------------------------------------------


def test_small_markdown_table_stays_in_one_chunk():
    """A markdown table that fits inside one chunk must not be split
    between header row and data rows."""
    from aikaboom.core.agentic_rag import HeaderAwareTextSplitter

    splitter = HeaderAwareTextSplitter(chunk_size=500, min_chunk_size=10)
    md = (
        "## Benchmarks\n\n"
        "Some prose before the table.\n\n"
        "| Benchmark | Score |\n"
        "|---|---|\n"
        "| MMLU | 60.1 |\n"
        "| HellaSwag | 81.3 |\n"
        "| Winogrande | 75.3 |\n\n"
        "Some prose after."
    )
    chunks = splitter.split_text(md)
    table_chunks = [c for c in chunks if "MMLU" in c]
    assert len(table_chunks) == 1
    chunk = table_chunks[0]
    # All data rows live with the header.
    assert "| Benchmark | Score |" in chunk
    assert "| MMLU | 60.1 |" in chunk
    assert "| HellaSwag | 81.3 |" in chunk
    assert "| Winogrande | 75.3 |" in chunk


def test_oversized_table_emits_per_row_chunks_with_header():
    """A table larger than chunk_size * 1.5 emits one chunk per data
    row, each carrying the column-header row."""
    from aikaboom.core.agentic_rag import HeaderAwareTextSplitter

    splitter = HeaderAwareTextSplitter(chunk_size=200, min_chunk_size=10)
    big = (
        "## Big\n\n"
        "| Bench | Score |\n"
        "|---|---|\n"
        + "\n".join(f"| Bench{i:03d} | {i * 1.1:.1f} |" for i in range(40))
    )
    chunks = splitter.split_text(big)
    # Should be at least one chunk per data row.
    assert len(chunks) >= 40
    sample = next(c for c in chunks if "Bench017" in c)
    # Header row travels with each data-row chunk.
    assert "| Bench | Score |" in sample
    assert "| Bench017 |" in sample


def test_plain_text_unaffected_by_table_logic():
    """Header-aware behaviour for plain text is unchanged."""
    from aikaboom.core.agentic_rag import HeaderAwareTextSplitter

    splitter = HeaderAwareTextSplitter(chunk_size=300, min_chunk_size=10)
    md = "## Section\n\n" + ("word " * 200)
    chunks = splitter.split_text(md)
    assert len(chunks) >= 2  # Splits on size as before.
    # No false-positive table detection.
    assert all("|---|" not in c for c in chunks)


def test_single_pipe_line_not_treated_as_table():
    """A stray ``|`` line without a header-separator is NOT a table."""
    from aikaboom.core.agentic_rag import HeaderAwareTextSplitter

    splitter = HeaderAwareTextSplitter(chunk_size=300, min_chunk_size=10)
    md = "## Section\n\n| this looks like a table line\n\nbut it isn't."
    chunks = splitter.split_text(md)
    # The pipe line should still appear, in some chunk.
    assert any("looks like a table line" in c for c in chunks)


# ---------------------------------------------------------------------------
# 12B — confidence-scored conflict grounding
# ---------------------------------------------------------------------------


def test_extract_conflict_statements_internal():
    from aikaboom.core.conflict_routing import extract_conflict_statements

    a, b = extract_conflict_statements('Yes: "the model has 7B params" vs "the model has 13B params"', "internal")
    assert a == "the model has 7B params"
    assert b == "the model has 13B params"


def test_extract_conflict_statements_external():
    from aikaboom.core.conflict_routing import extract_conflict_statements

    narrative = 'A says "decoder-only" vs B says "encoder-decoder"'
    a, b = extract_conflict_statements(narrative, "external")
    assert a == "decoder-only"
    assert b == "encoder-decoder"


def test_extract_conflict_statements_falls_back_when_unmatched():
    from aikaboom.core.conflict_routing import extract_conflict_statements

    # No quoted statements → empty fallback (treated as low grounding).
    assert extract_conflict_statements("Yes", "internal") == ("", "")
    assert extract_conflict_statements("", "internal") == ("", "")
    assert extract_conflict_statements(None, "internal") == ("", "")


def test_score_grounding_high_when_statement_is_quotable():
    """A statement that's verbatim in chunks should score higher than an
    unrelated statement.

    ``FakeEmbeddings`` from langchain is content-agnostic — every distinct
    string gets a random vector regardless of content — so it can't tell
    "verbatim quote" from "unrelated text" by similarity. We use a tiny
    deterministic stub that maps strings to one-hot vectors over a fixed
    vocabulary so a verbatim quote literally maps to the same vector as
    its source chunk.
    """
    from aikaboom.core.agentic_rag import AgenticRAG

    class _DeterministicEmbeddings:
        """Bag-of-words one-hot embeddings; identical strings → identical vectors."""
        def __init__(self):
            self._vocab: dict = {}

        def _vec(self, text: str):
            import numpy as np
            tokens = text.lower().split()
            for t in tokens:
                self._vocab.setdefault(t, len(self._vocab))
            v = np.zeros(max(64, len(self._vocab) * 2))
            for t in tokens:
                v[self._vocab[t]] += 1.0
            norm = np.linalg.norm(v)
            return (v / norm if norm else v).tolist()

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        def embed_query(self, text):
            return self._vec(text)

    rag = AgenticRAG.__new__(AgenticRAG)
    rag.embeddings = _DeterministicEmbeddings()
    chunks = [
        "The model has 7B parameters and uses GQA.",
        "It was trained on a multilingual corpus.",
    ]
    score_quote = rag._score_grounding(
        ["The model has 7B parameters and uses GQA."], chunks
    )
    score_off = rag._score_grounding(
        ["completely unrelated text about cats"], chunks
    )
    assert score_quote > score_off, (
        f"Verbatim quote should ground higher than unrelated text: "
        f"quote={score_quote!r}, off={score_off!r}"
    )


def test_score_grounding_zero_for_empty_inputs():
    from aikaboom.core.agentic_rag import AgenticRAG

    rag = AgenticRAG.__new__(AgenticRAG)
    rag.embeddings = FakeEmbeddings(size=4)
    assert rag._score_grounding([], ["chunk"]) == 0.0
    assert rag._score_grounding(["stmt"], []) == 0.0
    assert rag._score_grounding(["", "  "], ["chunk"]) == 0.0


# ---------------------------------------------------------------------------
# 12C — display: summary helper, web JSON, CLI line
# ---------------------------------------------------------------------------


def test_summarise_conflicts_counts_all_three_classes():
    """Direct triplet, RAG internal, and RAG external conflicts all
    count toward total; only the RAG ones get split into high/low by
    grounding."""
    from aikaboom.core.conflict_routing import summarise_conflicts

    md = {
        "direct_fields": {
            "license": {
                "value": "MIT",
                "conflict": {"value": "Apache-2.0", "source": "github", "type": "inter"},
            },
        },
        "rag_fields": {
            "domain": {
                "value": "NLP",
                "trace": {
                    "internal_conflicts": {
                        "huggingface": {"narrative": "Yes: A vs B", "grounding": 0.85},
                    },
                    "external_conflicts": [
                        {"sources": ["hf", "arxiv"], "description": "Yes: ...", "grounding": 0.40},
                    ],
                },
            },
        },
    }
    s = summarise_conflicts(md)
    assert s["total"] == 3
    assert s["high_confidence"] == 1  # the 0.85 grounded one
    assert s["low_confidence"] == 1  # the 0.40 grounded one
    # 3 - 1 - 1 = 1 (the deterministic direct triplet).


def test_summarise_conflicts_handles_missing_grounding():
    """When the auditor LLM returns a conflict but the embedding step
    didn't run (legacy state), grounding may be missing — those still
    count toward total but neither high nor low."""
    from aikaboom.core.conflict_routing import summarise_conflicts

    md = {
        "rag_fields": {
            "x": {
                "value": "v",
                "trace": {
                    "internal_conflicts": {"hf": {"narrative": "Yes: A vs B"}},
                    "external_conflicts": [],
                },
            },
        },
    }
    s = summarise_conflicts(md)
    assert s["total"] == 1
    assert s["high_confidence"] == 0
    assert s["low_confidence"] == 0


def test_summarise_conflicts_empty_metadata():
    from aikaboom.core.conflict_routing import summarise_conflicts

    assert summarise_conflicts({}) == {"total": 0, "high_confidence": 0, "low_confidence": 0}
    assert summarise_conflicts(None) == {"total": 0, "high_confidence": 0, "low_confidence": 0}


def test_confidence_label_threshold():
    """Exactly at threshold (0.7) is high; just below is low."""
    from aikaboom.core.conflict_routing import _confidence_label, HIGH_CONFIDENCE_THRESHOLD

    assert HIGH_CONFIDENCE_THRESHOLD == 0.7
    assert _confidence_label(0.7) == "high"
    assert _confidence_label(0.69) == "low"
    assert _confidence_label(None) == "n/a"
    assert _confidence_label(0.0) == "low"
    assert _confidence_label(1.0) == "high"


def test_extract_conflicts_surfaces_rag_internal_and_external():
    """Web UI helper picks up RAG trace conflicts, not just direct
    triplet conflicts."""
    from aikaboom.web.app import _extract_conflicts

    md = {
        "rag_fields": {
            "domain": {
                "value": "NLP",
                "source": "huggingface",
                "trace": {
                    "claims": {"huggingface": "NLP", "arxiv": "Computer Vision"},
                    "internal_conflicts": {
                        "huggingface": {"narrative": "Yes: foo vs bar", "grounding": 0.42},
                    },
                    "external_conflicts": [
                        {"sources": ["huggingface", "arxiv"],
                         "description": "Yes: NLP vs CV",
                         "grounding": 0.85},
                    ],
                },
            },
        },
    }
    conflicts = _extract_conflicts(md)
    kinds = {c["kind"] for c in conflicts}
    assert "rag-internal" in kinds
    assert "rag-external" in kinds
    high = [c for c in conflicts if c["confidence"] == "high"]
    low = [c for c in conflicts if c["confidence"] == "low"]
    assert len(high) == 1
    assert len(low) == 1
    # Each has a numeric grounding.
    assert any(isinstance(c["grounding"], float) for c in conflicts)


def test_extract_conflicts_includes_direct_triplet_with_na_confidence():
    """Direct/intra triplet conflicts (deterministic) get confidence
    'n/a' — they don't have an LLM-judgement score."""
    from aikaboom.web.app import _extract_conflicts

    md = {
        "direct_fields": {
            "license": {
                "value": "MIT",
                "source": "huggingface",
                "conflict": {"value": "Apache-2.0", "source": "github", "type": "inter"},
            },
        },
    }
    conflicts = _extract_conflicts(md)
    assert len(conflicts) == 1
    assert conflicts[0]["confidence"] == "n/a"
    assert conflicts[0]["grounding"] is None
