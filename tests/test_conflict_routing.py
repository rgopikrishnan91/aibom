"""Tests for the consensus-based chunk router ``_route_chunks``.

The router scores each speaking source on a +1-per-agreeing,
−2-per-self-contradicting basis and filters chunks to the highest-
scoring source(s). Static priority breaks ties only when there is
an unresolved external conflict.
"""

import pytest

from aikaboom.core.conflict_routing import _route_chunks


class _Doc:
    """Minimal stand-in for langchain Documents."""

    def __init__(self, source, content="..."):
        self.metadata = {"source": source}
        self.page_content = content


PRIORITY = ["huggingface", "arxiv", "github"]


def _docs(*sources):
    return [_Doc(s) for s in sources]


def _ext(a, b, desc="x vs y"):
    return {"sources": [a, b], "description": desc}


def test_no_conflicts_keeps_all_chunks():
    docs = _docs("huggingface", "arxiv", "github")
    claims = {"huggingface": "x", "arxiv": "x", "github": "x"}
    out, selected = _route_chunks(docs, claims, {}, [], PRIORITY)
    assert out == docs
    assert set(selected) == {"huggingface", "arxiv", "github"}


def test_single_source_only_keeps_all():
    docs = _docs("github", "github")
    claims = {"github": "decoder-only"}
    out, selected = _route_chunks(docs, claims, {}, [], PRIORITY)
    assert out == docs
    assert selected == ["github"]


def test_two_source_disagreement_breaks_tie_with_static_priority():
    """HF and arXiv disagree, no internal conflicts. Both score 0; static
    priority breaks tie → HF wins (top of priority list)."""
    docs = _docs("huggingface", "arxiv")
    claims = {"huggingface": "X", "arxiv": "Y"}
    external = [_ext("huggingface", "arxiv")]
    out, selected = _route_chunks(docs, claims, {}, external, PRIORITY)
    assert selected == ["huggingface"]
    assert all(d.metadata["source"] == "huggingface" for d in out)


def test_three_source_majority_overrides_static_priority():
    """Load-bearing case: arXiv and GitHub agree; HuggingFace disagrees.
    HF is top of static priority but consensus must override.
    Expected: HF dropped, arXiv + GitHub kept."""
    docs = _docs("huggingface", "arxiv", "github")
    claims = {"huggingface": "decoder-only", "arxiv": "encoder-decoder", "github": "encoder-decoder"}
    external = [_ext("huggingface", "arxiv"), _ext("huggingface", "github")]
    out, selected = _route_chunks(docs, claims, {}, external, PRIORITY)
    # HF must be dropped despite being top of priority.
    assert "huggingface" not in selected
    assert set(selected) == {"arxiv", "github"}
    assert {d.metadata["source"] for d in out} == {"arxiv", "github"}


def test_all_three_disagree_falls_to_static_priority():
    """All pairwise conflicts, no internal conflicts → all score 0;
    static priority breaks the three-way tie → HF wins."""
    docs = _docs("huggingface", "arxiv", "github")
    claims = {"huggingface": "X", "arxiv": "Y", "github": "Z"}
    external = [
        _ext("huggingface", "arxiv"),
        _ext("huggingface", "github"),
        _ext("arxiv", "github"),
    ]
    out, selected = _route_chunks(docs, claims, {}, external, PRIORITY)
    assert selected == ["huggingface"]


def test_self_contradicting_source_dropped():
    """HF self-contradicts; arXiv and GitHub are clean and agree.
    HF should be dropped."""
    docs = _docs("huggingface", "arxiv", "github")
    claims = {"huggingface": "ambiguous", "arxiv": "X", "github": "X"}
    internal = {"huggingface": '"a" vs "b"'}
    out, selected = _route_chunks(docs, claims, internal, [], PRIORITY)
    assert "huggingface" not in selected
    assert set(selected) == {"arxiv", "github"}


def test_all_sources_self_contradict_keeps_all():
    """Every source self-contradicts → best score < 0 → surrender,
    feed all chunks to the answerer."""
    docs = _docs("huggingface", "arxiv", "github")
    claims = {"huggingface": "x", "arxiv": "y", "github": "z"}
    internal = {
        "huggingface": "vs",
        "arxiv": "vs",
        "github": "vs",
    }
    out, selected = _route_chunks(docs, claims, internal, [], PRIORITY)
    assert out == docs
    assert set(selected) == {"huggingface", "arxiv", "github"}


def test_silent_source_does_not_score():
    """GitHub silent (None claim); HF and arXiv disagree.
    GitHub gets no score, tie between HF and arXiv → static priority."""
    docs = _docs("huggingface", "arxiv", "github")
    claims = {"huggingface": "X", "arxiv": "Y", "github": None}
    external = [_ext("huggingface", "arxiv")]
    out, selected = _route_chunks(docs, claims, {}, external, PRIORITY)
    # github has no claim so it should never appear in selection
    assert "github" not in selected
    assert selected == ["huggingface"]


def test_zero_speaking_sources_keeps_all():
    """All sources silent → no consensus to compute, return everything."""
    docs = _docs("huggingface", "arxiv")
    claims = {"huggingface": None, "arxiv": None}
    # Pretend an external conflict was flagged (defensive — silent sources
    # shouldn't be in conflict, but the router should still surrender safely).
    out, selected = _route_chunks(docs, claims, {}, [_ext("huggingface", "arxiv")], PRIORITY)
    assert out == docs
    assert selected == []
