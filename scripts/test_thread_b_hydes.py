#!/usr/bin/env python3
"""Offline A/B test of candidate HyDEs against cached source content.

For each retrieval-fail case, compare:
  - OLD: current question-bank HyDE/BM25
  - NEW: candidate HyDE/BM25 from a JSON config

Per case, print whether the gold chunk (or best gold-shaped chunk in
the pool) made the top-10 with each query, plus its rank.

No network. Uses the on-disk content cache populated by
``scripts/scan_thread_b.py``.

Usage::

    python scripts/test_thread_b_hydes.py \\
        --candidates test_outputs/_cache/thread_b_candidate_hydes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from diagnose_retrieval_faithful import (  # noqa: E402
    SOURCES,
    _make_pipeline,
    _chunk_and_index,
    _global_retrieve,
    _best_gold_in_pool,
    _gold_coverage,
    _is_gold,
    _load_gaps,
)
from aikaboom.utils.question_bank import load_question_bank, dense_query, sparse_query  # noqa: E402


CACHE = _REPO_ROOT / "test_outputs" / "_cache"


# Cases known to be retrieval-bottlenecked (from the Thread-B scan)
RETRIEVAL_FAIL_CASES = [
    ("mmlu", "knownBias"),
    ("mmlu", "dataCollectionProcess"),
    ("mmlu", "dataPreprocessing"),
    ("gsm8k", "datasetSize"),
    ("mbpp", "datasetNoise"),
    ("mbpp", "knownBias"),
    ("c4", "datasetNoise"),
    ("c4", "intendedUse"),
    # bumped from Thread A as borderline retrieval (code_contests excluded —
    # no cache because the scan was killed mid-fetch by GitHub rate limit)
    ("gsm8k", "datasetAvailability"),
]


def _rank_of_best_gold(top: List[dict], gold_preview: str) -> Optional[Tuple[int, float, str]]:
    """Return (rank, cov, source) of the first chunk in top-10 with cov >= 0.30,
    or None if none."""
    for i, c in enumerate(top, 1):
        cov = _gold_coverage(gold_preview, c["doc"].page_content)
        if cov >= 0.30:
            return (i, cov, c["source"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="JSON file with field → {hypothetical_passage, bm25_terms}")
    args = parser.parse_args()

    candidates: Dict[str, dict] = json.loads(Path(args.candidates).read_text())
    print("Loading bge-small-en-v1.5…", file=sys.stderr)
    rag = _make_pipeline()
    bank = load_question_bank("data")

    # Pre-load all cached content
    by_slug: Dict[str, Dict[str, str]] = {}
    for slug, _ in RETRIEVAL_FAIL_CASES:
        if slug in by_slug:
            continue
        cache_file = CACHE / f"{slug}_content.json"
        if not cache_file.exists():
            print(f"  ✗ no cache for {slug}; run scripts/scan_thread_b.py first", file=sys.stderr)
            return 1
        by_slug[slug] = json.loads(cache_file.read_text())

    # Build retrievers once per slug
    retrievers_by_slug: Dict[str, dict] = {}
    for slug, content in by_slug.items():
        print(f"  chunking {slug}…", file=sys.stderr)
        retrievers_by_slug[slug] = _chunk_and_index(content, rag.embeddings)

    print(f"\n{'='*80}\n  THREAD-B A/B  ({len(RETRIEVAL_FAIL_CASES)} cases)\n{'='*80}")

    summary_rows = []
    for slug, field in RETRIEVAL_FAIL_CASES:
        retrievers = retrievers_by_slug[slug]
        # Find this gap's gold preview
        gold_preview = None
        for gap_entry in _load_gaps(slug):
            if gap_entry[0] == field:
                gold_preview = gap_entry[1]
                break
        if not gold_preview:
            continue

        entry = bank.get(field, {})
        old_dense = dense_query(entry)
        old_sparse = sparse_query(entry)
        cand = candidates.get(field)
        if cand:
            new_dense = cand["hypothetical_passage"]
            new_sparse = " ".join(cand["bm25_terms"])
        else:
            new_dense = old_dense
            new_sparse = old_sparse

        top_old = _global_retrieve(retrievers, old_dense, old_sparse)
        top_new = _global_retrieve(retrievers, new_dense, new_sparse)
        old_hit = _rank_of_best_gold(top_old, gold_preview)
        new_hit = _rank_of_best_gold(top_new, gold_preview)

        old_str = f"#{old_hit[0]}({old_hit[2]}, cov={old_hit[1]:.2f})" if old_hit else "—"
        new_str = f"#{new_hit[0]}({new_hit[2]}, cov={new_hit[1]:.2f})" if new_hit else "—"
        verdict = (
            "✅ FIXED" if not old_hit and new_hit else
            "⚠ REGRESS" if old_hit and not new_hit else
            "(stayed in)" if old_hit else "(both miss)"
        )

        print(f"\n  {slug:14s} / {field:24s}  OLD: {old_str:30s}  →  NEW: {new_str:30s}  {verdict}")
        if not new_hit:
            best = _best_gold_in_pool(retrievers, gold_preview, top_n=1)
            if best:
                print(f"    (best in pool: [{best[0]['source']}] cov={best[0]['cov']:.2f} — exists but didn't retrieve)")

        summary_rows.append({
            "slug": slug, "field": field,
            "old_hit": old_hit, "new_hit": new_hit, "verdict": verdict,
        })

    # Tally
    fixed = sum(1 for r in summary_rows if r["verdict"] == "✅ FIXED")
    regressed = sum(1 for r in summary_rows if r["verdict"] == "⚠ REGRESS")
    stayed = sum(1 for r in summary_rows if r["verdict"] == "(stayed in)")
    both_miss = sum(1 for r in summary_rows if r["verdict"] == "(both miss)")
    print(f"\n  TOTALS: fixed={fixed}, regressed={regressed}, already-in={stayed}, both-miss={both_miss}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
