#!/usr/bin/env python3
"""Re-scan every gap field on every cached dataset with the *current*
question-bank state. Shows whether the gold chunk is in top-10 now.

Use to confirm no cross-field regression after Thread-B HyDE rewrites.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from diagnose_retrieval_faithful import (  # noqa: E402
    _make_pipeline, _chunk_and_index, _global_retrieve,
    _best_gold_in_pool, _gold_coverage, _is_gold, _load_gaps,
)
from aikaboom.utils.question_bank import load_question_bank, dense_query, sparse_query  # noqa: E402

CACHE = _REPO_ROOT / "test_outputs" / "_cache"

def main():
    rag = _make_pipeline()
    bank = load_question_bank("data")
    cached_slugs = [p.stem.replace("_content", "") for p in CACHE.glob("*_content.json")]
    rows = []
    for slug in cached_slugs:
        content = json.loads((CACHE / f"{slug}_content.json").read_text())
        retrievers = _chunk_and_index(content, rag.embeddings)
        for field, gold_preview, _src in _load_gaps(slug):
            entry = bank.get(field, {})
            top = _global_retrieve(retrievers, dense_query(entry), sparse_query(entry))
            hit = None
            for i, c in enumerate(top, 1):
                cov = _gold_coverage(gold_preview, c["doc"].page_content)
                if cov >= 0.30:
                    hit = (i, cov, c["source"])
                    break
            rows.append((slug, field, hit))

    in_top = sum(1 for *_, h in rows if h)
    miss = len(rows) - in_top
    print(f"\nGap-fields where gold-chunk is now in top-10: {in_top}/{len(rows)}")
    print(f"\n{'slug':14s}  {'field':28s}  result")
    print("-" * 70)
    for slug, field, hit in sorted(rows):
        if hit:
            print(f"  {slug:12s}  {field:28s}  #{hit[0]}({hit[2]}, cov={hit[1]:.2f}) ✅")
        else:
            print(f"  {slug:12s}  {field:28s}  not in top-10 ❌")

if __name__ == "__main__":
    main()
