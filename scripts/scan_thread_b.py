#!/usr/bin/env python3
"""Thread B scan: for every retrieval-fail case, show what the best
gold-matching chunk anywhere in the indexed pool looks like.

This is one Python process, so the heavy bits (bge-small-en-v1.5
embedding model load, langchain/torch imports) happen once. Fetched
content is cached to ``test_outputs/_cache/{slug}.json`` so subsequent
runs skip the network entirely.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from diagnose_retrieval_faithful import (  # noqa: E402
    SOURCES,
    _make_pipeline,
    _fetch_sources,
    _chunk_and_index,
    _global_retrieve,
    _best_gold_in_pool,
    _gold_coverage,
    _is_gold,
    _load_gaps,
)
from aikaboom.utils.question_bank import load_question_bank, dense_query, sparse_query  # noqa: E402


CACHE = _REPO_ROOT / "test_outputs" / "_cache"
CACHE.mkdir(exist_ok=True)


def fetch_with_cache(rag, slug: str) -> Dict[str, str]:
    cache_file = CACHE / f"{slug}_content.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    repo, arxiv, github = SOURCES[slug]
    content = _fetch_sources(rag, repo, arxiv, github)
    cache_file.write_text(json.dumps(content))
    return content


def main() -> int:
    print("Loading bge-small-en-v1.5 + ancillary…", file=sys.stderr)
    rag = _make_pipeline()
    bank = load_question_bank("data")

    for slug in ("mmlu", "gsm8k", "mbpp", "c4", "code_contests"):
        print(f"\n{'='*78}\n  {slug}\n{'='*78}", file=sys.stderr)
        content = fetch_with_cache(rag, slug)
        retrievers = _chunk_and_index(content, rag.embeddings)

        for field, gold_preview, sources_str in _load_gaps(slug):
            entry = bank.get(field, {})
            top = _global_retrieve(retrievers, dense_query(entry), sparse_query(entry))
            gold_hits = [(i + 1, c) for i, c in enumerate(top)
                         if _is_gold(c["doc"].page_content, gold_preview)]
            status = "IN-TOP-10" if gold_hits else "NOT-IN-TOP-10"
            print(f"\n──── {slug}/{field} [{status}]  (gap sources: {sources_str})")
            print(f"  gold preview: {gold_preview[:110]!r}")
            if gold_hits:
                gold_str = ", ".join(
                    f"#{r}({c['source']}, cov={_gold_coverage(gold_preview, c['doc'].page_content):.2f})"
                    for r, c in gold_hits
                )
                print(f"  HIT: {gold_str}")
            else:
                best = _best_gold_in_pool(retrievers, gold_preview, top_n=3)
                if best:
                    print("  BEST IN POOL (regardless of rank):")
                    for b in best:
                        prev = b["doc"].page_content.replace("\n", " ")[:140]
                        print(f"    [{b['source']:11s} cov={b['cov']:.2f}] {prev!r}")
                else:
                    print("  (no chunk in pool has any token overlap with gold preview)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
