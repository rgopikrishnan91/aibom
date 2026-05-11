#!/usr/bin/env python3
"""Production-faithful retrieval diagnostic.

Replaces the v1 diagnostic (``diagnose_dataset_retrieval.py``) which
diverged from production's chunking because it didn't insert
``## Page N`` markers in the arxiv text and didn't prepend the
structured-metadata chunks. That divergence over-predicted the
production gains.

This v2 diagnostic re-uses the *actual* production code:

  - ``AgenticRAG.fetch_arxiv_pdf_text`` — emits ``## Page N`` headers.
  - ``AgenticRAG.fetch_github_readme``
  - ``AgenticRAG.fetch_huggingface_readme``
  - ``AgenticRAG._normalize_markdown`` — adds the source-source header.
  - ``MetadataFetcher.{huggingface,github}_structured_chunk`` — the
    structured-metadata block prepended to each source's text.
  - ``HeaderAwareTextSplitter(chunk_size=1200, chunk_overlap=300, min_chunk_size=150)``
  - ``langchain_huggingface.HuggingFaceEmbeddings`` (BAAI/bge-small-en-v1.5,
    normalize_embeddings=True) — same instance shape as production.
  - The exact RRF fusion in ``_global_retrieve_top_k``
    (per-source dense20 + sparse20, RRF k=60, global top-K=10).

No LLM init is needed for retrieval; we bypass ``AgenticRAG.__init__``
via ``__new__`` to avoid the OpenAI client setup.

Usage::

    python scripts/diagnose_retrieval_faithful.py --slug mmlu
    python scripts/diagnose_retrieval_faithful.py --slug mmlu --new-hyde my_passage_file.json

When ``--new-hyde`` is given, the script re-runs retrieval with the
candidate HyDE/BM25 in that JSON in place of the question-bank entry,
so we can iterate offline without touching the question bank.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from aikaboom.core.agentic_rag import (  # noqa: E402
    AgenticRAG,
    HeaderAwareTextSplitter,
)
from aikaboom.utils.chunk_filter import is_useful_chunk  # noqa: E402
from aikaboom.utils.metadata_fetcher import MetadataFetcher  # noqa: E402
from aikaboom.utils.question_bank import load_question_bank, dense_query, sparse_query  # noqa: E402

import re
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #

SOURCES = {
    "mmlu":          ("cais/mmlu",                       "https://arxiv.org/pdf/2009.03300", "https://github.com/hendrycks/test"),
    "gsm8k":         ("openai/gsm8k",                    "https://arxiv.org/pdf/2110.14168", "https://github.com/openai/grade-school-math"),
    "mbpp":          ("google-research-datasets/mbpp",   "https://arxiv.org/pdf/2108.07732", "https://github.com/google-research/google-research/tree/master/mbpp"),
    "c4":            ("allenai/c4",                      "https://arxiv.org/pdf/1910.10683", "https://github.com/google-research/text-to-text-transfer-transformer#datasets"),
    "code_contests": ("deepmind/code_contests",          "https://arxiv.org/pdf/2203.07814", "https://github.com/google-deepmind/code_contests"),
}


# --------------------------------------------------------------------------- #
# Faithful pipeline assembly                                                   #
# --------------------------------------------------------------------------- #

def _make_pipeline():
    """Build a partial AgenticRAG that has fetchers + normaliser + embeddings,
    but no LLM. We bypass __init__ entirely to avoid OpenAI client setup."""
    rag = AgenticRAG.__new__(AgenticRAG)
    rag.embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    # The fetcher methods need an HfApi for huggingface card fetching
    from huggingface_hub import HfApi
    from github import Github
    rag.hf_api = HfApi(token=os.environ.get("HUGGINGFACE_TOKEN"))
    rag.g = Github(os.environ.get("GITHUB_TOKEN"))
    return rag


def _fetch_sources(rag, repo: str, arxiv_url: str, github_url: str) -> Dict[str, str]:
    """Mirror production: fetch arxiv/HF/GH content, prepend structured chunks
    when available, normalise. Returns ``{source: full_normalised_text}``."""
    content = {}
    print("Fetching arxiv…", file=sys.stderr)
    content["arxiv"] = rag.fetch_arxiv_pdf_text(arxiv_url) or ""
    print(f"  → {len(content['arxiv']):,} chars", file=sys.stderr)

    print("Fetching github…", file=sys.stderr)
    content["github"] = rag.fetch_github_readme(github_url) or ""
    print(f"  → {len(content['github']):,} chars", file=sys.stderr)

    print("Fetching huggingface…", file=sys.stderr)
    hf_url = f"https://huggingface.co/datasets/{repo}"
    content["huggingface"] = rag.fetch_huggingface_readme(hf_url) or ""
    print(f"  → {len(content['huggingface']):,} chars", file=sys.stderr)

    # Production injects structured chunks (HF metadata / GH metadata) at
    # the top of each source's text. Replicate that.
    from github import Github
    gh_client = Github(os.environ.get("GITHUB_TOKEN"))
    try:
        from aikaboom.utils.metadata_fetcher import MetadataFetcher as MF
        gh_repo_path = MF.extract_repo_path(github_url)
        gh_repo = gh_client.get_repo(gh_repo_path)
        gh_struct = MF.github_structured_chunk(gh_repo, bom_type="data")
        if gh_struct:
            content["github"] = (gh_struct + "\n\n" + content["github"]).strip()
            print(f"  + github structured chunk: {len(gh_struct):,} chars", file=sys.stderr)
    except Exception as e:
        print(f"  ⚠️ github structured chunk skipped: {e}", file=sys.stderr)

    try:
        hf_info = rag.hf_api.dataset_info(repo)
        hf_struct = MF.huggingface_structured_chunk(hf_info, bom_type="data")
        if hf_struct:
            content["huggingface"] = (hf_struct + "\n\n" + content["huggingface"]).strip()
            print(f"  + huggingface structured chunk: {len(hf_struct):,} chars", file=sys.stderr)
    except Exception as e:
        print(f"  ⚠️ huggingface structured chunk skipped: {e}", file=sys.stderr)

    # Normalise — same as production
    return {src: rag._normalize_markdown(text, src) for src, text in content.items() if text}


def _chunk_and_index(content_dict: Dict[str, str], embeddings) -> Dict[str, dict]:
    """Mirror ``create_vector_stores``: chunk, build FAISS + BM25 per source."""
    splitter = HeaderAwareTextSplitter(chunk_size=1200, chunk_overlap=300, min_chunk_size=150)
    retrievers = {}
    for source, content in content_dict.items():
        if not content or len(content) <= 50:
            continue
        chunks = splitter.split_text(content)
        if not chunks:
            continue
        docs = [Document(page_content=c, metadata={"source": source, "chunk_index": i})
                for i, c in enumerate(chunks)]
        vectorstore = FAISS.from_documents(docs, embeddings)
        tokenised = [d.page_content.lower().split() for d in docs]
        bm25 = BM25Okapi(tokenised) if tokenised else None
        retrievers[source] = {"vectorstore": vectorstore, "bm25": bm25, "docs": docs}
        print(f"  {source}: {len(chunks)} chunks indexed", file=sys.stderr)
    return retrievers


def _global_retrieve(
    retrievers: Dict[str, dict],
    dense_q: str,
    sparse_q: str,
    top_k_per_source: int = 20,
    final_top_k: int = 10,
    rrf_k: int = 60,
) -> List[Dict]:
    """Mirror ``_global_retrieve_top_k``: per-source RRF fusion, global top-K."""
    all_chunks: List[dict] = []
    for source, bundle in retrievers.items():
        vectorstore = bundle["vectorstore"]
        bm25 = bundle.get("bm25")
        docs = bundle.get("docs", [])

        dense_hits = vectorstore.similarity_search_with_score(dense_q, k=top_k_per_source)
        dense_ranked = [doc for doc, _ in dense_hits]

        sparse_ranked = []
        if bm25 is not None and sparse_q.strip() and docs:
            bm25_scores = bm25.get_scores(sparse_q.lower().split())
            sparse_indices = sorted(range(len(docs)), key=lambda i: -bm25_scores[i])[:top_k_per_source]
            sparse_ranked = [docs[i] for i in sparse_indices]

        rrf: Dict[int, float] = {}
        lookup: Dict[int, Document] = {}
        for ranked in (dense_ranked, sparse_ranked):
            for rank, doc in enumerate(ranked):
                doc_id = doc.metadata.get("chunk_index", id(doc))
                rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
                lookup[doc_id] = doc

        fused = sorted(rrf.items(), key=lambda x: -x[1])
        for doc_id, rrf_score in fused:
            doc = lookup[doc_id]
            if not is_useful_chunk(doc.page_content):
                continue
            all_chunks.append({"source": source, "doc": doc, "score": rrf_score})

    all_chunks.sort(key=lambda x: -x["score"])
    return all_chunks[:final_top_k]


# --------------------------------------------------------------------------- #
# Gold-chunk identification + reporting                                       #
# --------------------------------------------------------------------------- #

def _content_tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 3}


def _gold_coverage(gold_preview: str, chunk_text: str) -> float:
    g = _content_tokens(gold_preview)
    if not g:
        return 0.0
    c = _content_tokens(chunk_text)
    return len(g & c) / len(g) if c else 0.0


def _is_gold(chunk_text: str, gold_preview: str, threshold: float = 0.30) -> bool:
    return _gold_coverage(gold_preview, chunk_text) >= threshold


def _best_gold_in_pool(retrievers: Dict[str, dict], gold_preview: str, top_n: int = 3) -> List[dict]:
    """Find the highest-coverage chunks across the *entire* indexed pool
    (regardless of HyDE/BM25 score). Tells us what content the gold answer
    lives in, even when current retrieval doesn't surface it.
    """
    all_scored = []
    for source, bundle in retrievers.items():
        for doc in bundle.get("docs", []):
            cov = _gold_coverage(gold_preview, doc.page_content)
            if cov > 0:
                all_scored.append({"source": source, "doc": doc, "cov": cov})
    all_scored.sort(key=lambda x: -x["cov"])
    return all_scored[:top_n]


def _load_gaps(slug: str) -> List[Tuple[str, str, str]]:
    gap_path = _REPO_ROOT / "test_outputs" / "golden_baseline" / "gap_analysis.json"
    data = json.loads(gap_path.read_text())
    for entry in data:
        if entry["slug"] == slug and entry["kind"] == "data":
            return [tuple(g) for g in entry.get("gap", [])]
    return []


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def run(slug: str, hyde_overrides: Optional[Dict[str, dict]] = None) -> None:
    repo, arxiv, github = SOURCES[slug]
    print(f"\n{'='*80}\n  FAITHFUL DIAGNOSTIC: {slug}  ({repo})\n{'='*80}\n", file=sys.stderr)

    rag = _make_pipeline()
    content = _fetch_sources(rag, repo, arxiv, github)
    retrievers = _chunk_and_index(content, rag.embeddings)

    gaps = _load_gaps(slug)
    if not gaps:
        print(f"No gaps for {slug}", file=sys.stderr)
        return

    bank = load_question_bank("data")
    for field, gold_preview, sources_str in gaps:
        entry = bank.get(field, {})
        override = (hyde_overrides or {}).get(field)
        if override:
            dense_q = override["hypothetical_passage"]
            sparse_q = " ".join(override["bm25_terms"])
            tag = "NEW"
        else:
            dense_q = dense_query(entry)
            sparse_q = sparse_query(entry)
            tag = "OLD"

        top = _global_retrieve(retrievers, dense_q, sparse_q)
        gold_hits = [(i + 1, c) for i, c in enumerate(top) if _is_gold(c["doc"].page_content, gold_preview)]

        print(f"\n──── {field} [{tag}]  (gap sources: {sources_str}) ────")
        print(f"  gold preview: {gold_preview[:120]!r}")
        if gold_hits:
            gold_str = ", ".join(f"#{r}({c['source']}, cov={_gold_coverage(gold_preview, c['doc'].page_content):.2f})"
                                 for r, c in gold_hits)
            print(f"  GOLD CHUNK IN TOP-10: {gold_str}")
        else:
            print(f"  GOLD CHUNK NOT IN TOP-10")
            # Show the best gold-matching chunks anywhere in the pool so we
            # can see what we should be retrieving.
            best = _best_gold_in_pool(retrievers, gold_preview, top_n=3)
            if best:
                print(f"  BEST IN POOL (regardless of rank):")
                for b in best:
                    preview = b["doc"].page_content.replace("\n", " ")[:160]
                    print(f"    [{b['source']:11s} cov={b['cov']:.2f}] {preview!r}")
            else:
                print(f"  (no chunk in pool has any token overlap with gold preview!)")

        print(f"  Top-10 chunks:")
        for i, c in enumerate(top, 1):
            cov = _gold_coverage(gold_preview, c["doc"].page_content)
            marker = " ⭐" if _is_gold(c["doc"].page_content, gold_preview) else ""
            preview = c["doc"].page_content.replace("\n", " ")[:120]
            print(f"    {i:2d}. [{c['source']:11s} cov={cov:.2f}{marker}] {preview!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default="mmlu", choices=list(SOURCES.keys()))
    parser.add_argument("--new-hyde", type=str, help="Path to a JSON file mapping field → {hypothetical_passage, bm25_terms}")
    args = parser.parse_args()

    overrides = None
    if args.new_hyde:
        overrides = json.loads(Path(args.new_hyde).read_text())
    run(args.slug, overrides)
    return 0


if __name__ == "__main__":
    sys.exit(main())
