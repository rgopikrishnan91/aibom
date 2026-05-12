#!/usr/bin/env python3
"""Cheap A/B test for relaxed extraction prompts on Thread-A (extraction-
bottleneck) fields.

For each known extraction failure (chunks already in production top-10 but
the LLM emits ``noAssertion``), we:

1. Reproduce the top-10 chunks using the production-faithful diagnostic.
2. Build the answer-generation prompt twice — once with the current
   ``output_guidance`` and once with a relaxed candidate.
3. Run both through ``openai/gpt-4o-mini`` via OpenRouter.
4. Print the two answers side-by-side so we can judge whether the
   relaxation actually unblocks extraction.

Total LLM calls: ~2 per field × ~5 candidate fields = 10 small calls.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from aikaboom.utils.question_bank import (  # noqa: E402
    load_question_bank,
    dense_query,
    sparse_query,
    extraction_prompt_parts,
)
from aikaboom.core.prompt import prompt_generate_answer  # noqa: E402

from diagnose_retrieval_faithful import (  # noqa: E402
    SOURCES,
    _make_pipeline,
    _fetch_sources,
    _chunk_and_index,
    _global_retrieve,
)


# Cases where the faithful diagnostic showed gold chunk(s) already in top-10
# but production still returned noAssertion (or equivalent unhelpful value).
EXTRACTION_FAIL_CASES = [
    ("mmlu", "intendedUse"),
    ("mbpp", "intendedUse"),
    ("code_contests", "intendedUse"),
    ("code_contests", "datasetNoise"),
    ("code_contests", "knownBias"),
    ("mbpp", "dataPreprocessing"),
    # second-batch: datasetAvailability cases
    ("mmlu", "datasetAvailability"),
    ("gsm8k", "datasetAvailability"),
    ("c4", "datasetAvailability"),
    # and gsm8k/dataPreprocessing for completeness
    ("gsm8k", "dataPreprocessing"),
]


# Relaxed output_guidance — drops the "literally titled" requirement and
# the strict "do not infer from name/modality alone" rule for cases where
# the dataset description is the canonical place the info lives.
RELAXED_GUIDANCE: Dict[str, str] = {
    "intendedUse": (
        "Look for usage information anywhere in the dataset description, "
        "summary, the paper's abstract or introduction, or sections titled "
        "'Intended Uses', 'Use Cases', 'Direct Use', or 'Out-of-Scope Use'. "
        "If a section title isn't present, extract from the natural-language "
        "description of what the dataset is for. If sources distinguish "
        "recommended vs out-of-scope uses, include both with explicit "
        "labels. Return noAssertion only when no statement of purpose, "
        "intended audience, or use case appears anywhere in the context."
    ),
    "knownBias": (
        "Look for bias, fairness, demographic-imbalance, or "
        "representational-skew discussion anywhere in the context — sections "
        "titled 'Bias', 'Biases', 'Discussion of Biases', or 'Bias, Risks, "
        "and Limitations' are preferred but not required. Topic-distribution "
        "observations (e.g. 'most questions are mathematical', 'predominantly "
        "Codeforces'), demographic-imbalance statements, and performance "
        "skews across subgroups all count as documented bias. Quote concrete "
        "numbers when present. If multiple biases are listed, combine them. "
        "Return noAssertion only when no such discussion appears."
    ),
    "datasetNoise": (
        "Look for any acknowledgement of noise, label disagreement, "
        "annotation errors, mislabelled examples, duplicates, OCR errors, "
        "compilation failures, or other data-quality issues anywhere in the "
        "context. Statements about labels 'not being guaranteed correct', "
        "tests being 'flaky' or 'noisy', or known issues with subsets all "
        "count. Capture quantitative estimates (error rates, percentages) "
        "verbatim. Distinguish per-field noise from dataset-wide noise. "
        "Return noAssertion only when no quality caveat appears."
    ),
    "dataPreprocessing": (
        "Look for any processing applied to the data after collection: "
        "splitting into train/dev/test/few-shot, filtering, deduplication, "
        "tokenization, lowercasing, normalization, language ID, PII "
        "scrubbing, annotation injection, manual review, second-round "
        "curation, etc. The split definitions and annotation steps count "
        "as preprocessing even when not under a literal '## Preprocessing' "
        "header. Preserve numeric thresholds and tool names verbatim. "
        "Return noAssertion only when the context describes only raw "
        "collection with no subsequent processing of any kind."
    ),
    "datasetAvailability": (
        "Map phrasings to the enum: 'pip / load_dataset / wget / direct "
        "download / publicly available' without forms → direct-download; "
        "'click I agree / accept terms / click-through' → clickthrough; "
        "'fill this form / request access / gated / registration required' "
        "→ registration; 'use our SQL/REST/API endpoint to query' → query; "
        "'run this crawler / scraping script / scrape from source URLs' → "
        "scraping-script. A repository link like github.com/owner/repo or "
        "huggingface.co/datasets/owner/name without any access barrier counts "
        "as direct-download. If multiple access paths exist, choose the most "
        "permissive single mechanism. Return only the bare enum value with "
        "no explanatory prose. Return noAssertion only when no access "
        "mechanism appears anywhere in the context."
    ),
}


def call_llm(prompt: str, model: str = "openai/gpt-4o-mini") -> str:
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_NAME", "BOM_Tools"),
            "X-Title": "BOM_Tools",
        },
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def run(slug: str, field: str, rag, retrievers, bank, verbose: bool = False) -> dict:
    entry = bank.get(field, {})
    dense_q = dense_query(entry)
    sparse_q = sparse_query(entry)

    top = _global_retrieve(retrievers, dense_q, sparse_q)
    context = "---\n" + "\n---\n".join(c["doc"].page_content for c in top) + "\n---"

    parts = extraction_prompt_parts(entry)
    old_prompt = prompt_generate_answer(
        instruction=parts["instruction"],
        field_spec=parts["field_spec"],
        output_guidance=parts["output_guidance"],
        context=context,
    )
    new_prompt = prompt_generate_answer(
        instruction=parts["instruction"],
        field_spec=parts["field_spec"],
        output_guidance=RELAXED_GUIDANCE.get(field, parts["output_guidance"]),
        context=context,
    )

    old_ans = call_llm(old_prompt)
    new_ans = call_llm(new_prompt)
    return {"slug": slug, "field": field, "old": old_ans, "new": new_ans, "chunks": len(top)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="all",
                        help="comma-separated list of slug/field pairs (e.g. mmlu/intendedUse) or 'all'")
    args = parser.parse_args()

    cases = EXTRACTION_FAIL_CASES
    if args.cases != "all":
        cases = [tuple(c.split("/")) for c in args.cases.split(",")]

    # Group by slug so we only build retrievers once per dataset
    rag = _make_pipeline()
    bank = load_question_bank("data")
    by_slug: Dict[str, List[str]] = {}
    for slug, field in cases:
        by_slug.setdefault(slug, []).append(field)

    results = []
    for slug, fields in by_slug.items():
        repo, arxiv, github = SOURCES[slug]
        print(f"\n{'='*80}\n  PROCESSING {slug}\n{'='*80}", file=sys.stderr)
        content = _fetch_sources(rag, repo, arxiv, github)
        retrievers = _chunk_and_index(content, rag.embeddings)
        for field in fields:
            print(f"\n  → {slug}/{field}", file=sys.stderr)
            r = run(slug, field, rag, retrievers, bank)
            results.append(r)
            print(f"    OLD: {r['old'][:120]}", file=sys.stderr)
            print(f"    NEW: {r['new'][:120]}", file=sys.stderr)

    # Final report
    print(f"\n\n{'='*80}\n  A/B RESULTS  ({len(results)} cases)\n{'='*80}\n")
    for r in results:
        print(f"━━━ {r['slug']}/{r['field']}  (top-{r['chunks']} chunks) ━━━")
        print(f"  OLD prompt → {r['old']}")
        print(f"  NEW prompt → {r['new']}")
        old_nil = r["old"].strip().lower() in {"noassertion", "no assertion", ""}
        new_nil = r["new"].strip().lower() in {"noassertion", "no assertion", ""}
        if old_nil and not new_nil:
            verdict = "✅ FIXED"
        elif old_nil and new_nil:
            verdict = "❌ both nil"
        elif not old_nil and new_nil:
            verdict = "⚠ regression"
        else:
            verdict = "(both populated)"
        print(f"  Verdict: {verdict}\n")

    out_json = _REPO_ROOT / "test_outputs" / "extraction_prompts_ab.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
