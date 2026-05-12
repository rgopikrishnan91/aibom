#!/usr/bin/env python3
"""Experiment: does a smarter LLM prompt match or beat the Phase 12 NLI judge?

For every non-nil claim pair recorded in `test_outputs/phase12_rerun/<kind>/<slug>.json`
(under each RAG field's `trace.claims` block), call the LLM with a
silence-aware + complementary-aware prompt that emits both a verdict
and a confidence score. Compare against the NLI verdict already in the
same BOM and write `test_outputs/PROMPT_V2_RESULTS.md`.

Offline-after-LLM-call: each pair is one small prompt with no retrieval,
no context window concerns. ~200 pairs expected.

Run::

    python scripts/experiment_prompt_v2.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


PHASE12 = _REPO_ROOT / "test_outputs" / "phase12_rerun"
OUT_JSON = _REPO_ROOT / "test_outputs" / "prompt_v2_results.json"
OUT_MD = _REPO_ROOT / "test_outputs" / "PROMPT_V2_RESULTS.md"


PROMPT_TEMPLATE = """You are judging whether two factual claims from different sources about an AI model's "{field}" property genuinely contradict each other.

Claim A (from {src_a}): {claim_a}
Claim B (from {src_b}): {claim_b}

Apply these rules in strict order:

Rule 1 — Silence is not a conflict.
If either claim asserts absence of information rather than a positive fact (phrasings like "no relevant information", "not provided", "not specified", "the source does not mention", "[More Information Needed]", "information not available", "noassertion", "not addressed", "the property is not mentioned"), treat that source as silent. Silence cannot conflict with anything. Output: not_conflict.

Rule 2 — Complementary statements are not conflicts.
If both claims make positive factual assertions but the assertions are complementary rather than mutually exclusive — for example, one mentions {{A, B}} and the other mentions {{B, C}}, or one provides additional detail about the same underlying fact — that is not a conflict. Output: not_conflict.

Rule 3 — Different topics are not conflicts.
If the two claims are about different aspects or different sub-properties (e.g. one is about training data, the other is about model architecture) and do not address the same underlying assertion, output: not_conflict.

Rule 4 — Direct factual contradiction is a conflict.
Only output conflict when the two claims make positive assertions that cannot both be true at the same time (e.g. "licensed under MIT" vs "licensed under GPL"; "trained on 1B tokens" vs "trained on 10B tokens"; "supports up to 32K context" vs "supports up to 1M context" if both are stated as the maximum). When in doubt, prefer not_conflict.

For every conflict you do flag, output a confidence score between 0.0 (weak — the assertions could plausibly be reconciled) and 1.0 (the assertions are explicit opposites and cannot both be true). For not_conflict, set confidence to 0.0.

Respond with a single JSON object only, no other text:
{{"verdict": "conflict" | "not_conflict", "confidence": <0.0-1.0>, "reason": "<one short sentence>"}}
"""


def load_baseline_external_status(slug: str, kind: str, field: str) -> Optional[bool]:
    """Return True if the baseline LLM flagged the field's external conflict, False if not, None if missing."""
    p = _REPO_ROOT / "test_outputs" / "golden_baseline" / kind / f"{slug}.json"
    if not p.exists():
        return None
    bom = json.loads(p.read_text())
    triplet = (bom.get("rag_fields") or {}).get(field) or {}
    conflict = triplet.get("conflict") or {}
    ext = (conflict.get("external") or "").strip()
    if not ext or ext.lower().startswith("no"):
        return False
    return True


def load_nli_pair_score(slug: str, kind: str, field: str, src_a: str, src_b: str) -> Optional[float]:
    """If Phase 12 NLI flagged this exact pair on this field, return its contradiction score."""
    p = PHASE12 / kind / f"{slug}.json"
    if not p.exists():
        return None
    bom = json.loads(p.read_text())
    triplet = (bom.get("rag_fields") or {}).get(field) or {}
    conflict = triplet.get("conflict") or {}
    ext = (conflict.get("external") or "").strip()
    if not ext or ext.lower().startswith("no"):
        return None
    # Parse "NLI contradiction (p=0.97) between arxiv and github"
    pattern = re.compile(
        r"NLI contradiction \(p=([\d.]+)\) between (\S+) and (\S+)",
        re.IGNORECASE,
    )
    a_set, b_set = {src_a, src_b}, None
    for m in pattern.finditer(ext):
        p_score, x, y = m.group(1), m.group(2), m.group(3)
        if {x, y} == {src_a, src_b}:
            try:
                return float(p_score)
            except ValueError:
                return None
    return None


def iter_pairs():
    """Yield (kind, slug, field, src_a, claim_a, src_b, claim_b) for every non-nil claim pair."""
    nil_markers = {
        "", "no relevant information", "no relevant information found",
        "not provided", "not specified", "not stated", "not found",
        "not available", "unknown", "n/a", "none",
        "[more information needed]", "more information needed",
        "noassertion", "no assertion",
    }
    for kind in ("ai", "data"):
        for bom_path in sorted((PHASE12 / kind).glob("*.json")):
            slug = bom_path.stem
            bom = json.loads(bom_path.read_text())
            rag = bom.get("rag_fields") or {}
            for field, triplet in rag.items():
                if not isinstance(triplet, dict):
                    continue
                claims = ((triplet.get("trace") or {}).get("claims") or {})
                non_nil = []
                for src, c in claims.items():
                    if c is None:
                        continue
                    s = str(c).strip()
                    if not s or s.lower() in nil_markers:
                        continue
                    non_nil.append((src, s))
                # all unordered pairs
                for i, (sa, ca) in enumerate(non_nil):
                    for sb, cb in non_nil[i + 1:]:
                        yield (kind, slug, field, sa, ca, sb, cb)


def call_llm(prompt: str, model: str) -> str:
    """One synchronous LLM call. Uses OpenRouter via the openai SDK."""
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
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def parse_verdict(raw: str) -> dict:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {"verdict": "error", "confidence": 0.0, "reason": f"unparseable: {raw[:120]!r}"}
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in ("conflict", "not_conflict"):
        verdict = "error"
    try:
        conf = float(obj.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "verdict": verdict,
        "confidence": conf,
        "reason": str(obj.get("reason", ""))[:200],
    }


def main() -> int:
    model = os.environ.get("PROMPT_V2_MODEL", "openai/gpt-4o-mini")
    pairs = list(iter_pairs())
    print(f"Total non-nil claim pairs: {len(pairs)}", file=sys.stderr)

    results = []
    for i, (kind, slug, field, sa, ca, sb, cb) in enumerate(pairs, 1):
        prompt = PROMPT_TEMPLATE.format(field=field, src_a=sa, claim_a=ca, src_b=sb, claim_b=cb)
        try:
            raw = call_llm(prompt, model)
            verdict = parse_verdict(raw)
        except Exception as exc:  # noqa: BLE001
            verdict = {"verdict": "error", "confidence": 0.0, "reason": str(exc)[:200]}

        nli_score = load_nli_pair_score(slug, kind, field, sa, sb)
        baseline_flagged = load_baseline_external_status(slug, kind, field)

        entry = {
            "kind": kind, "slug": slug, "field": field,
            "src_a": sa, "src_b": sb,
            "claim_a": ca, "claim_b": cb,
            "new_prompt": verdict,
            "nli_pair_score": nli_score,
            "nli_flagged": nli_score is not None,
            "baseline_field_flagged": baseline_flagged,
        }
        results.append(entry)

        marker = "🚩" if verdict["verdict"] == "conflict" else "  "
        nli_marker = f"NLI={nli_score:.2f}" if nli_score is not None else "NLI=-"
        print(f"  {i:3d}/{len(pairs)} {marker} {kind}/{slug:14s} {field:30s} {sa:>11s}↔{sb:<11s} | "
              f"new={verdict['verdict']:14s} c={verdict['confidence']:.2f} | {nli_marker}",
              file=sys.stderr)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT_JSON}", file=sys.stderr)

    _write_summary(results)
    print(f"Wrote {OUT_MD}", file=sys.stderr)
    return 0


def _write_summary(results: List[dict]) -> None:
    n_total = len(results)
    if n_total == 0:
        OUT_MD.write_text("# Prompt v2 results\n\nNo pairs.\n")
        return

    # 4-way contingency on (new_prompt flagged, NLI flagged)
    quadrants: Dict[Tuple[bool, bool], List[dict]] = defaultdict(list)
    for r in results:
        new_flagged = r["new_prompt"]["verdict"] == "conflict"
        nli_flagged = r["nli_flagged"]
        quadrants[(new_flagged, nli_flagged)].append(r)

    md = []
    md.append("# Prompt v2 vs Phase 12 NLI — pairwise results\n")
    md.append(f"Total claim pairs analysed: **{n_total}**.\n")
    md.append("## Agreement table\n")
    md.append("| | NLI flagged | NLI did not flag |")
    md.append("|---|---|---|")
    md.append(f"| **Prompt-v2 flagged** | {len(quadrants[(True, True)])} (both flagged) | {len(quadrants[(True, False)])} (prompt-v2 only) |")
    md.append(f"| **Prompt-v2 did not flag** | {len(quadrants[(False, True)])} (NLI only) | {len(quadrants[(False, False)])} (both cleared) |")

    md.append("\n## Confidence distribution (prompt-v2 flagged cases)\n")
    confs = sorted(r["new_prompt"]["confidence"] for r in results if r["new_prompt"]["verdict"] == "conflict")
    if confs:
        md.append(f"- N = {len(confs)}")
        md.append(f"- Min = {min(confs):.2f}, Median = {confs[len(confs)//2]:.2f}, Max = {max(confs):.2f}")
        buckets = {"≥0.95": 0, "0.80–0.95": 0, "0.50–0.80": 0, "<0.50": 0}
        for c in confs:
            if c >= 0.95: buckets["≥0.95"] += 1
            elif c >= 0.80: buckets["0.80–0.95"] += 1
            elif c >= 0.50: buckets["0.50–0.80"] += 1
            else: buckets["<0.50"] += 1
        md.append("- Buckets: " + ", ".join(f"{k}={v}" for k, v in buckets.items()))

    md.append("\n## Cases where prompt-v2 disagrees with NLI\n")
    md.append("\n### NLI flagged, prompt-v2 cleared (NLI-only)\n")
    md.append("Candidate NLI false positives that the smarter prompt rejects.")
    for r in quadrants[(False, True)][:25]:
        md.append(f"\n- **{r['kind']}/{r['slug']}** · `{r['field']}` · {r['src_a']} vs {r['src_b']} · NLI p={r['nli_pair_score']:.2f}")
        md.append(f"  - {r['src_a']}: {r['claim_a'][:200]}")
        md.append(f"  - {r['src_b']}: {r['claim_b'][:200]}")
        md.append(f"  - prompt-v2: _{r['new_prompt']['reason']}_")
    if len(quadrants[(False, True)]) > 25:
        md.append(f"\n_…{len(quadrants[(False, True)]) - 25} more omitted_")

    md.append("\n### Prompt-v2 flagged, NLI cleared (prompt-v2-only)\n")
    md.append("Candidate real conflicts the NLI missed.")
    for r in quadrants[(True, False)][:25]:
        md.append(f"\n- **{r['kind']}/{r['slug']}** · `{r['field']}` · {r['src_a']} vs {r['src_b']} · prompt-v2 c={r['new_prompt']['confidence']:.2f}")
        md.append(f"  - {r['src_a']}: {r['claim_a'][:200]}")
        md.append(f"  - {r['src_b']}: {r['claim_b'][:200]}")
        md.append(f"  - prompt-v2: _{r['new_prompt']['reason']}_")
    if len(quadrants[(True, False)]) > 25:
        md.append(f"\n_…{len(quadrants[(True, False)]) - 25} more omitted_")

    md.append("\n### Both flagged — sample\n")
    for r in quadrants[(True, True)][:10]:
        md.append(f"\n- **{r['kind']}/{r['slug']}** · `{r['field']}` · {r['src_a']} vs {r['src_b']} · NLI p={r['nli_pair_score']:.2f} · prompt-v2 c={r['new_prompt']['confidence']:.2f}")
        md.append(f"  - prompt-v2 reason: _{r['new_prompt']['reason']}_")

    OUT_MD.write_text("\n".join(md))


if __name__ == "__main__":
    sys.exit(main())
