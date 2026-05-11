#!/usr/bin/env python3
"""Per-improvement attribution for the Phase 12 conflict-detection rebuild.

Reads:
  - test_outputs/golden_baseline/{ai,data}/<slug>.json   (pre-Phase-12)
  - test_outputs/phase12_rerun/{ai,data}/<slug>.json     (post-Phase-12)

Emits ``test_outputs/PHASE12_RESULTS.md`` attributing the deltas back to
findings #34 / #35 / #37.

The script is deterministic and offline — it only reads the BOM JSON
files, no LLM/API calls. Re-run anytime to refresh the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = REPO_ROOT / "test_outputs" / "golden_baseline"
NEW = REPO_ROOT / "test_outputs" / "phase12_rerun"
OUTPUT = REPO_ROOT / "test_outputs" / "PHASE12_RESULTS.md"


AI_SLUGS = ["bert", "clip", "distilbert", "qwen", "roberta"]
DATA_SLUGS = ["c4", "code_contests", "gsm8k", "mbpp", "mmlu"]


NIL_MARKERS = {
    "", "noassertion", "no relevant information",
    "no relevant information found", "not provided", "not specified",
    "not stated", "not found", "n/a", "none", "unknown",
    "[more information needed]", "more information needed",
}


def _is_nil(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in NIL_MARKERS


def load(path: Path) -> Optional[dict]:
    if not path.exists() or not path.stat().st_size:
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def count_direct_conflicts(bom: dict) -> Dict[str, Any]:
    """Per-field conflict status from direct_fields. Returns counts + per-field detail."""
    direct = bom.get("direct_fields", {}) or {}
    flagged: List[str] = []
    has_alternates: List[str] = []
    for field, triplet in direct.items():
        if not isinstance(triplet, dict):
            continue
        if triplet.get("conflict"):
            flagged.append(field)
        if isinstance(triplet.get("alternates"), dict):
            has_alternates.append(field)
    return {
        "flagged_conflicts": flagged,
        "n_conflicts": len(flagged),
        "fields_with_alternates": has_alternates,
        "n_alternates_fields": len(has_alternates),
    }


def count_rag_external_conflicts(bom: dict) -> Dict[str, Any]:
    """Detect RAG external conflicts in rag_fields."""
    rag = bom.get("rag_fields", {}) or {}
    flagged: List[Tuple[str, str]] = []
    nli_scored: List[Tuple[str, float]] = []
    silence_vs_presence: List[Tuple[str, str]] = []
    for field, triplet in rag.items():
        if not isinstance(triplet, dict):
            continue
        conflict = triplet.get("conflict") or {}
        external = ""
        if isinstance(conflict, dict):
            external = (conflict.get("external") or "").strip()
        if not external or external.lower().startswith("no"):
            continue
        flagged.append((field, external))
        # Extract NLI scores from descriptions
        import re
        scores = re.findall(r'p=([\d.]+)', external)
        for s in scores:
            try:
                nli_scored.append((field, float(s)))
            except ValueError:
                pass
        # Silence-vs-presence: heuristic — narrative mentions "No relevant information" or nil-marker
        if any(m in external.lower() for m in (
            "no relevant information", "not provided", "[more information needed]",
            "noassertion",
        )):
            silence_vs_presence.append((field, external))
    return {
        "flagged": flagged,
        "n_flagged": len(flagged),
        "nli_scored": nli_scored,
        "silence_vs_presence": silence_vs_presence,
        "n_silence_vs_presence": len(silence_vs_presence),
    }


def get_package_version(bom: dict) -> Tuple[Optional[str], Optional[str]]:
    pv = (bom.get("direct_fields", {}) or {}).get("packageVersion") or {}
    if not isinstance(pv, dict):
        return None, None
    return pv.get("value"), pv.get("source")


def count_rag_populated(bom: dict) -> int:
    rag = bom.get("rag_fields", {}) or {}
    n = 0
    for k, v in rag.items():
        val = v.get("value") if isinstance(v, dict) else v
        if not _is_nil(val):
            n += 1
    return n


def per_bom_row(slug: str, kind: str) -> Optional[dict]:
    sub = "ai" if kind == "ai" else "data"
    base = load(BASELINE / sub / f"{slug}.json")
    new = load(NEW / sub / f"{slug}.json")
    if base is None or new is None:
        return {"slug": slug, "kind": kind, "status": "missing",
                "baseline_exists": base is not None, "new_exists": new is not None}

    base_direct = count_direct_conflicts(base)
    new_direct = count_direct_conflicts(new)
    base_rag = count_rag_external_conflicts(base)
    new_rag = count_rag_external_conflicts(new)
    base_pkg = get_package_version(base)
    new_pkg = get_package_version(new)

    return {
        "slug": slug,
        "kind": kind,
        "status": "ok",
        "direct_conflicts_before": base_direct["n_conflicts"],
        "direct_conflicts_after": new_direct["n_conflicts"],
        "direct_conflicts_before_fields": base_direct["flagged_conflicts"],
        "direct_conflicts_after_fields": new_direct["flagged_conflicts"],
        "alternates_after_fields": new_direct["fields_with_alternates"],
        "rag_external_before": base_rag["n_flagged"],
        "rag_external_after": new_rag["n_flagged"],
        "rag_silence_vs_presence_before": base_rag["n_silence_vs_presence"],
        "rag_silence_vs_presence_after": new_rag["n_silence_vs_presence"],
        "rag_nli_scores": new_rag["nli_scored"],
        "rag_populated_before": count_rag_populated(base),
        "rag_populated_after": count_rag_populated(new),
        "package_version_before": base_pkg,
        "package_version_after": new_pkg,
    }


def section_header(t: str) -> str:
    return f"\n## {t}\n"


def fmt_table(headers: List[str], rows: List[List[str]]) -> str:
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join("---" for _ in headers) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out


def main() -> int:
    all_rows = []
    for slug in AI_SLUGS:
        all_rows.append(per_bom_row(slug, "ai"))
    for slug in DATA_SLUGS:
        all_rows.append(per_bom_row(slug, "data"))

    md = []
    md.append("# Phase 12 Conflict-Detection Rebuild — Results\n")
    md.append("Generated by `scripts/analyse_phase12_rerun.py` from `test_outputs/golden_baseline/` (pre-Phase-12) and `test_outputs/phase12_rerun/` (post-Phase-12). Diffs are attributed to findings #34, #35, and #37 from `test_outputs/golden_baseline/IMPROVEMENT_REPORT.md`.\n")

    # ----- Coverage -----
    missing = [r for r in all_rows if r["status"] == "missing"]
    md.append(section_header("Coverage"))
    md.append(f"- BOMs analysed: {len([r for r in all_rows if r['status'] == 'ok'])} / {len(all_rows)}")
    if missing:
        md.append("- **Missing pairs:**")
        for r in missing:
            md.append(f"  - `{r['kind']}/{r['slug']}` — baseline={r['baseline_exists']}, new={r['new_exists']}")

    ok_rows = [r for r in all_rows if r["status"] == "ok"]
    if not ok_rows:
        md.append("\n_No analysable pairs yet — rerun in progress?_\n")
        OUTPUT.write_text("\n".join(md))
        return 0

    # ----- Finding #34: per-platform schema + SAIL + alias index -----
    md.append(section_header("Finding #34 — per-platform direct-field schema, SAIL packageVersion, alias index"))
    md.append("\n### Direct-field conflict count\n")
    direct_before = sum(r["direct_conflicts_before"] for r in ok_rows)
    direct_after = sum(r["direct_conflicts_after"] for r in ok_rows)
    delta = direct_after - direct_before
    md.append(f"- **Total direct-field conflicts:** {direct_before} (baseline) → {direct_after} (Phase 12). Δ = {delta:+d}.")
    md.append("- Cross-platform structural conflicts (`releaseTime`, `downloadLocation`, `contentIdentifier`, `builtTime`) should no longer fire; remaining direct conflicts should be artefact-kind disagreements (`license`, `suppliedBy` after alias resolution).\n")
    md.append(fmt_table(
        ["BOM", "Before", "After", "Before fields", "After fields", "Alternates fields"],
        [
            [
                f"{r['kind']}/{r['slug']}",
                r["direct_conflicts_before"],
                r["direct_conflicts_after"],
                ", ".join(r["direct_conflicts_before_fields"]) or "—",
                ", ".join(r["direct_conflicts_after_fields"]) or "—",
                ", ".join(r["alternates_after_fields"]) or "—",
            ]
            for r in ok_rows
        ],
    ))

    md.append("\n### SAIL packageVersion adoption (AI BOMs only)\n")
    md.append(fmt_table(
        ["BOM", "Before value", "Before source", "After value", "After source"],
        [
            [
                r["slug"],
                (r["package_version_before"] or (None, None))[0] or "—",
                (r["package_version_before"] or (None, None))[1] or "—",
                (r["package_version_after"] or (None, None))[0] or "—",
                (r["package_version_after"] or (None, None))[1] or "—",
            ]
            for r in ok_rows if r["kind"] == "ai"
        ],
    ))
    md.append("Target: AI BOM rows show `_source = huggingface_name` and a human-readable `<size>-<version|variant>` value (e.g., `7B-instruction-tuned`). Models with no name-extractable signal (`bert-base-uncased`) correctly fall through to the existing cascade.\n")

    # ----- Finding #35: NLI prose conflict -----
    md.append(section_header("Finding #35 — NLI prose conflict detection"))
    md.append("\n### RAG external-conflict counts\n")
    rag_before = sum(r["rag_external_before"] for r in ok_rows)
    rag_after = sum(r["rag_external_after"] for r in ok_rows)
    silence_before = sum(r["rag_silence_vs_presence_before"] for r in ok_rows)
    silence_after = sum(r["rag_silence_vs_presence_after"] for r in ok_rows)
    md.append(f"- **RAG external conflicts:** {rag_before} → {rag_after} (Δ {rag_after - rag_before:+d}).")
    md.append(f"- **Silence-vs-presence false positives:** {silence_before} → {silence_after} (target: 0).")
    md.append(fmt_table(
        ["BOM", "Before", "After", "Silence-vs-presence before", "Silence-vs-presence after"],
        [
            [
                f"{r['kind']}/{r['slug']}",
                r["rag_external_before"],
                r["rag_external_after"],
                r["rag_silence_vs_presence_before"],
                r["rag_silence_vs_presence_after"],
            ]
            for r in ok_rows
        ],
    ))

    md.append("\n### Contradiction-score distribution (NLI, Phase 12 only)\n")
    all_scores: List[float] = []
    for r in ok_rows:
        for _f, s in r["rag_nli_scores"]:
            all_scores.append(s)
    if all_scores:
        all_scores.sort()
        md.append(f"- N = {len(all_scores)} NLI-flagged contradiction events across the 10 BOMs.")
        md.append(f"- Min = {min(all_scores):.2f} | Median = {all_scores[len(all_scores)//2]:.2f} | Max = {max(all_scores):.2f}")
        # Bucketed
        buckets = defaultdict(int)
        for s in all_scores:
            if s >= 0.95: buckets[">= 0.95"] += 1
            elif s >= 0.80: buckets["0.80–0.95"] += 1
            elif s >= 0.50: buckets["0.50–0.80"] += 1
        md.append(f"- Buckets: high (≥0.95) = {buckets['>= 0.95']}, mid (0.80–0.95) = {buckets['0.80–0.95']}, low (0.50–0.80) = {buckets['0.50–0.80']}.")
    else:
        md.append("- No NLI conflict events recorded across the 10 BOMs.")

    # ----- Finding #37: dataset RAG recall (not addressed by Phase 12) -----
    md.append(section_header("Finding #37 — dataset RAG recall (not addressed by Phase 12)"))
    md.append("- This experiment did not touch the dataset HyDE prompts or arxiv chunking strategy; dataset RAG-pop should be roughly unchanged. Any movement here is incidental.\n")
    md.append(fmt_table(
        ["BOM", "RAG populated before", "RAG populated after", "Δ"],
        [
            [f"{r['kind']}/{r['slug']}", r["rag_populated_before"], r["rag_populated_after"],
             r["rag_populated_after"] - r["rag_populated_before"]]
            for r in ok_rows if r["kind"] == "data"
        ],
    ))

    md.append(section_header("AI BOM RAG-pop sanity check (Phase 12 shouldn't regress this)"))
    md.append(fmt_table(
        ["BOM", "RAG populated before", "RAG populated after", "Δ"],
        [
            [f"{r['kind']}/{r['slug']}", r["rag_populated_before"], r["rag_populated_after"],
             r["rag_populated_after"] - r["rag_populated_before"]]
            for r in ok_rows if r["kind"] == "ai"
        ],
    ))

    OUTPUT.write_text("\n".join(md))
    print(f"Wrote {OUTPUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
