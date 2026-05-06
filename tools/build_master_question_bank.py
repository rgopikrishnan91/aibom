#!/usr/bin/env python3
"""Build a single consolidated question-bank JSON.

Walks ``src/aikaboom/question_bank/{ai,data}/*.json``, layers in the
priority for each field (from ``aikaboom.utils.source_priority``), and
where the field maps to a published SPDX 3.0.1 property, attaches the
spec URL from ``docs/SPDX_3.0.1_FIELD_REFERENCE.json``.

Output: ``docs/question_bank.json``.

Run after editing any per-field JSON, the source-priority config, or
re-syncing SPDX descriptions:

    python tools/build_master_question_bank.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parent.parent
QB_ROOT = REPO_ROOT / "src" / "aikaboom" / "question_bank"
SPDX_INDEX = REPO_ROOT / "docs" / "SPDX_3.0.1_FIELD_REFERENCE.json"
OUT_PATH = REPO_ROOT / "docs" / "question_bank.json"


def _load_spdx_index() -> Dict[str, Any]:
    if not SPDX_INDEX.exists():
        return {"properties": {}, "aikaboom_field_to_spdx": {"ai": {}, "data": {}}}
    return json.loads(SPDX_INDEX.read_text(encoding="utf-8"))


def _entry_for(path: Path, bom_type: str, spdx_index: Dict[str, Any]) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    field = path.stem
    spdx_name = spdx_index.get("aikaboom_field_to_spdx", {}).get(bom_type, {}).get(field)
    spdx_url = None
    if spdx_name:
        prop = spdx_index.get("properties", {}).get(spdx_name) or {}
        spdx_url = prop.get("spec_url")

    # Layer in priority via the live config so the master file matches
    # what the loader would produce at module import.
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from aikaboom.utils.source_priority import get_rag_priority
        priority = get_rag_priority(field, bom_type=bom_type)
    except Exception:
        priority = []

    out: Dict[str, Any] = {
        "field": field,
        "bom_type": bom_type,
        "aikaboom_internal": bool(raw.get("aikaboom_internal", False)),
        "spdx_property": spdx_name,
        "spec_url": spdx_url,
        "question": raw.get("question", ""),
        "keywords": raw.get("keywords", ""),
        "summary": raw.get("summary", ""),
        "description": raw.get("description", ""),
        "retrieval": raw.get("retrieval", {}),
        "extraction": raw.get("extraction", {}),
        "post_process": raw.get("post_process"),
        "priority": priority,
    }
    return out


def main() -> int:
    spdx_index = _load_spdx_index()
    fields: Dict[str, Dict[str, Any]] = {"ai": {}, "data": {}}
    counts = {"ai_mapped": 0, "ai_internal": 0, "data_mapped": 0, "data_internal": 0}

    for bom_type in ("ai", "data"):
        folder = QB_ROOT / bom_type
        for path in sorted(folder.glob("*.json")):
            entry = _entry_for(path, bom_type, spdx_index)
            fields[bom_type][entry["field"]] = entry
            bucket = "internal" if entry["aikaboom_internal"] else "mapped"
            counts[f"{bom_type}_{bucket}"] += 1

    payload = {
        "generated_by": "tools/build_master_question_bank.py",
        "spdx_version": spdx_index.get("spdx_version", "3.0.1"),
        "field_count": {
            "ai": len(fields["ai"]),
            "data": len(fields["data"]),
            "total": len(fields["ai"]) + len(fields["data"]),
            **counts,
        },
        "schema": {
            "field":              "AIkaBoOM field name (filename stem of the per-field JSON)",
            "bom_type":           "'ai' or 'data'",
            "aikaboom_internal":  "true if the field has no SPDX 3.0.1 property page; false if it maps to one",
            "spdx_property":      "full SPDX property name (e.g. 'ai_typeOfModel') or null for internal fields",
            "spec_url":           "SPDX 3.0.1 spec page URL or null for internal fields",
            "question":           "legacy prompt text (audit-only; no longer used at runtime)",
            "keywords":           "legacy retrieval-only tokens (audit-only; superseded by retrieval.bm25_terms)",
            "summary":            "verbatim SPDX 3.0.1 Summary block (mapped fields only; '' for internal)",
            "description":        "verbatim SPDX 3.0.1 Description block (mapped) or hand-written text (internal)",
            "retrieval":          "{'hypothetical_passage': HyDE prose for FAISS dense retrieval (Gao et al. ACL 2023), 'bm25_terms': list of exact-match strings for BM25 sparse retrieval}",
            "extraction":         "{'instruction': imperative for the LLM, 'field_spec': SPDX/internal contract + legal values, 'output_guidance': edge-case decision rules}",
            "post_process":       "name of a post-processor in aikaboom.utils.post_process or null",
            "priority":           "ordered list of source names used by the agentic RAG router",
        },
        "fields": fields,
    }

    OUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[build] wrote {OUT_PATH.relative_to(REPO_ROOT)}: "
        f"{counts['ai_mapped']} AI mapped + {counts['ai_internal']} AI internal, "
        f"{counts['data_mapped']} data mapped + {counts['data_internal']} data internal",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
