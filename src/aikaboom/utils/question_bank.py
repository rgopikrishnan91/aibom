"""Question-bank loader.

The RAG question bank is a folder of one-JSON-per-field files under
``aikaboom/question_bank/<bom_type>/<field>.json``. Each file carries
the question prompt, keywords for retrieval, the SPDX-citing
description, and an optional ``post_process`` callable name. The
priority list lives separately in
``aikaboom/config/source_priority.json`` so prompt edits and ranking
edits don't collide in PR review.

This module reads every JSON file under each BOM-type folder once, then
overlays the priority from the source-priority config so the runtime
question bank looks identical to the legacy inline dicts.

Edit ``aikaboom/question_bank/<bom_type>/<field>.json`` to tune any
field's prompt, keywords, or description without touching Python.
"""
from __future__ import annotations

import json
import sys
from importlib import resources
from typing import Any, Dict, List


_PACKAGE = "aikaboom.question_bank"
_REQUIRED_KEYS = {"question", "keywords", "description"}


def _entry_files(bom_type: str) -> List[Any]:
    """Return Traversable references for every JSON file in the BOM-type
    folder. Returns an empty list if the folder is missing or unreadable.
    """
    try:
        folder = resources.files(_PACKAGE).joinpath(bom_type)
    except (ModuleNotFoundError, FileNotFoundError):
        return []
    try:
        return [e for e in folder.iterdir() if e.name.endswith(".json")]
    except (FileNotFoundError, NotADirectoryError):
        return []


def _read_entry(ref: Any) -> Dict[str, Any]:
    """Load one JSON entry. Validates required keys and returns a sanitised
    dict. Raises ``ValueError`` on malformed entries so callers can choose
    to skip vs hard-fail."""
    with ref.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{ref.name}: top-level must be an object, got {type(data).__name__}")
    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"{ref.name}: missing required keys: {sorted(missing)}")
    return data


def load_question_bank(bom_type: str) -> Dict[str, Dict[str, Any]]:
    """Load every entry under ``question_bank/<bom_type>/`` keyed by field.

    Each returned entry has ``question``, ``keywords``, ``description``,
    optional ``post_process``, and a placeholder ``priority`` (empty
    list). Callers should overlay priorities from the source-priority
    config (see :func:`aikaboom.utils.source_priority.get_rag_priority`).
    """
    bom_type = bom_type.lower()
    bank: Dict[str, Dict[str, Any]] = {}
    for ref in _entry_files(bom_type):
        try:
            data = _read_entry(ref)
        except (ValueError, json.JSONDecodeError) as exc:
            print(
                f"[aikaboom] warning: skipping malformed question-bank entry "
                f"{ref.name}: {exc}",
                file=sys.stderr,
            )
            continue
        # Filename takes precedence over the field key inside the JSON
        # (the dispatch is the filename), but keep the inner field for
        # introspection / debugging.
        field = ref.name[:-5]  # drop ".json"
        bank[field] = {
            "question": data["question"],
            "keywords": data.get("keywords", ""),
            "summary": data.get("summary", ""),
            "description": data.get("description", ""),
            "retrieval": data.get("retrieval", {}),
            "extraction": data.get("extraction", {}),
            "post_process": data.get("post_process"),
            "priority": [],  # filled by overlay step
        }
    return bank


def composite_description(entry: Dict[str, Any]) -> str:
    """Return the SPDX Summary and Description blocks concatenated for
    runtime use. Kept for any caller that still wants the merged
    spec text (e.g. legacy LLM prompts, audit dumps). The HyDE-driven
    pipeline uses ``dense_query`` and ``sparse_query`` instead.
    """
    summary = (entry.get("summary") or "").strip()
    description = (entry.get("description") or "").strip()
    if summary and description:
        return f"{summary}\n\n{description}"
    return summary or description


def dense_query(entry: Dict[str, Any]) -> str:
    """Return the HyDE hypothetical passage for FAISS retrieval.
    Falls back to composite_description if the optimized block is
    missing (e.g. a field added before re-running the codegen)."""
    passage = ((entry.get("retrieval") or {}).get("hypothetical_passage") or "").strip()
    return passage or composite_description(entry)


def sparse_query(entry: Dict[str, Any]) -> str:
    """Return the BM25 query as a whitespace-joined string. Falls back
    to the legacy keywords field if the optimized block is missing."""
    terms = (entry.get("retrieval") or {}).get("bm25_terms") or []
    if isinstance(terms, list) and terms:
        return " ".join(str(t) for t in terms)
    return entry.get("keywords", "")


def extraction_prompt_parts(entry: Dict[str, Any]) -> Dict[str, str]:
    """Return the three-part extraction prompt slots
    ``{instruction, field_spec, output_guidance}``. Falls back to the
    composite description as ``field_spec`` and the existing ``question``
    as ``instruction`` if the optimized block is missing."""
    extraction = entry.get("extraction") or {}
    return {
        "instruction":     (extraction.get("instruction") or entry.get("question", "")).strip(),
        "field_spec":      (extraction.get("field_spec") or composite_description(entry)).strip(),
        "output_guidance": (extraction.get("output_guidance") or "").strip(),
    }


def overlay_priorities(bank: Dict[str, Dict[str, Any]], bom_type: str) -> None:
    """Mutate ``bank`` in place: overlay each entry's ``priority`` from
    the source-priority config so the question bank reflects the active
    config without callers having to remember a second step.
    """
    try:
        from aikaboom.utils.source_priority import get_rag_priority
    except Exception:
        return
    for field, entry in bank.items():
        entry["priority"] = get_rag_priority(field, bom_type=bom_type)


def load_with_priorities(bom_type: str) -> Dict[str, Dict[str, Any]]:
    """Convenience: read the JSON entries and overlay the priorities in
    one call. This is what :mod:`aikaboom.core.agentic_rag` uses to build
    the runtime ``FIXED_QUESTIONS_*`` dicts."""
    bank = load_question_bank(bom_type)
    overlay_priorities(bank, bom_type)
    return bank
