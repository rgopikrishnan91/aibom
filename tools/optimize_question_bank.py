#!/usr/bin/env python3
"""Generate `retrieval` + `extraction` blocks for each question-bank field.

For every ``src/aikaboom/question_bank/<bom>/<field>.json`` this script:

1. Substitutes the field's metadata into ``optimize_question_bank.prompt.md``.
2. Sends the assembled prompt to Anthropic Claude Opus 4.7 (one call per
   field; concurrent across fields).
3. Parses the JSON response, validates structural completeness and length
   guardrails.
4. Layers the new ``retrieval`` and ``extraction`` blocks into the per-field
   JSON, preserving every existing key (Phase 1 SPDX provenance stays intact).

Output: 37 updated per-field JSONs + an audit snapshot at
``docs/optimized_question_bank.json``.

Idempotent at temperature=0: re-running with the same prompt produces the
same JSON, so a clean re-run yields an empty ``git diff``.

Usage::

    export ANTHROPIC_API_KEY=sk-ant-...
    python tools/optimize_question_bank.py
    python tools/optimize_question_bank.py --field autonomyType
    python tools/optimize_question_bank.py --bom-type data --dry-run
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
QB_ROOT = REPO_ROOT / "src" / "aikaboom" / "question_bank"
PROMPT_PATH = REPO_ROOT / "tools" / "optimize_question_bank.prompt.md"
SNAPSHOT_PATH = REPO_ROOT / "docs" / "optimized_question_bank.json"

MODEL = "claude-opus-4-7"
TEMPERATURE = 0.0
MAX_TOKENS = 4096
MAX_WORKERS = 4
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds


def _placeholders(entry: dict, field: str, bom_type: str) -> Dict[str, str]:
    """Variables substituted into the prompt template."""
    return {
        "field":              field,
        "bom_type":           bom_type,
        "aikaboom_internal":  str(bool(entry.get("aikaboom_internal", False))).lower(),
        "spdx_property":      entry.get("spdx_property") or "(none — AIkaBoOM-internal)",
        "spec_url":           entry.get("spec_url") or "(none)",
        "question":           entry.get("question", ""),
        "keywords":           entry.get("keywords", ""),
        "summary":            entry.get("summary", "") or "(none)",
        "description":        entry.get("description", "") or "(none)",
    }


def _render_prompt(template: str, vars: Dict[str, str]) -> str:
    """Replace ``{name}`` placeholders. Uses str.replace to dodge
    Python's str.format complaints about JSON braces in the template."""
    out = template
    for key, val in vars.items():
        out = out.replace("{" + key + "}", val)
    return out


def _strip_json_envelope(text: str) -> str:
    """Pull a JSON object out of an LLM response that may carry markdown
    fences or a tiny preamble."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    # First-{ to last-} as a fallback
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        return text[first : last + 1]
    return text


def _validate(payload: dict, field: str, bom_type: str) -> List[str]:
    """Return a list of validation errors; empty list means valid."""
    errors: List[str] = []
    if not isinstance(payload, dict):
        return [f"{bom_type}/{field}: response is not a JSON object"]

    retrieval = payload.get("retrieval")
    if not isinstance(retrieval, dict):
        errors.append(f"{bom_type}/{field}: missing 'retrieval' object")
    else:
        passage = retrieval.get("hypothetical_passage")
        if not isinstance(passage, str) or not passage.strip():
            errors.append(f"{bom_type}/{field}: retrieval.hypothetical_passage missing/empty")
        else:
            # Real-tokenizer (BGE WordPiece) → tiktoken → conservative fallback
            try:
                from aikaboom.utils.token_count import count_tokens
                n_tokens = count_tokens(passage)
            except Exception:
                n_tokens = int(len(passage) / 3.5)
            if n_tokens > 200:
                errors.append(
                    f"{bom_type}/{field}: hypothetical_passage {n_tokens} tokens (>200 cap)"
                )
        terms = retrieval.get("bm25_terms")
        if not isinstance(terms, list) or len(terms) < 5:
            errors.append(
                f"{bom_type}/{field}: bm25_terms must be a list of ≥5 strings (got {type(terms).__name__})"
            )
        elif not all(isinstance(t, str) and t.strip() for t in terms):
            errors.append(f"{bom_type}/{field}: every bm25_term must be a non-empty string")

    extraction = payload.get("extraction")
    if not isinstance(extraction, dict):
        errors.append(f"{bom_type}/{field}: missing 'extraction' object")
    else:
        for key in ("instruction", "field_spec", "output_guidance"):
            val = extraction.get(key)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"{bom_type}/{field}: extraction.{key} missing/empty")

    return errors


def _call_anthropic(prompt: str, client: Any) -> str:
    """Single Opus call with bounded retries on transient errors."""
    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            # The SDK returns a Message; concatenate any text blocks.
            parts = []
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
            return "".join(parts) if parts else str(resp.content)
        except Exception as exc:  # noqa: BLE001 — retry-then-raise is intentional
            last_err = exc
            if attempt + 1 == MAX_RETRIES:
                raise
            time.sleep(RETRY_BASE_DELAY * (2**attempt))
    if last_err:
        raise last_err
    raise RuntimeError("anthropic call exhausted retries with no exception captured")


def _process_field(
    bom_type: str,
    field: str,
    entry: dict,
    template: str,
    client: Any,
) -> Tuple[str, str, Optional[dict], List[str]]:
    """Worker: render prompt, call Opus, validate. Returns
    (bom_type, field, payload_or_None, errors)."""
    vars = _placeholders(entry, field, bom_type)
    prompt = _render_prompt(template, vars)
    try:
        raw = _call_anthropic(prompt, client)
    except Exception as exc:  # noqa: BLE001
        return bom_type, field, None, [f"{bom_type}/{field}: API call failed — {exc!r}"]

    cleaned = _strip_json_envelope(raw)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return bom_type, field, None, [
            f"{bom_type}/{field}: response was not valid JSON — {exc}",
            f"  raw response (first 500 chars): {raw[:500]!r}",
        ]

    errors = _validate(payload, field, bom_type)
    if errors:
        return bom_type, field, None, errors
    return bom_type, field, payload, []


def _load_master_entry(field_path: Path) -> dict:
    """Per-field source JSON includes the SPDX-property/url info layered
    in by build_master_question_bank — but here we work from the
    per-field source-of-truth, so we read it directly. To get
    spdx_property and spec_url at prompt-render time, we cross-reference
    docs/SPDX_3.0.1_FIELD_REFERENCE.json."""
    return json.loads(field_path.read_text(encoding="utf-8"))


def _spdx_lookup() -> Dict[str, Dict[str, Optional[str]]]:
    """Load the Phase 1 SPDX index so we can attach spec_url / spdx_property
    to per-field entries at prompt-render time."""
    idx_path = REPO_ROOT / "docs" / "SPDX_3.0.1_FIELD_REFERENCE.json"
    if not idx_path.exists():
        return {"ai": {}, "data": {}}
    raw = json.loads(idx_path.read_text(encoding="utf-8"))
    field_to_spdx = raw.get("aikaboom_field_to_spdx", {})
    properties = raw.get("properties", {})
    out: Dict[str, Dict[str, Optional[str]]] = {"ai": {}, "data": {}}
    for bom_type in ("ai", "data"):
        for field, spdx_name in (field_to_spdx.get(bom_type) or {}).items():
            if spdx_name:
                prop = properties.get(spdx_name) or {}
                out[bom_type][field] = {
                    "spdx_property": spdx_name,
                    "spec_url":      prop.get("spec_url"),
                }
            else:
                out[bom_type][field] = {"spdx_property": None, "spec_url": None}
    return out


def _enriched_entry(raw: dict, bom_type: str, field: str, spdx_index: dict) -> dict:
    """Merge SPDX cross-ref into the raw per-field entry for prompt rendering."""
    enriched = dict(raw)
    cross = spdx_index.get(bom_type, {}).get(field, {})
    enriched["spdx_property"] = cross.get("spdx_property")
    enriched["spec_url"] = cross.get("spec_url")
    return enriched


def _walk_targets(args: argparse.Namespace) -> Iterable[Tuple[str, str, Path]]:
    bom_types = [args.bom_type] if args.bom_type else ["ai", "data"]
    for bom in bom_types:
        for path in sorted((QB_ROOT / bom).glob("*.json")):
            field = path.stem
            if args.field and field != args.field:
                continue
            yield bom, field, path


def _write_back(path: Path, raw: dict, payload: dict, dry_run: bool) -> Path:
    """Layer retrieval + extraction into the per-field JSON, preserving
    every existing key. Field order: existing keys first, then retrieval
    + extraction at the end."""
    updated = dict(raw)  # preserve insertion order
    updated["retrieval"] = payload["retrieval"]
    updated["extraction"] = payload["extraction"]
    target = path if not dry_run else path.with_suffix(".json.proposed")
    target.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--field", help="optimise a single field by name")
    parser.add_argument("--bom-type", choices=("ai", "data"), help="restrict to one profile")
    parser.add_argument("--dry-run", action="store_true", help="write *.proposed sidecars instead of overwriting")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="concurrent Opus calls")
    args = parser.parse_args(argv)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "[optimize] FATAL: ANTHROPIC_API_KEY is not set. Export your key:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-...\n"
            "If you don't have a key, the produced JSONs are committed; this script\n"
            "is only needed to refresh them after the prompt template changes."
        )

    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError:
        sys.exit(
            "[optimize] FATAL: the `anthropic` SDK is not installed. Install with:\n"
            "    pip install anthropic"
        )

    template = PROMPT_PATH.read_text(encoding="utf-8")
    spdx_index = _spdx_lookup()
    targets = list(_walk_targets(args))
    if not targets:
        sys.exit(f"[optimize] no fields matched (--field {args.field!r}, --bom-type {args.bom_type!r})")

    print(f"[optimize] {len(targets)} field(s); {args.workers} workers; model={MODEL}", file=sys.stderr)
    client = anthropic.Anthropic(api_key=api_key)

    successes: Dict[Tuple[str, str], dict] = {}
    failures: List[List[str]] = []
    raw_entries: Dict[Tuple[str, str], dict] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for bom_type, field, path in targets:
            raw = _load_master_entry(path)
            enriched = _enriched_entry(raw, bom_type, field, spdx_index)
            raw_entries[(bom_type, field)] = raw
            futures.append(
                pool.submit(_process_field, bom_type, field, enriched, template, client)
            )

        for fut in concurrent.futures.as_completed(futures):
            bom_type, field, payload, errors = fut.result()
            if errors or payload is None:
                failures.append(errors)
                print(f"  ✗ {bom_type}/{field}", file=sys.stderr)
                for err in errors:
                    print(f"      {err}", file=sys.stderr)
            else:
                successes[(bom_type, field)] = payload
                print(f"  ✓ {bom_type}/{field}", file=sys.stderr)

    if failures:
        print(
            f"\n[optimize] {len(failures)} field(s) failed; "
            f"{len(successes)} succeeded. Re-run failed fields with --field <name>.",
            file=sys.stderr,
        )

    if not successes:
        return 1

    # Write per-field JSONs
    for (bom_type, field), payload in successes.items():
        path = QB_ROOT / bom_type / f"{field}.json"
        target = _write_back(path, raw_entries[(bom_type, field)], payload, args.dry_run)
        print(f"[optimize] wrote {target.relative_to(REPO_ROOT)}", file=sys.stderr)

    # Audit snapshot
    snapshot: Dict[str, Any] = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "fields": {"ai": {}, "data": {}},
    }
    for (bom_type, field), payload in successes.items():
        snapshot["fields"][bom_type][field] = payload
    SNAPSHOT_PATH.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[optimize] wrote {SNAPSHOT_PATH.relative_to(REPO_ROOT)}", file=sys.stderr)

    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
