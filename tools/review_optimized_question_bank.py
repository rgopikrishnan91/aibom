#!/usr/bin/env python3
"""Side-by-side review printer for the optimized question bank.

Prints the legacy ``question`` / ``keywords`` / ``summary`` / ``description``
next to the new ``retrieval.*`` and ``extraction.*`` blocks for human
inspection. Useful after running ``tools/optimize_question_bank.py`` to
spot-check the codegen output.

Usage::

    # All fields
    python tools/review_optimized_question_bank.py

    # One field
    python tools/review_optimized_question_bank.py --field autonomyType

    # One bom_type
    python tools/review_optimized_question_bank.py --bom-type ai

    # Just the token-count audit
    python tools/review_optimized_question_bank.py --check-tokens
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
QB_ROOT = REPO_ROOT / "src" / "aikaboom" / "question_bank"


def _entries(args: argparse.Namespace) -> Iterable[Tuple[str, str, dict]]:
    bom_types = [args.bom_type] if args.bom_type else ["ai", "data"]
    for bom in bom_types:
        for path in sorted((QB_ROOT / bom).glob("*.json")):
            field = path.stem
            if args.field and field != args.field:
                continue
            yield bom, field, json.loads(path.read_text(encoding="utf-8"))


def _wrap(text: str, indent: str = "    ", width: int = 78) -> str:
    if not text:
        return f"{indent}(empty)"
    return "\n".join(
        textwrap.fill(line, width=width, initial_indent=indent, subsequent_indent=indent)
        for line in text.splitlines()
    )


def _est_tokens(text: str) -> int:
    try:
        from aikaboom.utils.token_count import count_tokens
        return count_tokens(text)
    except Exception:
        return int(len(text) / 3.5)


def _print_field(bom_type: str, field: str, entry: dict) -> None:
    print()
    print("=" * 80)
    print(f"  {bom_type}/{field}    (aikaboom_internal={entry.get('aikaboom_internal', False)})")
    print("=" * 80)

    print("\n  legacy question:")
    print(_wrap(entry.get("question", "")))

    print("\n  legacy keywords:")
    print(_wrap(entry.get("keywords", "")))

    print("\n  SPDX summary:")
    print(_wrap(entry.get("summary", "")))

    print("\n  SPDX description:")
    print(_wrap(entry.get("description", "")))

    retrieval = entry.get("retrieval") or {}
    print("\n  ---- NEW: retrieval ----")
    passage = retrieval.get("hypothetical_passage", "")
    print(f"  hypothetical_passage   ({len(passage)} chars, ~{_est_tokens(passage)} tokens):")
    print(_wrap(passage))
    terms = retrieval.get("bm25_terms", [])
    print(f"  bm25_terms             ({len(terms)} terms):")
    print(_wrap(", ".join(terms)))

    extraction = entry.get("extraction") or {}
    print("\n  ---- NEW: extraction ----")
    print("  instruction:")
    print(_wrap(extraction.get("instruction", "")))
    print("  field_spec:")
    print(_wrap(extraction.get("field_spec", "")))
    print("  output_guidance:")
    print(_wrap(extraction.get("output_guidance", "")))


def _print_token_audit(args: argparse.Namespace) -> None:
    rows = []
    for bom_type, field, entry in _entries(args):
        passage = (entry.get("retrieval") or {}).get("hypothetical_passage", "")
        terms = (entry.get("retrieval") or {}).get("bm25_terms", [])
        rows.append((_est_tokens(passage), len(passage), len(terms), bom_type, field))

    rows.sort(reverse=True)
    print(f"\n{'tokens':>7}  {'chars':>5}  {'terms':>5}  field")
    print("-" * 60)
    for est, chars, term_count, bom_type, field in rows:
        flag = "  ← long" if est > 200 else ""
        print(f"{est:>7}  {chars:>5}  {term_count:>5}  {bom_type}/{field}{flag}")
    if rows:
        print(f"\nmin/median/max tokens: {rows[-1][0]} / {rows[len(rows)//2][0]} / {rows[0][0]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--field", help="show only this field")
    parser.add_argument("--bom-type", choices=("ai", "data"), help="restrict to one profile")
    parser.add_argument("--check-tokens", action="store_true", help="print only the token-count audit")
    args = parser.parse_args()

    if args.check_tokens:
        _print_token_audit(args)
        return 0

    n = 0
    for bom_type, field, entry in _entries(args):
        _print_field(bom_type, field, entry)
        n += 1
    if not n:
        print("(no fields matched)", file=sys.stderr)
        return 1
    print(f"\n{n} field(s) printed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
