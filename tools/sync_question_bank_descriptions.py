#!/usr/bin/env python3
"""Sync the question-bank `description` slot to the SPDX 3.0.1 reference.

Reads ``docs/SPDX_<version>_FIELD_REFERENCE.json`` (produced by
``tools/harvest_spdx_3_0_1.py``) and proposes a new description for
each ``src/aikaboom/question_bank/<bom_type>/<field>.json``.

Default: writes proposed values to ``<field>.json.proposed`` next to
the original so a human can diff before accepting. Pass ``--apply`` to
overwrite the originals in place (use after diffing).

For AIkaBoOM-internal fields (``trainedOnDatasets`` etc.) the script
sets ``"aikaboom_internal": true`` and leaves the description untouched.

Usage:
    python tools/sync_question_bank_descriptions.py
    python tools/sync_question_bank_descriptions.py --apply
    python tools/sync_question_bank_descriptions.py --version 3.0.1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
QB_ROOT = REPO_ROOT / "src" / "aikaboom" / "question_bank"


def _spdx_description(prop: dict) -> str:
    """Concatenate Summary + Description verbatim per the design.
    The spec text is canonical; no truncation, no length cap."""
    summary = (prop.get("summary") or "").strip()
    description = (prop.get("description") or "").strip()
    if summary and description:
        return f"{summary}\n\n{description}"
    return summary or description


def _load_index(version: str) -> dict:
    idx_path = REPO_ROOT / "docs" / f"SPDX_{version}_FIELD_REFERENCE.json"
    if not idx_path.exists():
        sys.exit(
            f"[sync] FATAL: {idx_path} not found. Run "
            f"`python tools/harvest_spdx_3_0_1.py --version {version}` first."
        )
    return json.loads(idx_path.read_text(encoding="utf-8"))


def _process_entry(
    bom_type: str,
    field: str,
    entry_path: Path,
    aik_to_spdx: Dict[str, Optional[str]],
    properties: dict,
) -> Optional[dict]:
    """Return the new entry dict, or ``None`` if no change is needed."""
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    spdx_name = aik_to_spdx.get(field)

    if spdx_name is None:
        # AIkaBoOM-internal: mark and bail.
        if entry.get("aikaboom_internal") is True:
            return None
        entry["aikaboom_internal"] = True
        return entry

    prop = properties.get(spdx_name)
    if prop is None:
        sys.exit(
            f"[sync] FATAL: {bom_type}/{field}.json maps to SPDX property "
            f"'{spdx_name}' but no such property in the reference index. "
            f"Either fix the mapping in tools/harvest_spdx_3_0_1.py or "
            f"re-run the harvester."
        )

    new_description = _spdx_description(prop)
    if entry.get("description") == new_description and not entry.get("aikaboom_internal"):
        return None  # already in sync

    entry["description"] = new_description
    # Make sure we don't carry a stale aikaboom_internal flag.
    entry.pop("aikaboom_internal", None)
    return entry


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default="3.0.1", help="SPDX 3.0.x release tag")
    parser.add_argument(
        "--apply", action="store_true",
        help="overwrite the original JSON files (default: write *.proposed sidecars)",
    )
    args = parser.parse_args(argv)

    index = _load_index(args.version)
    properties = index["properties"]
    mappings = index["aikaboom_field_to_spdx"]

    changed = 0
    untouched = 0
    proposed_paths: list = []
    for bom_type in ("ai", "data"):
        aik_to_spdx = mappings[bom_type]
        folder = QB_ROOT / bom_type
        for entry_path in sorted(folder.glob("*.json")):
            field = entry_path.stem
            new_entry = _process_entry(bom_type, field, entry_path, aik_to_spdx, properties)
            if new_entry is None:
                untouched += 1
                continue
            target = entry_path if args.apply else entry_path.with_suffix(".json.proposed")
            target.write_text(
                json.dumps(new_entry, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            changed += 1
            proposed_paths.append(target)

    print(f"[sync] {changed} entries changed, {untouched} unchanged", file=sys.stderr)
    if not args.apply and proposed_paths:
        print(
            f"[sync] {len(proposed_paths)} *.proposed files written; review with "
            "`git diff --no-index` and rename when accepted, or re-run with --apply.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
