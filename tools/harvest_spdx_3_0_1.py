#!/usr/bin/env python3
"""Harvest SPDX 3.0.1 property descriptions into the AIkaBoOM reference doc.

Pulls the spdx/spdx-3-model release tarball at the requested version,
parses every Property markdown file under the AI / Dataset / Software /
Core / SimpleLicensing namespaces, and writes a reference document to
docs/SPDX_<version>_FIELD_REFERENCE.md.

The reference doc is the single citation source for question_bank
description text. tools/sync_question_bank_descriptions.py consumes it
to update the per-field JSON files; tests/test_question_bank_descriptions.py
locks the alignment.

Usage:
    python tools/harvest_spdx_3_0_1.py                  # default 3.0.1
    python tools/harvest_spdx_3_0_1.py --version 3.1.0
    python tools/harvest_spdx_3_0_1.py --version 3.0.1 --refresh

Hard-fails if the tarball is unreachable AND the cache is empty.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "aikaboom"

# SPDX namespaces we care about. Each tuple is (folder_under_model,
# subfolder, prefix). Properties live under "Properties/", classes under
# "Classes/". The prefix is what shows up in JSON-LD: ``ai_typeOfModel``.
NAMESPACES: Tuple[Tuple[str, str, str], ...] = (
    ("AI",              "Properties", "ai_"),
    ("Dataset",         "Properties", "dataset_"),
    ("Software",        "Properties", "software_"),
    ("Core",            "Properties", ""),
    ("SimpleLicensing", "Classes",    "simplelicensing_"),
)


# AIkaBoOM field name -> full SPDX property name (with namespace prefix).
# ``None`` marks AIkaBoOM-internal fields with no SPDX counterpart; those
# get a ``"aikaboom_internal": true`` flag in their JSON.
AI_FIELD_TO_SPDX: Dict[str, Optional[str]] = {
    "autonomyType":                       "ai_autonomyType",
    "domain":                             "ai_domain",
    "energyConsumption":                  "ai_energyConsumption",
    "hyperparameter":                     "ai_hyperparameter",
    "informationAboutApplication":        "ai_informationAboutApplication",
    "informationAboutTraining":           "ai_informationAboutTraining",
    "limitation":                         "ai_limitation",
    "metric":                             "ai_metric",
    "metricDecisionThreshold":            "ai_metricDecisionThreshold",
    "modelDataPreprocessing":             "ai_modelDataPreprocessing",
    "modelExplainability":                "ai_modelExplainability",
    "safetyRiskAssessment":               "ai_safetyRiskAssessment",
    "standardCompliance":                 "ai_standardCompliance",
    "typeOfModel":                        "ai_typeOfModel",
    "useSensitivePersonalInformation":    "ai_useSensitivePersonalInformation",
    "license":                            "simplelicensing_LicenseExpression",
    "primaryPurpose":                     "software_primaryPurpose",
    "trainedOnDatasets":                  None,  # AIkaBoOM-internal
    "testedOnDatasets":                   None,  # AIkaBoOM-internal
    "modelLineage":                       None,  # AIkaBoOM-internal
}

DATA_FIELD_TO_SPDX: Dict[str, Optional[str]] = {
    "anonymizationMethodUsed":            "dataset_anonymizationMethodUsed",
    "confidentialityLevel":               "dataset_confidentialityLevel",
    "dataCollectionProcess":              "dataset_dataCollectionProcess",
    "dataPreprocessing":                  "dataset_dataPreprocessing",
    "datasetAvailability":                "dataset_datasetAvailability",
    "datasetNoise":                       "dataset_datasetNoise",
    "datasetSize":                        "dataset_datasetSize",
    "datasetType":                        "dataset_datasetType",
    "datasetUpdateMechanism":             "dataset_datasetUpdateMechanism",
    "hasSensitivePersonalInformation":    "dataset_hasSensitivePersonalInformation",
    "intendedUse":                        "dataset_intendedUse",
    "knownBias":                          "dataset_knownBias",
    "sensorUsed":                         "dataset_sensor",  # spec drops "Used"
    "license":                            "simplelicensing_LicenseExpression",
    "primaryPurpose":                     "software_primaryPurpose",
    "description":                        "description",  # core property, no prefix
    "sourceInfo":                         None,  # AIkaBoOM-internal
}


# ---------------------------------------------------------------------------
# Tarball fetch + cache
# ---------------------------------------------------------------------------


def tarball_url(version: str) -> str:
    return f"https://github.com/spdx/spdx-3-model/archive/refs/tags/{version}.tar.gz"


def cache_dir(version: str, cache_root: Path) -> Path:
    return cache_root / f"spdx-{version}-model"


def fetch_and_extract(version: str, cache_root: Path, refresh: bool) -> Path:
    """Return the path to the extracted ``spdx-3-model-<version>`` folder.

    Caches the extraction under ``<cache_root>/spdx-<version>-model/``;
    on a cold cache, fetches the GitHub tarball. Hard-fails if both the
    cache miss and the network fetch fail.
    """
    target = cache_dir(version, cache_root)
    extracted_root = target / f"spdx-3-model-{version}"
    if extracted_root.is_dir() and not refresh:
        return extracted_root

    target.mkdir(parents=True, exist_ok=True)
    url = tarball_url(version)
    print(f"[harvest] fetching {url}", file=sys.stderr)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        if extracted_root.is_dir():
            print(
                f"[harvest] network error ({exc}); using cached extraction at {extracted_root}",
                file=sys.stderr,
            )
            return extracted_root
        sys.exit(
            f"[harvest] FATAL: could not fetch {url} ({exc}) and no cache at "
            f"{extracted_root}. Check network or pre-populate the cache."
        )

    print(f"[harvest] extracting to {target}", file=sys.stderr)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(target)

    if not extracted_root.is_dir():
        # Some GitHub tarballs use slightly different top-level names.
        candidates = [p for p in target.iterdir() if p.is_dir() and p.name.startswith("spdx-3-model")]
        if not candidates:
            sys.exit(f"[harvest] FATAL: extracted tree has no spdx-3-model-* root under {target}")
        extracted_root = candidates[0]
    return extracted_root


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------


_H1 = re.compile(r"^#\s+(\S.*)$", re.MULTILINE)
_H2 = re.compile(r"^##\s+(\S.*)$", re.MULTILINE)


def parse_property_md(text: str) -> Dict[str, str]:
    """Pull the property name + Summary + Description blocks out of a
    Property markdown file. Returns ``{"name", "summary", "description"}``;
    summary or description may be empty strings if a section is absent.
    """
    name_match = _H1.search(text)
    name = name_match.group(1).strip() if name_match else ""

    sections: Dict[str, str] = {}
    matches = list(_H2.finditer(text))
    for i, m in enumerate(matches):
        section_name = m.group(1).strip().lower()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections[section_name] = body

    return {
        "name": name,
        "summary": sections.get("summary", ""),
        "description": sections.get("description", ""),
    }


def harvest_namespace(
    spdx_root: Path,
    namespace_folder: str,
    subfolder: str,
    prefix: str,
) -> Iterable[Tuple[str, Dict[str, str]]]:
    """Yield ``(full_property_name, parsed_dict)`` for every md file in
    ``spdx_root/model/<namespace_folder>/<subfolder>/``."""
    folder = spdx_root / "model" / namespace_folder / subfolder
    if not folder.is_dir():
        return
    for md in sorted(folder.glob("*.md")):
        if md.name.startswith("_"):
            continue  # skip _property.md and similar
        text = md.read_text(encoding="utf-8")
        parsed = parse_property_md(text)
        if not parsed["name"]:
            continue
        full_name = f"{prefix}{parsed['name']}"
        parsed["spec_url"] = (
            f"https://spdx.github.io/spdx-spec/v3.0.1/model/{namespace_folder}/"
            f"{'Properties' if subfolder == 'Properties' else 'Classes'}/{parsed['name']}/"
        )
        parsed["source_path"] = str(md.relative_to(spdx_root))
        yield full_name, parsed


def harvest_all(spdx_root: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for ns_folder, subfolder, prefix in NAMESPACES:
        for full_name, parsed in harvest_namespace(spdx_root, ns_folder, subfolder, prefix):
            out[full_name] = parsed
    return out


# ---------------------------------------------------------------------------
# Reference doc emission
# ---------------------------------------------------------------------------


def aikaboom_field_index() -> Dict[str, list]:
    """Reverse index: SPDX property -> [(bom_type, aikaboom_field), ...]."""
    idx: Dict[str, list] = {}
    for bom_type, mapping in (("ai", AI_FIELD_TO_SPDX), ("data", DATA_FIELD_TO_SPDX)):
        for field, spdx_name in mapping.items():
            if spdx_name is None:
                continue
            idx.setdefault(spdx_name, []).append((bom_type, field))
    return idx


def emit_reference(
    properties: Dict[str, Dict[str, str]],
    version: str,
    out_path: Path,
) -> None:
    """Write the reference doc with one section per SPDX property."""
    aik_idx = aikaboom_field_index()

    lines: list = []
    lines.append(f"# SPDX {version} Field Reference\n")
    lines.append(
        "Auto-generated by `tools/harvest_spdx_3_0_1.py`. Re-run that script "
        "to refresh after a SPDX rev. The Summary and Description blocks below "
        "are quoted **verbatim** from the [`spdx/spdx-3-model`](https://github.com/spdx/spdx-3-model) "
        f"repository at tag `{version}` and are the canonical text used for "
        "question-bank descriptions in `src/aikaboom/question_bank/`. The "
        "`tests/test_question_bank_descriptions.py` regression test fails if "
        "any JSON description drifts from the text below.\n"
    )

    # AIkaBoOM-internal fields with no SPDX counterpart
    internal: list = []
    for bom_type, mapping in (("ai", AI_FIELD_TO_SPDX), ("data", DATA_FIELD_TO_SPDX)):
        for field, spdx_name in mapping.items():
            if spdx_name is None:
                internal.append(f"- `{bom_type}/{field}`")

    lines.append("## AIkaBoOM-internal fields (no SPDX 3.0.1 property page)\n")
    lines.append(
        "These fields exist only inside AIkaBoOM. They drive the recursive-BOM "
        "walker (`trainedOnDatasets`, `testedOnDatasets`, `modelLineage`) or "
        "aggregate multiple SPDX sources (`sourceInfo`). Their question-bank "
        "JSON entries carry `\"aikaboom_internal\": true` and are exempt from "
        "the spec-text regression check.\n"
    )
    for line in internal:
        lines.append(line)
    lines.append("")

    # AIkaBoOM-mapped fields, in their bundled order
    for spdx_name in sorted(properties.keys()):
        if spdx_name not in aik_idx:
            continue
        prop = properties[spdx_name]
        lines.append(f"## `{spdx_name}`\n")
        owners = ", ".join(f"`{bt}/{f}`" for bt, f in aik_idx[spdx_name])
        lines.append(f"**AIkaBoOM field(s):** {owners}\n")
        lines.append(f"**Spec URL:** {prop['spec_url']}\n")
        lines.append(f"**Source:** `{prop['source_path']}`\n")
        lines.append("### Summary\n")
        lines.append(prop["summary"] or "_(no Summary block)_")
        lines.append("")
        lines.append("### Description\n")
        lines.append(prop["description"] or "_(no Description block)_")
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[harvest] wrote {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Sidecar JSON (used by the sync script and the test)
# ---------------------------------------------------------------------------


def emit_machine_index(
    properties: Dict[str, Dict[str, str]],
    version: str,
    out_path: Path,
) -> None:
    """Write a machine-readable JSON sidecar so downstream scripts don't
    have to parse the human reference doc back."""
    aik_idx = aikaboom_field_index()
    payload = {
        "spdx_version": version,
        "aikaboom_field_to_spdx": {
            "ai": {k: v for k, v in AI_FIELD_TO_SPDX.items()},
            "data": {k: v for k, v in DATA_FIELD_TO_SPDX.items()},
        },
        "properties": {
            spdx_name: {
                "summary": prop["summary"],
                "description": prop["description"],
                "spec_url": prop["spec_url"],
                "source_path": prop["source_path"],
                "aikaboom_owners": aik_idx.get(spdx_name, []),
            }
            for spdx_name, prop in properties.items()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[harvest] wrote {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default="3.0.1", help="SPDX 3.0.x release tag (default: 3.0.1)")
    parser.add_argument("--refresh", action="store_true", help="ignore cached extraction and re-fetch")
    parser.add_argument(
        "--cache-root", type=Path, default=DEFAULT_CACHE_ROOT,
        help="cache directory (default: ~/.cache/aikaboom)",
    )
    parser.add_argument(
        "--reference-out", type=Path, default=None,
        help="output path for the human reference doc "
             "(default: docs/SPDX_<version>_FIELD_REFERENCE.md)",
    )
    parser.add_argument(
        "--index-out", type=Path, default=None,
        help="output path for the machine-readable JSON sidecar "
             "(default: docs/SPDX_<version>_FIELD_REFERENCE.json)",
    )
    args = parser.parse_args(argv)

    spdx_root = fetch_and_extract(args.version, args.cache_root, args.refresh)
    properties = harvest_all(spdx_root)
    if not properties:
        sys.exit(f"[harvest] FATAL: no properties parsed from {spdx_root}")

    ref_path = args.reference_out or REPO_ROOT / "docs" / f"SPDX_{args.version}_FIELD_REFERENCE.md"
    idx_path = args.index_out or REPO_ROOT / "docs" / f"SPDX_{args.version}_FIELD_REFERENCE.json"
    emit_reference(properties, args.version, ref_path)
    emit_machine_index(properties, args.version, idx_path)
    print(f"[harvest] done: {len(properties)} properties parsed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
