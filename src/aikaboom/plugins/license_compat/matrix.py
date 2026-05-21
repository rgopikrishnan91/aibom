"""License matrix loader + canonical-name resolver.

Lifted from LicenseRec.py with the I/O isolated so the engine layer can stay
pure. The matrix is a dict of license entries keyed by canonical primary name;
each entry has aliases, a category, and a list of pairwise compatibility
verdicts against every other license. The upstream_compat_index is the inverse
view: for each upstream license, which downstream licenses can legally consume it.
"""
from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Optional

_OTHER_SUFFIX_RE = re.compile(r"\s*\(\s*other\s*\)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class LicenseMatrix:
    name_alias_lookup: dict[str, str]
    details: dict[str, dict]
    upstream_compat_index: dict[str, frozenset[str]]
    allowed_licenses: frozenset[str]
    missing_licenses: frozenset[str]
    timestamp: Optional[str] = None


@dataclass(frozen=True)
class ResolvedLicense:
    primary_name: Optional[str]
    is_unknown: bool
    is_missing: bool


def _bundled(name: str) -> Path:
    return Path(str(files("aikaboom.plugins.license_compat.data").joinpath(name)))


def _clean(raw: str) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    stripped = raw.strip()
    if not stripped or stripped.upper() == "MISSING":
        return None
    return _OTHER_SUFFIX_RE.sub("", stripped).strip().lower()


def load_matrix(
    matrix_path: Optional[Path] = None,
    allowed_path: Optional[Path] = None,
    missing_path: Optional[Path] = None,
) -> LicenseMatrix:
    matrix_path = Path(matrix_path) if matrix_path else _bundled("matrix.json")
    allowed_path = Path(allowed_path) if allowed_path else _bundled("allowed_licenses.json")
    missing_path = Path(missing_path) if missing_path else _bundled("missing.json")

    matrix_data = json.loads(matrix_path.read_text(encoding="utf-8"))
    timestamp = matrix_data.get("timestamp")

    name_alias_lookup: dict[str, str] = {}
    details: dict[str, dict] = {}

    for item in matrix_data.get("licenses", []):
        if not isinstance(item, dict):
            continue
        primary = item.get("name")
        if not isinstance(primary, str):
            continue
        primary = primary.strip()
        item = dict(item)
        item["category"] = item.get("category") or "UNKNOWN"
        details[primary] = item
        name_alias_lookup[primary.lower()] = primary
        for alias in item.get("aliases", []) or []:
            if isinstance(alias, str):
                a = alias.lower().strip()
                if a:
                    name_alias_lookup[a] = primary

    if "UNKNOWN" not in details:
        details["UNKNOWN"] = {"name": "UNKNOWN", "category": "UNKNOWN", "permissions": [], "compatibilities": []}
    name_alias_lookup["unknown"] = "UNKNOWN"

    upstream_compat_index: dict[str, set[str]] = defaultdict(set)
    for downstream, info in details.items():
        for entry in info.get("compatibilities", []) or []:
            if isinstance(entry, dict) and entry.get("compatibility") in ("Yes", "Same"):
                up = entry.get("name")
                if isinstance(up, str):
                    upstream_compat_index[up].add(downstream)

    allowed_raw = json.loads(allowed_path.read_text(encoding="utf-8")) if allowed_path.exists() else []
    allowed = frozenset(
        item["key"].lower().strip()
        for item in allowed_raw
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    )

    missing_raw = json.loads(missing_path.read_text(encoding="utf-8")) if missing_path.exists() else {}
    missing = frozenset(
        lic.strip().lower() for lic in (missing_raw.get("licenses", []) if isinstance(missing_raw, dict) else [])
        if isinstance(lic, str)
    )

    return LicenseMatrix(
        name_alias_lookup=name_alias_lookup,
        details=details,
        upstream_compat_index={k: frozenset(v) for k, v in upstream_compat_index.items()},
        allowed_licenses=allowed,
        missing_licenses=missing,
        timestamp=timestamp,
    )


def resolve_license(raw: str, matrix: LicenseMatrix) -> ResolvedLicense:
    if not isinstance(raw, str):
        return ResolvedLicense(primary_name=None, is_unknown=False, is_missing=False)
    if raw.strip().upper() == "MISSING":
        return ResolvedLicense(primary_name=None, is_unknown=False, is_missing=True)
    cleaned = _clean(raw)
    if cleaned is None:
        return ResolvedLicense(primary_name=None, is_unknown=False, is_missing=False)
    primary = matrix.name_alias_lookup.get(cleaned)
    if primary is None:
        return ResolvedLicense(primary_name="UNKNOWN", is_unknown=True, is_missing=False)
    return ResolvedLicense(primary_name=primary, is_unknown=False, is_missing=False)


def normalize_license_field(value) -> list[str]:
    """Mirrors LicenseRec.py: accept a string, list, or stringified list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if isinstance(v, str)]
    if isinstance(value, str):
        s = value.strip()
        if not s or s.upper() == "MISSING":
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if isinstance(v, str)]
            except (ValueError, SyntaxError):
                return [s]
        return [s]
    return []
