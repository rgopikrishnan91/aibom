"""Source-priority configuration loader.

When AIkaBoOM detects that two or more sources disagree on a field, it
still has to choose one value to surface in the triplet's ``value`` slot
and in the SPDX/CycloneDX exports. The ranking lives in a single JSON
config so the community can tune it without editing Python.

Resolution order for the loader:

1. An explicit path passed to :func:`load_source_priority` or registered
   via :func:`set_source_priority_path`.
2. The ``AIKABOOM_SOURCE_PRIORITY`` environment variable.
3. The bundled config at ``aikaboom/config/source_priority.json``.

A user config is merged onto the bundled defaults *per field*, so
overriding a single priority does not require copying the whole file. A
malformed user config logs a warning and falls back to the bundled
defaults.
"""
from __future__ import annotations

import json
import os
import sys
from importlib import resources
from typing import Any, Dict, List, Optional


_CONFIG_PACKAGE = "aikaboom.config"
_CONFIG_FILENAME = "source_priority.json"
_ENV_VAR = "AIKABOOM_SOURCE_PRIORITY"

# Hard-coded last-resort defaults. These are only used if both the user
# config and the bundled config are missing or unreadable — they exist so
# a corrupt install never breaks recursion or value selection.
_HARDCODED_FALLBACK: Dict[str, Dict[str, List[str]]] = {
    "direct_fields": {"default": ["huggingface", "github"]},
    "rag_fields_ai": {"default": ["huggingface", "arxiv", "github"]},
    "rag_fields_data": {"default": ["huggingface", "arxiv", "github"]},
}

_explicit_path: Optional[str] = None
_cache: Optional[Dict[str, Any]] = None


def set_source_priority_path(path: Optional[str]) -> None:
    """Pin the config to a specific path (or clear the pin) and flush the cache.

    Useful in tests and for programmatic override; production callers
    should usually rely on the ``AIKABOOM_SOURCE_PRIORITY`` env var.
    """
    global _explicit_path, _cache
    _explicit_path = path
    _cache = None


def _read_bundled() -> Dict[str, Any]:
    try:
        ref = resources.files(_CONFIG_PACKAGE).joinpath(_CONFIG_FILENAME)
        with resources.as_file(ref) as p:
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"[aikaboom] warning: bundled source-priority config unreadable ({exc}); "
            "falling back to hard-coded defaults.",
            file=sys.stderr,
        )
        return dict(_HARDCODED_FALLBACK)


def _read_user(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(
            f"[aikaboom] warning: source-priority override path '{path}' not found; "
            "using bundled defaults.",
            file=sys.stderr,
        )
    except json.JSONDecodeError as exc:
        print(
            f"[aikaboom] warning: source-priority override at '{path}' is not valid "
            f"JSON ({exc}); using bundled defaults.",
            file=sys.stderr,
        )
    except OSError as exc:
        print(
            f"[aikaboom] warning: source-priority override at '{path}' could not be "
            f"read ({exc}); using bundled defaults.",
            file=sys.stderr,
        )
    return None


def _merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Per-field merge: each top-level section is a dict whose entries are
    independent priority lists, so overlay entries replace base entries one
    field at a time without dropping unspecified fields."""
    merged = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for section, value in overlay.items():
        if section.startswith("$") or not isinstance(value, dict):
            merged[section] = value
            continue
        section_dict = merged.setdefault(section, {})
        if not isinstance(section_dict, dict):
            section_dict = {}
            merged[section] = section_dict
        for field, priority in value.items():
            section_dict[field] = priority
    return merged


def load_source_priority(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and cache the merged priority config.

    Args:
        path: If given, takes precedence over the env var and the cache.
    """
    global _cache
    if path is not None:
        cfg = _read_bundled()
        user = _read_user(path)
        return _merge(cfg, user) if user else cfg

    if _cache is not None:
        return _cache

    cfg = _read_bundled()
    override = _explicit_path or os.getenv(_ENV_VAR)
    if override:
        user = _read_user(override)
        if user:
            cfg = _merge(cfg, user)

    _cache = cfg
    return cfg


def _section_for_rag(bom_type: str) -> str:
    return "rag_fields_data" if bom_type.lower() in ("data", "dataset") else "rag_fields_ai"


def get_direct_priority(field: str) -> List[str]:
    """Priority list for a direct (structured-API) field.

    Falls back to ``direct_fields.default`` and finally to the
    hard-coded ``["huggingface", "github"]``.
    """
    cfg = load_source_priority()
    section = cfg.get("direct_fields") or {}
    if field in section and isinstance(section[field], list):
        return list(section[field])
    if isinstance(section.get("default"), list):
        return list(section["default"])
    return list(_HARDCODED_FALLBACK["direct_fields"]["default"])


def get_rag_priority(field: str, bom_type: str = "ai") -> List[str]:
    """Priority list for a RAG-extracted field of the given BOM type."""
    cfg = load_source_priority()
    section_name = _section_for_rag(bom_type)
    section = cfg.get(section_name) or {}
    if field in section and isinstance(section[field], list):
        return list(section[field])
    if isinstance(section.get("default"), list):
        return list(section["default"])
    return list(_HARDCODED_FALLBACK[section_name]["default"])
