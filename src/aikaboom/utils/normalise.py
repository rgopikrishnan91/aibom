"""Field-comparison normalisers shared across the direct and RAG pipelines.

Each helper normalises one kind of value so disagreement between sources
can be detected reliably (and so trivial differences — trailing slash on a
URL, leading ``v`` on a version, capitalisation on an org name — don't
register as a conflict). The functions are intentionally tiny and
side-effect-free; they're composed by the resolution code in
:mod:`aikaboom.core.source_handler` and :mod:`aikaboom.core.processors`.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------

def normalize_url(url: Any) -> str:
    """Return a comparable form of a download / repo URL.

    Lowercases the scheme and host, strips a leading ``www.``, drops the
    fragment, removes a single trailing slash from the path, and otherwise
    preserves the URL. Non-string and empty inputs return ``""``.
    """
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return url.strip().lower()
    scheme = (parsed.scheme or "").lower()
    netloc = (parsed.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path or ""
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    cleaned = urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))
    return cleaned


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

_VERSION_BUILD_META = re.compile(r"\+[0-9A-Za-z.-]+$")
_VERSION_LEADING_V = re.compile(r"^[vV]")


def normalize_version(version: Any) -> str:
    """Return a comparable form of a package / release version.

    Strips a leading ``v`` / ``V``, drops semver build metadata after a
    ``+``, and trims whitespace. Pre-release suffixes (``-rc1``, ``-alpha``)
    are preserved because they really do indicate different artefacts.
    """
    if version is None:
        return ""
    text = str(version).strip()
    if not text:
        return ""
    text = _VERSION_LEADING_V.sub("", text)
    text = _VERSION_BUILD_META.sub("", text)
    return text.lower()


# ---------------------------------------------------------------------------
# Org name
# ---------------------------------------------------------------------------

# Starts empty by design — populating this is left to operators / a future
# config slot. Keys are lower-cased; values are the canonical form.
_ORG_ALIASES: Dict[str, str] = {}


def normalize_org(name: Any, aliases: Optional[Dict[str, str]] = None) -> str:
    """Return a comparable form of an organisation / supplier name.

    Lowercases and strips whitespace, then maps through the alias table if
    a key is present. Non-string and empty inputs return ``""``.
    """
    if not name or not isinstance(name, (str, bytes)):
        return ""
    text = str(name).strip().lower()
    if not text:
        return ""
    table = aliases if aliases is not None else _ORG_ALIASES
    return table.get(text, text)


# ---------------------------------------------------------------------------
# Date-window conflict
# ---------------------------------------------------------------------------

def _parse_date(value: Any) -> Optional[datetime]:
    """Best-effort date parse using the same format set as
    :func:`SourceHandler.get_field`'s internal helper. Kept here so the
    conflict helper has no upward dependency.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    formats = (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m",
        "%Y",
    )
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=None)  # compare as naive
        except ValueError:
            continue
    return None


def date_window_conflict(
    chosen_value: Any,
    chosen_source: Optional[str],
    others: Sequence[Tuple[str, Any]],
    window_days: int = 7,
) -> Optional[Dict[str, Any]]:
    """Return a conflict dict if any sibling date is more than ``window_days``
    away from the chosen date, else ``None``.

    ``others`` is a sequence of ``(source_name, raw_value)`` tuples for the
    runner-up sources. The conflict dict has shape
    ``{value, source, type='inter', delta_days}``, recording the **largest**
    delta and the source that produced it.
    """
    chosen = _parse_date(chosen_value)
    if chosen is None:
        return None
    worst: Optional[Tuple[str, Any, int]] = None
    for src_name, raw in others:
        other = _parse_date(raw)
        if other is None or src_name == chosen_source:
            continue
        delta = abs((chosen - other).days)
        if delta > window_days and (worst is None or delta > worst[2]):
            worst = (src_name, raw, delta)
    if worst is None:
        return None
    return {
        "value": worst[1],
        "source": worst[0],
        "type": "inter",
        "delta_days": worst[2],
    }


# ---------------------------------------------------------------------------
# Named-entity dedupe (used for sourceInfo / trainedOnDatasets / etc.)
# ---------------------------------------------------------------------------

_LIST_SPLITTER = re.compile(r"[,;\n]+")
_BULLET_PREFIX = re.compile(r"^\s*[-*]\s*")


def dedupe_named_entities(value: Any) -> List[str]:
    """Tokenise a free-form RAG answer (or list) into a deduplicated,
    case-preserving list of entity names. Drops empty / "noAssertion"-style
    placeholders.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        pieces: Iterable[Any] = value
    elif isinstance(value, dict):
        pieces = value.values()
    else:
        pieces = _LIST_SPLITTER.split(str(value))

    out: List[str] = []
    seen = set()
    for piece in pieces:
        if isinstance(piece, dict) and "value" in piece:
            piece = piece.get("value")
        text = str(piece or "").strip()
        text = _BULLET_PREFIX.sub("", text)
        if not text or text.lower() in {"unknown", "none", "n/a", "noassertion", "not found"}:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# Enum coercion (delegates to spdx_validator's tables for SPDX 3.0.1 fidelity)
# ---------------------------------------------------------------------------

def collapse_whitespace(answer: Any) -> str:
    """Strip leading/trailing whitespace and collapse internal runs to one
    space. Used as the description post-processor.
    """
    if answer is None:
        return ""
    return re.sub(r"\s+", " ", str(answer)).strip()


def normalize_license(answer: Any) -> str:
    """Canonicalise an LLM-extracted license string through the SPDX alias
    table; falls back to the cleaned input.
    """
    from aikaboom.core.internal_conflict import LicenseConflictChecker
    if answer is None:
        return ""
    return LicenseConflictChecker.normalize_license(str(answer))


_POST_PROCESSORS = {
    "normalize_license": normalize_license,
    "normalize_purpose_enum": None,        # filled below to avoid forward-ref
    "normalize_availability_enum": None,
    "dedupe_named_entities": dedupe_named_entities,
    "collapse_whitespace": collapse_whitespace,
}


def get_post_processor(name: Optional[str]):
    """Return the callable named in a question's ``post_process`` key, or
    ``None`` if the name is unknown / unset. Unknown names emit a one-time
    warning so typos surface; missing names are a silent no-op.
    """
    if not name:
        return None
    fn = _POST_PROCESSORS.get(name)
    if fn is None and name not in _POST_PROCESSORS:
        print(f"[aikaboom] warning: unknown RAG post_process '{name}' — skipping")
    return fn


def normalize_purpose_enum(answer: Any) -> str:
    """Coerce a free-text answer to the SPDX ``software_primaryPurpose``
    enum, defaulting to ``"other"`` when the text doesn't match any value.
    """
    from aikaboom.utils.spdx_validator import SPDXValidator, _SOFTWARE_PURPOSES
    return SPDXValidator(bom_type="ai")._normalize_enum(answer, _SOFTWARE_PURPOSES, "other")


def normalize_availability_enum(answer: Any) -> str:
    """Coerce a free-text answer to the SPDX
    ``dataset_datasetAvailability`` enum, defaulting to ``"directDownload"``
    (the SPDX default) when the text doesn't match.
    """
    from aikaboom.utils.spdx_validator import SPDXValidator, _DATASET_AVAILABILITY_VALUES
    return SPDXValidator(bom_type="data")._normalize_enum(
        answer, _DATASET_AVAILABILITY_VALUES, "directDownload",
    )


# Now that the enum helpers are defined, plug them into the dispatch table.
_POST_PROCESSORS["normalize_purpose_enum"] = normalize_purpose_enum
_POST_PROCESSORS["normalize_availability_enum"] = normalize_availability_enum
