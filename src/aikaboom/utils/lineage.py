"""Lineage / relationship-target string parsing.

Both the SPDX exporter (``_build_dataset_relationships``) and the
recursive walker (``recursive_bom._split_targets``) need to parse
``trainedOnDatasets`` / ``testedOnDatasets`` / ``modelLineage`` value
strings into individual targets. Until Phase 10 they each had their
own implementation, and the recursive walker's missed the arrow-
notation form (``"A -> B"``) — every modelLineage chain became a
single fake child node (Finding #17).

This module is the single source of truth so the two paths agree.
"""
from __future__ import annotations

import re
from typing import List, Optional

from aikaboom.utils.value_helpers import _is_nil_value


# ``,`` ``;`` newlines, plus the arrow forms LLMs often emit when an
# extracted lineage is a chain. Whitespace either side of an arrow is
# tolerated.
_LINEAGE_SEPARATORS = re.compile(r'[,;\n]|\s*(?:->|→)\s*')


# Phrases an LLM emits for "this model has no base model" — they answer
# the modelLineage question, they are not walkable parents. Matched
# case-insensitively as substrings; every phrase contains a space, so it
# cannot collide with an ``org/name`` identifier slug (slugs have no
# spaces). ``"scratch"`` alone is also caught as a whole-segment value.
_NO_LINEAGE_PHRASES = (
    "from scratch",
    "no base model",
    "no parent model",
    "no prior model",
    "no predecessor model",
    "no foundation model",
    "not derived from",
    "not based on",
    "no lineage",
    "no ancestor",
    "trained independently",
)


def _is_no_lineage_phrase(segment: str) -> bool:
    """True when a segment is a "no base model" answer, not a real target."""
    low = segment.strip().lower().rstrip(" .")
    if low in ("scratch", "not applicable", "no base model"):
        return True
    return any(phrase in low for phrase in _NO_LINEAGE_PHRASES)


def split_lineage_targets(
    value: str,
    parent_identifier: Optional[str] = None,
) -> List[str]:
    """Parse a lineage / relationship value string into unique targets.

    Returns a list of distinct target names with:
      - separators handled: comma, semicolon, newline, ``->``, ``→``
      - nil sentinels (``noAssertion``, ``Not found``, …) dropped
      - "no base model" answers (``from scratch``, ``no base model``, …)
        dropped — they answer the modelLineage question, they are not
        walkable parents
      - duplicates removed (case-insensitive)
      - self-loops dropped (any segment equal to ``parent_identifier``)

    Used by ``spdx_validator._build_dataset_relationships`` and
    ``recursive_bom._split_targets``. Empty / nil input → empty list.
    """
    if not isinstance(value, str) or _is_nil_value(value):
        return []

    parent_lower = (parent_identifier or "").strip().lower()
    seen: set[str] = set()
    out: List[str] = []
    for raw in _LINEAGE_SEPARATORS.split(value):
        candidate = (raw or "").strip()
        if not candidate:
            continue
        low = candidate.lower()
        if (
            low in seen
            or low == parent_lower
            or _is_nil_value(low)
            or _is_no_lineage_phrase(candidate)
        ):
            continue
        seen.add(low)
        out.append(candidate)
    return out


__all__ = ["split_lineage_targets"]
