"""Shared "is this a nil-value sentinel?" helper.

The pipeline treats a number of strings as "the source said nothing
useful here": ``noAssertion``, ``Not found.``, ``N/A``, etc. Several
emitters need the same check (SPDX builder, CycloneDX builder, the
shared lineage splitter), so the definition lives in one place.

Originally a private helper in ``spdx_validator.py``; lifted into a
shared module in Phase 10 so the CycloneDX exporter could apply the
same filter without duplicating the constant (Finding #24).
"""
from __future__ import annotations

from typing import Any


# Recognised case-insensitively; normalised via str().strip().lower().
_NIL_VALUES = frozenset({
    "not found",
    "not found.",
    "noassertion",
    "n/a",
    "none",
    "",
})


def _is_nil_value(value: Any) -> bool:
    """True if ``value`` is one of the no-information sentinels.

    Accepts ``None`` (treated as nil) and any object whose ``str()``
    representation matches a sentinel after stripping + lowercasing.
    """
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in _NIL_VALUES


__all__ = ["_NIL_VALUES", "_is_nil_value"]
