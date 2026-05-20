"""Canonicalize artifact identifiers and pick the primary one.

Pure functions only — no I/O, no graph access. The output is fully
determined by the input plus the supplier alias index loaded at startup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

from aikaboom.utils.supplier_alias import default_alias_index

PLATFORM_PRIORITY: tuple[str, ...] = ("huggingface", "github", "arxiv", "doi", "url")


@dataclass(frozen=True)
class Identifier:
    """A platform-typed identifier value."""

    platform: str
    value: str


_SEPARATOR_RUN = re.compile(r"[-_]{2,}")
_UNDERSCORE = re.compile(r"_")
_ARXIV_VERSION_SUFFIX = re.compile(r"v\d+$")
_GITHUB_DOT_GIT = re.compile(r"\.git$")

# Bare-host prefixes (no scheme) that should still be parsed as URLs.
# Lets us canonicalize "arxiv.org/abs/...", "huggingface.co/owner/repo",
# "github.com/owner/repo" the same way as their https:// forms.
_BARE_HOST_PREFIXES: tuple[str, ...] = (
    "huggingface.co/",
    "github.com/",
    "arxiv.org/",
)


# Platforms whose canonical value is shaped like "owner/repo" — i.e.,
# the segment before the first slash is a supplier handle eligible for
# alias resolution. arxiv/doi/url are not owner-shaped and must be skipped.
_OWNER_SHAPED_PLATFORMS: frozenset[str] = frozenset({"huggingface", "github"})


def _strip_url(platform: str, value: str) -> str:
    """Reduce a URL form to its canonical path component."""
    if platform == "url":
        # The url platform is an opaque last-resort identifier — preserve the
        # full value rather than discarding scheme/host.
        return value
    has_scheme = "://" in value
    is_bare_host = value.startswith("www.") or value.startswith(_BARE_HOST_PREFIXES)
    if not has_scheme and not is_bare_host:
        return value
    parsed = urlparse(value if has_scheme else f"https://{value}")
    path = parsed.path.lstrip("/")
    if platform == "huggingface":
        # Drop /tree/main, /blob/<sha>/..., etc — keep "owner/repo".
        parts = path.split("/")
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return path
    if platform == "github":
        parts = path.split("/")
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return path
    if platform == "arxiv":
        # /abs/2310.06825 or /pdf/2310.06825 → 2310.06825
        parts = [p for p in path.split("/") if p and p not in ("abs", "pdf")]
        return parts[-1] if parts else path
    return path


def _resolve_owner_alias(platform: str, value: str) -> str:
    """For owner/repo-shaped identifiers, canonicalize the owner via the alias index.

    Identity-level canonicalization is intentionally conservative: it only
    accepts alias rewrites that *expand* (or preserve length of) the owner
    handle — for example ``QwenLM`` is unchanged at this layer, and an
    already-canonical handle like ``mistralai`` is preserved verbatim.
    Shortening rewrites such as ``mistralai → mistral`` (produced by the
    alias index's ``normalize_org`` suffix-stripping) are *not* applied here:
    those collapses belong to a higher matching layer (same-supplier
    detection, used for conflict checks and provenance), where one Artifact
    can be linked to several supplier handles without losing identity.

    Only owner-shaped platforms (HuggingFace, GitHub) are eligible — arxiv/doi/url
    use opaque identifiers whose leading segment is not a supplier handle.
    """
    if platform not in _OWNER_SHAPED_PLATFORMS:
        return value
    if "/" not in value:
        return value
    owner, _, rest = value.partition("/")
    canonical = default_alias_index().canonicalize(owner)
    # Apply the alias only when it is at least as long as the input owner —
    # a true expansion or rename, not a suffix-stripping collapse.
    if canonical and len(canonical) >= len(owner):
        return f"{canonical.lower()}/{rest}"
    return f"{owner.lower()}/{rest}"


def canonicalize(ident: Identifier) -> Identifier:
    """Apply the canonicalization pipeline to a single identifier.

    Steps: strip URL noise → lowercase → trim → resolve owner alias →
    collapse separator runs → platform-specific trimming.
    """
    value = ident.value.strip()
    value = _strip_url(ident.platform, value)
    value = value.lower().strip()
    value = _UNDERSCORE.sub("-", value)
    value = _SEPARATOR_RUN.sub("-", value)
    value = _resolve_owner_alias(ident.platform, value).lower()
    if ident.platform == "github":
        value = _GITHUB_DOT_GIT.sub("", value)
    if ident.platform == "arxiv":
        value = _ARXIV_VERSION_SUFFIX.sub("", value)
    return Identifier(platform=ident.platform, value=value)


def canonicalize_set(ids: Iterable[Identifier]) -> list[Identifier]:
    """Canonicalize each id and dedupe by (platform, value) within the set."""
    seen: set[tuple[str, str]] = set()
    out: list[Identifier] = []
    for ident in ids:
        canon = canonicalize(ident)
        key = (canon.platform, canon.value)
        if key not in seen:
            seen.add(key)
            out.append(canon)
    return out


def pick_primary(ids: Iterable[Identifier]) -> Identifier:
    """Pick the highest-priority identifier from a set."""
    canon = canonicalize_set(ids)
    if not canon:
        raise ValueError("pick_primary requires at least one identifier")
    by_platform = {i.platform: i for i in canon}
    for platform in PLATFORM_PRIORITY:
        if platform in by_platform:
            return by_platform[platform]
    # Unknown platform — return the first one.
    return canon[0]
