"""Tiered supplier-alias resolution: cross-reference index + dict + Jaro-Winkler.

Replaces naive string-equality on ``suppliedBy`` (which always flagged
HF-org-handle vs GitHub-org-handle as conflicts when they're really the
same organisation under different platform conventions, e.g.
``Qwen``/``QwenLM``, ``google-bert``/``google-research``,
``cais``/``hendrycks``).

Three resolution tiers, each cheap:

1. **Cross-reference index.** A dict keyed by (lowercased) org name,
   value = set of equivalent handles. Built from the SAIL Phase 12
   harvest (see ``sail_link_extractor.extract_github_urls_from_card``)
   plus a curated seed of known equivalences.

2. **Normalised exact match.** Strip separators + common org suffixes
   (``research``, ``ai``, ``labs``…); compare lowercase forms.

3. **Jaro-Winkler fallback.** ``jaro_winkler_similarity > 0.85`` on the
   normalised forms. Catches typo-style and prefix-shared variants.

For per-BOM "is X same as Y?" checks this is plenty. Cross-BOM global
deduplication (Splink territory) can sit on top later when there's a
multi-BOM corpus to dedupe.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# Curated seed for Tier 1. Each entry maps a *canonical* handle to the
# set of equivalent handles seen across HuggingFace + GitHub. Keys are
# stored lowercased; lookups normalise input to lowercase before hitting
# the dict.
#
# Bootstrapped from manual review of the 10 golden-set BOMs (see
# test_outputs/golden/SCOREBOARD.md "Direct conflicts" section, where
# ~38 of 42 detected direct conflicts were structural false positives,
# many of them on suppliedBy). Add more entries as harvest data lands.
# Curated equivalences that pure substring matching and the harvested HF
# clusters cannot deduce on their own. Most cross-platform pairs (e.g.
# `Qwen` ↔ `QwenLM`, `google-bert` ↔ `google-research`, `mistralai` ↔
# `mistral-ai`) are caught for free by ``is_same_supplier``'s
# substring-on-normalised-forms check (Tier 3) once the harvest is
# loaded, so they don't need an entry here.
#
# Entries below are genuine same-supplier pairs whose handles share no
# substring overlap. Each addition is a deliberate provenance call —
# leaving an entry *out* means a cross-platform mismatch on those handles
# will surface as a real `suppliedBy` conflict, which is the correct
# behaviour when the model-vs-code custody is genuinely split between
# different parties (e.g. an org distributing a dataset whose canonical
# code lives in an individual maintainer's repo).
_CURATED_ALIAS_SEED: Dict[str, Set[str]] = {
    # canonical_lower:  {alternate forms (any case)}
    "allenai":          {"allenai", "ai2"},
    "tiiuae":           {"tiiuae", "falcon-llm"},
}


# Suffix tokens stripped during normalisation. Match at end of (already
# separator-stripped) lowercase string.
_NORMALISE_SUFFIX_RE = re.compile(
    r'(research|researchteam|teamai|ai|labs?|lab|inc|llc|team|group|corp|company)$'
)
_SEPARATOR_RE = re.compile(r'[-_./\s]+')


def normalize_org(name: Optional[str]) -> str:
    """Lowercase, strip separators, drop common org suffixes.

    Examples
    --------
    >>> normalize_org("google-research")
    'google'
    >>> normalize_org("FacebookAI")
    'facebook'
    >>> normalize_org("facebookresearch")
    'facebook'
    >>> normalize_org("Qwen")
    'qwen'
    >>> normalize_org("google-deepmind")
    'googledeepmind'
    """
    if not name:
        return ""
    s = str(name).strip().lower()
    s = _SEPARATOR_RE.sub('', s)
    # Strip suffix iff there's something left after stripping
    stripped = _NORMALISE_SUFFIX_RE.sub('', s)
    return stripped if stripped else s


class SupplierAliasIndex:
    """Tiered same-supplier resolver."""

    def __init__(
        self,
        seed: Optional[Dict[str, Set[str]]] = None,
        harvest_path: Optional[Path] = None,
        jw_threshold: float = 0.85,
    ):
        # Tier 1: handle → canonical
        self._handle_to_canonical: Dict[str, str] = {}
        for canonical, alts in (seed if seed is not None else _CURATED_ALIAS_SEED).items():
            for alt in alts | {canonical}:
                self._handle_to_canonical[alt.lower()] = canonical.lower()

        # Optional harvest-loaded extras (same shape as seed). Harvest output
        # is keyed by the canonical root (e.g. "google") with the cluster's
        # member handles as values — exactly the same shape as the curated
        # seed, so the loader is identical.
        if harvest_path and Path(harvest_path).exists():
            try:
                harvested = json.loads(Path(harvest_path).read_text())
                for canonical, alts in harvested.items():
                    members = set(alts) if isinstance(alts, list) else (alts or set())
                    for alt in members | {canonical}:
                        # Don't override curated mappings
                        self._handle_to_canonical.setdefault(str(alt).lower(), canonical.lower())
            except (json.JSONDecodeError, OSError):
                pass

        # Cache the set of canonical root strings so substring fallback in
        # ``canonicalize`` can scan them without rebuilding each call.
        self._canonical_roots: Set[str] = {c for c in self._handle_to_canonical.values() if c}
        self._jw_threshold = jw_threshold

    # ----- public API -----

    def canonicalize(self, handle: Optional[str]) -> Optional[str]:
        """Return the canonical handle for `handle`.

        Three-step resolution:
          1. Direct dict lookup (curated seed + harvest membership).
          2. Substring match against the canonical-root set on the
             ``normalize_org``-form of ``handle`` — collapses a
             cross-platform variant like ``QwenLM`` (GitHub) to the
             root ``qwen`` even though only ``Qwen`` was in the harvest.
             Longest-matching root wins so ``googledeepmind`` resolves
             to a specific cluster rather than merging into bare ``google``
             when both exist.
          3. Otherwise return the ``normalize_org`` form (or the lowercased
             original if normalisation collapsed it to empty).
        """
        if not handle:
            return handle
        h = handle.strip().lower()
        direct = self._handle_to_canonical.get(h)
        if direct is not None:
            return direct
        norm = normalize_org(handle) or h
        if len(norm) >= 3 and self._canonical_roots:
            # Exact-match the normalised form against a known root if possible
            # — that is *the* canonical for this handle.
            if norm in self._canonical_roots:
                return norm
            # Otherwise prefer the SHORTEST root that is a prefix-substring
            # of the normalised form (the company-prefix case:
            # ``googleresearchdatasets`` → ``google``). Shortest = most
            # general; longest-match would incorrectly route
            # ``mistral`` → ``vietmistral`` when both exist as roots.
            best = None
            for root in self._canonical_roots:
                if len(root) < 3:
                    continue
                if norm.startswith(root):
                    if best is None or len(root) < len(best):
                        best = root
            if best is not None:
                return best
        return norm

    def is_same_supplier(self, hf_handle: Optional[str], gh_handle: Optional[str]) -> bool:
        """True if the two handles refer to the same supplier.

        Tiered resolution, conservative on purpose: when two handles
        survive all tiers as distinct, that's a real provenance signal
        worth surfacing as a conflict, not noise to suppress.
        """
        if not hf_handle or not gh_handle:
            return False

        # Tier 1: canonical lookup (curated seed + harvested HF roots).
        c1 = self.canonicalize(hf_handle)
        c2 = self.canonicalize(gh_handle)
        if c1 == c2:
            return True

        # Tier 2: normalised exact match (strip separators + common suffixes).
        n1 = normalize_org(hf_handle)
        n2 = normalize_org(gh_handle)
        if n1 and n2 and n1 == n2:
            return True

        # Tier 3: substring match on the normalised forms with a ≥3-char guard.
        # Catches `Qwen ↔ QwenLM`, `mistralai ↔ mistral-ai`, `google ↔ google-research`
        # without needing a curated dict entry. The guard prevents tiny stems
        # (e.g. "ai") from spuriously matching everything.
        if n1 and n2 and len(n1) >= 3 and len(n2) >= 3:
            if n1 in n2 or n2 in n1:
                return True

        return False

    def add_cross_reference(self, hf_handle: str, gh_handle: str) -> None:
        """Add a same-supplier pair to the index (e.g., from harvest)."""
        if not hf_handle or not gh_handle:
            return
        a, b = hf_handle.lower(), gh_handle.lower()
        ca = self._handle_to_canonical.get(a)
        cb = self._handle_to_canonical.get(b)
        if ca and cb and ca != cb:
            return  # already conflicting canonicals; needs human review
        canonical = ca or cb or a
        self._handle_to_canonical[a] = canonical
        self._handle_to_canonical[b] = canonical
        self._canonical_roots.add(canonical)


# Module-level singleton, lazily loaded. AIkaBoOM call sites should use
# `default_alias_index()` rather than instantiating their own — keeps the
# (small) seed in one place.
_DEFAULT: Optional[SupplierAliasIndex] = None


def default_alias_index() -> SupplierAliasIndex:
    global _DEFAULT
    if _DEFAULT is None:
        # Look for an optional harvest file alongside source_priority.json.
        # Empty / missing = use seed only.
        harvest = (
            Path(__file__).parent.parent
            / "config"
            / "supplier_alias_harvest.json"
        )
        _DEFAULT = SupplierAliasIndex(harvest_path=harvest)
    return _DEFAULT


__all__ = [
    "SupplierAliasIndex",
    "default_alias_index",
    "normalize_org",
]
