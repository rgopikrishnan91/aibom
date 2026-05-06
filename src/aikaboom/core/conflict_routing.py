"""Phase 4 — group-anonymized auditor parsing + consensus chunk routing.

Pure functions, no LLM or framework dependencies, so the unit tests
import this module directly without pulling in langchain/langgraph.

The conflict-detection prompt buckets chunks by source into groups
labelled A/B/C/... and asks the LLM to compare groups without telling
it which is which (eliminates source-name bias). The auditor's
deterministic line-based output is parsed here and consumed by the
consensus router, which scores each speaking source by agreement,
penalises self-contradicting sources, and filters chunks to the
highest-scoring source(s). Static priority breaks ties only under
unresolved external conflict.
"""

from __future__ import annotations

import re
import string
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


def _build_groups(documents, source_order):
    """Bucket documents by source into anonymized groups A/B/C/...

    ``source_order`` (typically the static priority list) determines the
    group-letter assignment so it stays deterministic across runs:
    e.g. huggingface→A, arxiv→B, github→C. Sources present in
    ``documents`` but absent from ``source_order`` are appended in
    alphabetical order.

    Returns:
        group_chunks: ``OrderedDict[letter -> list[chunk_text]]``
        group_to_source: ``Dict[letter -> source_name]``
    """
    by_src: "OrderedDict[str, List[str]]" = OrderedDict()
    seen = set()
    for src in source_order:
        for d in documents:
            if d.metadata.get('source') == src:
                by_src.setdefault(src, []).append(d.page_content)
                seen.add(src)
    extras = sorted(
        {d.metadata.get('source') for d in documents}
        - seen
        - {None, 'unknown'}
    )
    for src in extras:
        for d in documents:
            if d.metadata.get('source') == src:
                by_src.setdefault(src, []).append(d.page_content)

    letters = string.ascii_uppercase
    group_chunks: "OrderedDict[str, List[str]]" = OrderedDict()
    group_to_source: Dict[str, str] = {}
    for i, (src, chunks) in enumerate(by_src.items()):
        L = letters[i]
        group_chunks[L] = chunks
        group_to_source[L] = src
    return group_chunks, group_to_source


_SILENT_MARKERS = {"no relevant information", "n/a", "none", ""}


def _parse_detector_output(text, group_to_source):
    """Parse the auditor LLM's deterministic CLAIM / CONFLICT_* output.

    Returns:
        source_claims: ``Dict[source -> Optional[str]]``
            ``None`` when the source said "No relevant information".
        internal_conflicts: ``Dict[source -> str]``
            Only sources flagged self-contradicting; value is the LLM
            narrative (the bit after "Yes:").
        external_conflicts: ``List[Dict]``
            Each entry: ``{"sources": [src_a, src_b], "description": str}``.

    Never raises. Missing labels fall back to safe defaults
    (silent for claims, no-conflict for conflicts).
    """
    cleaned = re.sub(r'\*+', '', text or "")

    def find_line(label: str):
        m = re.search(
            rf'^\s*{re.escape(label)}\s*:\s*(.+?)\s*$',
            cleaned,
            flags=re.MULTILINE,
        )
        return m.group(1).strip() if m else None

    source_claims: Dict[str, Optional[str]] = {}
    internal_conflicts: Dict[str, str] = {}
    external_conflicts: List[Dict] = []

    for letter, src in group_to_source.items():
        raw = find_line(f"CLAIM_{letter}")
        if raw is None:
            source_claims[src] = None
            continue
        v = raw.strip().strip('"').strip("'").rstrip('.')
        if v.lower() in _SILENT_MARKERS:
            source_claims[src] = None
        else:
            source_claims[src] = v

    for letter, src in group_to_source.items():
        raw = find_line(f"CONFLICT_WITHIN_{letter}")
        if raw is None:
            continue
        if raw.lower().lstrip().startswith("yes"):
            narrative = re.sub(r'^\s*yes\s*:?\s*', '', raw, flags=re.I).strip()
            internal_conflicts[src] = narrative or "Yes"

    letters = list(group_to_source.keys())
    for i, a in enumerate(letters):
        for b in letters[i + 1 :]:
            raw = find_line(f"CONFLICT_{a}_VS_{b}")
            if raw is None:
                continue
            if raw.lower().lstrip().startswith("yes"):
                narrative = re.sub(r'^\s*yes\s*:?\s*', '', raw, flags=re.I).strip()
                external_conflicts.append({
                    "sources": [group_to_source[a], group_to_source[b]],
                    "description": narrative or "Yes",
                })

    return source_claims, internal_conflicts, external_conflicts


def _in_conflict(a, b, pairs):
    """True if (a, b) — in either order — appears in ``pairs``."""
    target = {a, b}
    return any(target == set(p) for p in pairs)


def _route_chunks(documents, source_claims, internal_conflicts,
                  external_conflicts, static_priority):
    """Consensus-based chunk routing.

    Returns ``(filtered_documents, selected_sources)``.

    Reliability score per speaking source:
      −2 if self-contradicting
      +1 per other speaking-and-clean source whose claim does not
         contradict this source's claim

    Tie-break under an external conflict uses ``static_priority``;
    tie without external conflict keeps all top-scoring sources.
    """
    speaking = [s for s, c in source_claims.items() if c is not None]

    if not internal_conflicts and not external_conflicts:
        return documents, speaking

    if len(speaking) <= 1:
        return documents, speaking

    internal_set = set(internal_conflicts.keys())
    external_pairs = [tuple(c["sources"]) for c in external_conflicts]

    scores: Dict[str, int] = {}
    for src in speaking:
        s = -2 if src in internal_set else 0
        for other in speaking:
            if other == src or other in internal_set:
                continue
            if not _in_conflict(src, other, external_pairs):
                s += 1
        scores[src] = s

    best = max(scores.values())

    if best < 0:
        return documents, speaking

    top = [s for s in speaking if scores[s] == best]

    # Tie-break to a single source ONLY when the top scorers are
    # themselves in conflict with each other. If they agree (e.g.
    # arxiv + github both override a contradicting huggingface), keep
    # all of them — that's the whole point of consensus routing.
    top_in_mutual_conflict = any(
        _in_conflict(top[i], top[j], external_pairs)
        for i in range(len(top))
        for j in range(i + 1, len(top))
    )

    if len(top) == 1:
        selected = {top[0]}
    elif top_in_mutual_conflict:
        prio = {s: i for i, s in enumerate(static_priority)}
        selected = {sorted(top, key=lambda s: prio.get(s, 99))[0]}
    else:
        selected = set(top)

    filtered = [d for d in documents if d.metadata.get('source') in selected]
    return filtered, sorted(selected)
