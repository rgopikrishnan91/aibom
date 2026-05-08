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


# Phase 12B — grounding-score threshold for "high-confidence" vs
# "low-confidence" conflicts. Conflicts whose quoted statements
# match well against the actual chunks (cosine ≥ this value) are
# high-confidence; lower scores suggest paraphrase, scope confusion,
# or LLM hallucination. Single source of truth so the BOM, web UI,
# HF Space, and CLI all agree on the cutoff.
HIGH_CONFIDENCE_THRESHOLD = 0.7


# Match `"<stmt 1>" vs "<stmt 2>"` (internal) or `A says "<stmt>" vs
# B says "<stmt>"` (external). Captures the two quoted statements.
_INTERNAL_STMTS_RE = re.compile(r'"([^"]+)"\s*vs\s*"([^"]+)"', re.IGNORECASE)
_EXTERNAL_STMTS_RE = re.compile(
    r'[A-Z]\s+says\s+"([^"]+)"\s*vs\s+[A-Z]\s+says\s+"([^"]+)"',
    re.IGNORECASE,
)


def extract_conflict_statements(narrative, kind="internal"):
    """Extract the two quoted statements from an auditor narrative.

    ``kind`` is ``"internal"`` for ``"<a>" vs "<b>"`` shape, or
    ``"external"`` for ``A says "<a>" vs B says "<b>"`` shape.

    Returns a ``(stmt_a, stmt_b)`` tuple. Falls back to
    ``("", "")`` when the regex doesn't match — the narrative still
    surfaces in the BOM, but grounding scoring will see an empty
    statement set and return 0.0 (treated as low confidence).
    """
    if not narrative:
        return ("", "")
    pattern = _EXTERNAL_STMTS_RE if kind == "external" else _INTERNAL_STMTS_RE
    m = pattern.search(narrative)
    if m:
        return (m.group(1).strip(), m.group(2).strip())
    # External regex falls back to internal-style for narratives that
    # don't carry the "A says" / "B says" prefixes.
    if kind == "external":
        m = _INTERNAL_STMTS_RE.search(narrative)
        if m:
            return (m.group(1).strip(), m.group(2).strip())
    return ("", "")


def _confidence_label(grounding):
    """Map a grounding score to ``"high"`` / ``"low"`` / ``"n/a"``.

    ``None`` (no LLM-judgement; deterministic direct/intra triplet
    conflicts) → ``"n/a"``.
    """
    if grounding is None:
        return "n/a"
    return "high" if grounding >= HIGH_CONFIDENCE_THRESHOLD else "low"


def summarise_conflicts(metadata):
    """Walk a BOM metadata dict and return a count summary.

    Returns ``{"total": int, "high_confidence": int, "low_confidence": int}``.
    Counts (a) every direct/rag triplet ``conflict`` entry as
    ``n/a``-confidence (deterministic; no grounding score), and
    (b) every RAG ``trace.internal_conflicts`` entry and
    ``trace.external_conflicts`` entry, classified by their
    ``grounding`` field against ``HIGH_CONFIDENCE_THRESHOLD``.

    Direct/intra triplet conflicts count toward ``total`` but split
    into neither ``high_confidence`` nor ``low_confidence`` (they have
    no LLM-judgement score). Display surfaces should report:
    ``"N total (H high-confidence, L low-confidence)"`` and let the
    remainder ``N - H - L`` represent the deterministic rest.
    """
    total = 0
    high = 0
    low = 0
    if not isinstance(metadata, dict):
        return {"total": 0, "high_confidence": 0, "low_confidence": 0}

    for section in ("direct_fields", "rag_fields"):
        fields = metadata.get(section, {})
        if not isinstance(fields, dict):
            continue
        for triplet in fields.values():
            if not isinstance(triplet, dict):
                continue
            if isinstance(triplet.get("conflict"), dict):
                # Direct/intra triplet conflict — deterministic, n/a confidence.
                total += 1
            trace = triplet.get("trace") or {}
            for entry in (trace.get("internal_conflicts") or {}).values():
                total += 1
                if isinstance(entry, dict):
                    g = entry.get("grounding")
                    if g is not None:
                        if g >= HIGH_CONFIDENCE_THRESHOLD:
                            high += 1
                        else:
                            low += 1
            for entry in (trace.get("external_conflicts") or []):
                total += 1
                if isinstance(entry, dict):
                    g = entry.get("grounding")
                    if g is not None:
                        if g >= HIGH_CONFIDENCE_THRESHOLD:
                            high += 1
                        else:
                            low += 1
    return {"total": total, "high_confidence": high, "low_confidence": low}


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

    Tie-break: when multiple sources tie at the highest score AND the
    tied sources are themselves in mutual external conflict, fall back
    to ``static_priority`` to pick one. If the tied sources agree with
    each other (e.g. arxiv + github both overriding a contradicting
    huggingface), all are kept — that's the whole point of consensus
    routing. The presence of *other* external conflicts elsewhere does
    not trigger the tie-break.
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

    # Safety net: a too-thin context starves the answer LLM. If the consensus
    # winner contributed fewer than 3 chunks, fall back to all documents and
    # let the answerer cope — better than handing it 1-2 chunks.
    if len(filtered) < 3:
        return documents, sorted(speaking)

    return filtered, sorted(selected)
