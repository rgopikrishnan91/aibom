"""Artifact-to-artifact relationship edges for the worldofBOMs graph.

A BOM's relationship fields (trainedOnDatasets, testedOnDatasets,
modelLineage, sourceInfo) name other artifacts. This module turns those
names into real `trainedOn` / `testedOn` / `dependsOn` edges between
Artifact nodes, so the stored graph is connected rather than a set of
disconnected stars.
"""

from __future__ import annotations

from typing import Any, Mapping

from aikaboom.utils.lineage import split_lineage_targets
from aikaboom.utils.recursive_bom import (
    AI_RELATIONSHIP_FIELDS,
    DATA_RELATIONSHIP_FIELDS,
    _is_walkable_target,
)

# {bom_field_name: edge_predicate_name}. Reuses the single source of truth
# in recursive_bom.py — the second tuple element is the predicate.
_FIELD_TO_PREDICATE: dict[str, str] = {
    field: spec[1]
    for field, spec in {**AI_RELATIONSHIP_FIELDS, **DATA_RELATIONSHIP_FIELDS}.items()
}


def _split_targets(value: Any) -> list[str]:
    """Normalize a relationship-field value into a list of target names.

    Delegates to ``split_lineage_targets`` so that arrow-notation chains
    (``"bert-base -> distilbert"`` / ``"A → B"``) split correctly in
    addition to comma/semicolon/newline separators.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for v in value:
            out.extend(split_lineage_targets(str(v)))
        return out
    return split_lineage_targets(str(value))


def extract_relationship_targets(bom_json: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return `(predicate, target_name)` pairs for every walkable edge target.

    `predicate` is one of "trainedOn" / "testedOn" / "dependsOn".

    Scans both ``direct_fields`` and ``rag_fields`` because the graph store
    may receive BOMs that store relationship fields in either section depending
    on how the extractor classified them (high-confidence direct extraction vs.
    RAG-assisted retrieval).
    """
    out: list[tuple[str, str]] = []
    for section in ("direct_fields", "rag_fields"):
        fields = bom_json.get(section) or {}
        if not isinstance(fields, Mapping):
            continue
        for field_name, predicate in _FIELD_TO_PREDICATE.items():
            triplet = fields.get(field_name)
            if not isinstance(triplet, Mapping):
                continue
            for target in _split_targets(triplet.get("value")):
                if _is_walkable_target(target):
                    out.append((predicate, target))
    return out
