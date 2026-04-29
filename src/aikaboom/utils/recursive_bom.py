"""Beta recursive BOM generation gated on relationship conflicts.

The recursion strategy follows the design discussed during the SPDX work:

1. ``trainedOnDatasets`` / ``testedOnDatasets`` / ``modelLineage`` are extracted
   by the existing RAG question bank in :mod:`aikaboom.core.agentic_rag`.
   Those fields land in ``rag_fields`` as ``{value, source, conflict}``
   triplets, with both inter-source and intra-source conflicts already
   detected against the README, arXiv, GitHub and HuggingFace model tree
   sources.
2. For each relationship field, if a conflict is present the field is
   skipped — we cannot safely recurse on contested data. The skip and the
   underlying conflict are reported back to the caller.
3. For conflict-free fields, the named targets are split out and used to
   seed child Provenance / SPDX / CycloneDX exports.

The module never performs additional network or LLM enrichment on its own.
Callers can feed the seed records back into the full pipeline if they want
deeper recursion.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


AI_RELATIONSHIP_FIELDS = {
    "trainedOnDatasets": ("data", "trainedOn"),
    "testedOnDatasets": ("data", "testedOn"),
    "modelLineage": ("ai", "dependsOn"),
}


def _extract_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value and "conflict" in value:
        return value.get("value")
    if isinstance(value, dict) and "value" in value and len(value) <= 3:
        return value.get("value")
    return value


def _conflict_of(triplet: Any) -> Optional[Dict[str, Any]]:
    """Return a structured conflict dict if the triplet has one, else None.

    Handles both shapes that flow through the pipeline:
      * RAG-style: ``{'internal': 'No|Yes...', 'external': 'No|Yes...'}``
      * Parsed-style: ``{'value': ..., 'source': ..., 'type': 'inter'|'intra'}``
    """
    if not isinstance(triplet, dict):
        return None
    raw = triplet.get("conflict")
    if not raw:
        return None
    if not isinstance(raw, dict):
        text = str(raw).strip().lower()
        if not text or text.startswith("no"):
            return None
        return {"value": str(raw), "type": "inter"}

    flagged: Dict[str, Any] = {}
    for key in ("internal", "external"):
        v = raw.get(key)
        if isinstance(v, str) and v.strip().lower().startswith("yes"):
            flagged[key] = v
    if flagged:
        return {**flagged, "type": "inter" if "external" in flagged else "intra"}

    if raw.get("type") and (raw.get("value") or raw.get("source")):
        return raw
    return None


def _split_targets(value: Any) -> List[str]:
    value = _extract_value(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        pieces: Iterable[Any] = value
    elif isinstance(value, dict):
        pieces = value.values()
    else:
        pieces = re.split(r"[,;\n]+", str(value))

    targets: List[str] = []
    seen = set()
    for piece in pieces:
        text = str(_extract_value(piece) or "").strip()
        text = re.sub(r"^\s*[-*]\s*", "", text)
        if not text or text.lower() in {"unknown", "none", "n/a", "noassertion"}:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            targets.append(text)
    return targets


def _safe_id(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip()).strip("-")
    return slug or "related-artifact"


def discover_recursive_targets(
    metadata: Dict[str, Any],
    bom_type: str = "ai",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Find conflict-free relationship targets that can seed child BOMs.

    Returns a tuple of ``(targets, audit)`` where ``audit`` contains any
    fields that were skipped because the parent BOM reported a conflict on
    that relationship.
    """
    audit: Dict[str, Any] = {"skipped_due_to_conflict": [], "considered": []}
    if bom_type != "ai":
        audit["reason"] = "recursion only supported for AI BOMs"
        return [], audit

    rag_fields = metadata.get("rag_fields") or {}
    parent_id = metadata.get("model_id") or metadata.get("repo_id") or "parent-bom"
    targets: List[Dict[str, Any]] = []

    for field, (child_bom_type, relationship_type) in AI_RELATIONSHIP_FIELDS.items():
        triplet = rag_fields.get(field)
        audit["considered"].append(field)

        conflict = _conflict_of(triplet)
        if conflict is not None:
            audit["skipped_due_to_conflict"].append({
                "field": field,
                "relationship_type": relationship_type,
                "reason": "conflict-detected",
                "conflict": conflict,
            })
            continue

        for target in _split_targets(_extract_value(triplet) if isinstance(triplet, dict) else triplet):
            targets.append({
                "source_field": field,
                "relationship_type": relationship_type,
                "target": target,
                "bom_type": child_bom_type,
                "parent": parent_id,
                "resolvable_hint": "/" in target and " " not in target,
            })

    return targets, audit


def _build_child_metadata(target: Dict[str, Any]) -> Dict[str, Any]:
    name = target["target"]
    safe_id = _safe_id(name)
    if target["bom_type"] == "data":
        urls = {}
        if target["resolvable_hint"]:
            urls["huggingface"] = f"https://huggingface.co/datasets/{name}"
        return {
            "dataset_id": safe_id,
            "direct_metadata": {
                "name": name,
                "license": "NOASSERTION",
            },
            "rag_metadata": {
                "intendedUse": f"Referenced by {target['parent']} via {target['relationship_type']}",
            },
            "urls": urls,
            "recursive_source": target,
        }

    return {
        "model_id": safe_id,
        "repo_id": name if target["resolvable_hint"] else safe_id,
        "direct_fields": {
            "license": "NOASSERTION",
        },
        "rag_fields": {
            "model_name": name,
        },
        "recursive_source": target,
    }


def generate_recursive_boms(
    metadata: Dict[str, Any],
    bom_type: str = "ai",
    max_depth: int = 1,
    validate_spdx: bool = True,
    strict_spdx: bool = False,
) -> Dict[str, Any]:
    """Generate beta recursive child BOM exports from discovered relationships.

    Children are only generated for relationship fields that have no
    conflict in the parent BOM. Skipped fields are reported in
    ``skipped_due_to_conflict``. ``max_depth`` currently controls a single
    recursion level; values >1 are accepted but do not yet trigger deeper
    expansion (beta).
    """
    max_depth = max(0, int(max_depth or 0))
    if max_depth > 0:
        targets, audit = discover_recursive_targets(metadata, bom_type=bom_type)
    else:
        targets, audit = [], {"skipped_due_to_conflict": [], "considered": []}

    generated: List[Dict[str, Any]] = []

    for target in targets:
        child_metadata = _build_child_metadata(target)
        child_type = target["bom_type"]
        item: Dict[str, Any] = {
            "beta": True,
            "depth": 1,
            "relationship_type": target["relationship_type"],
            "source_field": target["source_field"],
            "target": target["target"],
            "bom_type": child_type,
            "metadata": child_metadata,
        }

        try:
            from aikaboom.utils.spdx_validator import SPDXValidator, validate_spdx_export

            spdx = SPDXValidator(bom_type=child_type).validate_and_convert(child_metadata)
            item["spdx_data"] = spdx
            if validate_spdx:
                item["spdx_validation"] = validate_spdx_export(
                    spdx,
                    strict=strict_spdx,
                    bom_type=child_type,
                )
        except Exception as exc:
            item["spdx_error"] = str(exc)

        try:
            from aikaboom.utils.cyclonedx_exporter import bom_to_cyclonedx

            item["cyclonedx_data"] = bom_to_cyclonedx(child_metadata, bom_type=child_type)
        except Exception as exc:
            item["cyclonedx_error"] = str(exc)

        generated.append(item)

    return {
        "beta": True,
        "enabled": True,
        "max_depth": max_depth,
        "strategy": "conflict-gated relationship recursion",
        "generated_count": len(generated),
        "generated": generated,
        "skipped_due_to_conflict": audit.get("skipped_due_to_conflict", []),
        "considered_fields": audit.get("considered", []),
        "warnings": [
            "Recursive BOM generation is beta.",
            "Fields with internal or external conflicts are skipped; resolve "
            "the conflict in the parent BOM before recursing.",
            "max_depth>1 is accepted but currently produces a single recursion level.",
        ],
    }
