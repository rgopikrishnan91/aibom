"""Beta recursive BOM generation gated on relationship conflicts.

The recursion walks the dependency tree of an AI BOM:

* ``trainedOnDatasets`` and ``testedOnDatasets`` produce *data* BOM children.
* ``modelLineage`` produces *ai* BOM children that can themselves be
  expanded one level deeper.

Each level reuses the existing RAG question bank in
:mod:`aikaboom.core.agentic_rag` to build conflict-aware triplets — fields
with internal or external conflicts are *skipped*, so we never recurse on
contested data.

Recursion runs until one of three things happens:

1. ``max_depth`` is reached.
2. The unique-target set is exhausted (the natural end of the tree).
3. Every newly discovered field is conflict-flagged or already visited.

The module performs no network/LLM calls on its own. Callers can pass an
``enrich_fn`` callback to fetch the full metadata of a discovered target
(e.g. by running the existing ``AIBOMProcessor`` / ``DATABOMProcessor``);
without it the children are seed records derived from the parent's
relationship strings, which means the tree typically terminates after one
level.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


AI_RELATIONSHIP_FIELDS = {
    "trainedOnDatasets": ("data", "trainedOn"),
    "testedOnDatasets": ("data", "testedOn"),
    "modelLineage": ("ai", "dependsOn"),
}


EnrichFn = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]


def _extract_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value.get("value")
    return value


def _conflict_of(triplet: Any) -> Optional[Dict[str, Any]]:
    """Return a structured conflict dict if the triplet has one, else None."""
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


def _visit_key(bom_type: str, target: str) -> Tuple[str, str]:
    return (bom_type.lower(), target.strip().lower())


def discover_recursive_targets(
    metadata: Dict[str, Any],
    bom_type: str = "ai",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Find conflict-free relationship targets that can seed child BOMs.

    Returns ``(targets, audit)`` where ``audit`` lists fields skipped
    because the parent BOM reported a conflict on that relationship.
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

        for target in _split_targets(triplet):
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
    """Seed-level metadata for a target when no enrich callback is provided."""
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


def _build_node(
    target: Dict[str, Any],
    child_metadata: Dict[str, Any],
    depth: int,
    validate_spdx: bool,
    strict_spdx: bool,
    enrichment_error: Optional[str] = None,
    enriched: bool = False,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "beta": True,
        "depth": depth,
        "relationship_type": target["relationship_type"],
        "source_field": target["source_field"],
        "target": target["target"],
        "bom_type": target["bom_type"],
        "parent": target["parent"],
        "metadata": child_metadata,
        "enriched": enriched,
    }
    if enrichment_error:
        item["enrichment_error"] = enrichment_error

    try:
        from aikaboom.utils.spdx_validator import SPDXValidator, validate_spdx_export

        spdx = SPDXValidator(bom_type=target["bom_type"]).validate_and_convert(child_metadata)
        item["spdx_data"] = spdx
        if validate_spdx:
            item["spdx_validation"] = validate_spdx_export(
                spdx, strict=strict_spdx, bom_type=target["bom_type"]
            )
    except Exception as exc:
        item["spdx_error"] = str(exc)

    try:
        from aikaboom.utils.cyclonedx_exporter import bom_to_cyclonedx

        item["cyclonedx_data"] = bom_to_cyclonedx(child_metadata, bom_type=target["bom_type"])
    except Exception as exc:
        item["cyclonedx_error"] = str(exc)

    return item


def generate_recursive_boms(
    metadata: Dict[str, Any],
    bom_type: str = "ai",
    max_depth: int = 1,
    validate_spdx: bool = True,
    strict_spdx: bool = False,
    enrich_fn: Optional[EnrichFn] = None,
) -> Dict[str, Any]:
    """Walk the dependency tree of an AI BOM and emit child BOMs.

    Args:
        metadata: Parent BOM metadata dict (with ``rag_fields``).
        bom_type: Parent BOM type. Recursion only descends through AI BOMs;
            ``data`` parents are leaves.
        max_depth: Maximum tree depth (1 = direct children only). Recursion
            also stops naturally when the unique-target set is exhausted.
        validate_spdx: Validate each generated child SPDX export.
        strict_spdx: Use the SHACL strict pass (beta).
        enrich_fn: Optional callable ``(target_dict) -> metadata_dict`` that
            fetches full metadata for a discovered target. Without it,
            children carry only seed metadata and recursion typically stops
            after one level.
    """
    max_depth = max(0, int(max_depth or 0))
    parent_id = metadata.get("model_id") or metadata.get("repo_id") or "parent-bom"
    visited: Set[Tuple[str, str]] = {_visit_key(bom_type, parent_id)}

    generated: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []

    # Frontier of (parent_metadata, parent_target_label, parent_bom_type, current_depth)
    frontier: List[Tuple[Dict[str, Any], str, str, int]] = []
    if max_depth > 0 and bom_type == "ai":
        frontier.append((metadata, parent_id, "ai", 0))

    tree_exhausted = True

    while frontier:
        parent_meta, parent_label, parent_bom_type, depth = frontier.pop(0)
        if depth >= max_depth:
            # We could still discover more but are truncating
            targets, audit = discover_recursive_targets(parent_meta, bom_type=parent_bom_type)
            if targets:
                tree_exhausted = False
            for skip in audit.get("skipped_due_to_conflict", []):
                skipped.append({**skip, "parent": parent_label, "depth": depth + 1})
            continue

        targets, audit = discover_recursive_targets(parent_meta, bom_type=parent_bom_type)
        for skip in audit.get("skipped_due_to_conflict", []):
            skipped.append({**skip, "parent": parent_label, "depth": depth + 1})

        for t in targets:
            key = _visit_key(t["bom_type"], t["target"])
            if key in visited:
                duplicates.append({
                    "target": t["target"],
                    "bom_type": t["bom_type"],
                    "relationship_type": t["relationship_type"],
                    "parent": parent_label,
                    "depth": depth + 1,
                })
                continue
            visited.add(key)
            t_with_parent = {**t, "parent": parent_label}

            enriched = False
            enrichment_error: Optional[str] = None
            child_metadata: Dict[str, Any]
            if enrich_fn is not None:
                try:
                    enriched_metadata = enrich_fn(t_with_parent)
                except Exception as exc:
                    enriched_metadata = None
                    enrichment_error = str(exc)
                if enriched_metadata:
                    child_metadata = enriched_metadata
                    enriched = True
                else:
                    child_metadata = _build_child_metadata(t_with_parent)
            else:
                child_metadata = _build_child_metadata(t_with_parent)

            node = _build_node(
                t_with_parent, child_metadata, depth + 1,
                validate_spdx, strict_spdx,
                enrichment_error=enrichment_error, enriched=enriched,
            )
            generated.append(node)

            # Only AI children carry relationship fields worth descending into.
            if t["bom_type"] == "ai":
                frontier.append((child_metadata, t["target"], "ai", depth + 1))

    deepest = max((n["depth"] for n in generated), default=0)
    return {
        "beta": True,
        "enabled": True,
        "max_depth": max_depth,
        "deepest_level_reached": deepest,
        "tree_exhausted": tree_exhausted,
        "strategy": "conflict-gated dependency-tree recursion",
        "generated_count": len(generated),
        "generated": generated,
        "skipped_due_to_conflict": skipped,
        "duplicates": duplicates,
        "visited": sorted(f"{bt}:{name}" for bt, name in visited),
        "warnings": [
            "Recursive BOM generation is beta.",
            "Each level walks the unique-target set: trainedOn/testedOn "
            "produce data BOM leaves; modelLineage produces AI BOM nodes "
            "that may themselves have dependencies.",
            "Fields with internal/external conflicts are skipped; resolve "
            "the conflict in the parent BOM before recursing.",
            "Without an enrich callback, children only carry seed metadata "
            "and the tree usually terminates after one level. Provide a "
            "real enricher to walk the full dependency tree.",
        ],
    }


# ---------------------------------------------------------------------------
# Linked SPDX bundle
# ---------------------------------------------------------------------------


def build_linked_spdx_bundle(
    parent_metadata: Dict[str, Any],
    recursive_result: Dict[str, Any],
    bom_type: str = "ai",
) -> Dict[str, Any]:
    """Combine the parent BOM and all recursive children into a single
    spec-clean SPDX 3.0.1 JSON-LD document.

    The returned dict has only ``@context`` and ``@graph`` — the SPDX 3.0.1
    JSON Schema rejects unknown root keys, so AIkaBoOM-private metadata is
    available separately via :func:`linked_bundle_summary`.

    The merged ``@graph`` contains:
      * the parent SPDX elements (CreationInfo, Person, Organization,
        SpdxDocument, Bom, AIPackage/DatasetPackage, license),
      * every child element from each recursive node, and
      * a Relationship element per parent→child edge in the dependency
        tree, using the SPDX 3.0.1 vocab (``trainedOn``, ``testedOn``,
        ``dependsOn``). Stub packages auto-emitted by the parent SPDX
        validator are suppressed when a recursive child covers the same
        target so the merged graph is properly de-duplicated.
    """
    from aikaboom.utils.spdx_validator import SPDXValidator

    parent_spdx = SPDXValidator(bom_type=bom_type).validate_and_convert(parent_metadata)

    # Suppress the parent's auto-generated stub DatasetPackages (and the
    # relationships pointing at them) when a recursive child already covers
    # that target — the recursive child carries richer metadata so it is
    # the canonical node for that name in the linked bundle.
    suppressed_names = {
        str(n["target"]).strip().lower() for n in recursive_result.get("generated", [])
    }
    stub_ids_to_drop = set()
    for elem in parent_spdx.get("@graph", []):
        if elem.get("type") == "dataset_DatasetPackage":
            name = str(elem.get("name") or "").strip().lower()
            if name in suppressed_names:
                stub_ids_to_drop.add(elem.get("spdxId") or elem.get("@id"))

    graph: List[Dict[str, Any]] = []
    for elem in parent_spdx.get("@graph", []):
        sid = elem.get("spdxId") or elem.get("@id")
        if sid in stub_ids_to_drop:
            continue
        if elem.get("type") == "Relationship":
            tos = elem.get("to") or []
            if any(t in stub_ids_to_drop for t in tos):
                continue
        graph.append(elem)

    # Map (bom_type, target_name_lower) -> root spdxId in the merged graph,
    # so child→grandchild relationships resolve correctly.
    root_id_by_target: Dict[Tuple[str, str], str] = {}
    parent_root = _root_package_id(parent_spdx)
    if parent_root is not None:
        parent_label = parent_metadata.get("model_id") or parent_metadata.get("repo_id") or "parent-bom"
        root_id_by_target[_visit_key(bom_type, parent_label)] = parent_root

    seen_node_ids = {e.get("spdxId") or e.get("@id") for e in graph}

    relationships: List[Dict[str, Any]] = []

    parent_creation_id = _creation_info_id(parent_spdx)
    parent_person_id = _first_id_of_type(parent_spdx, "Person")
    parent_org_id = _first_id_of_type(parent_spdx, "Organization")

    for node in recursive_result.get("generated", []):
        spdx_doc = node.get("spdx_data")
        if not isinstance(spdx_doc, dict):
            continue
        child_root = _root_package_id(spdx_doc)
        if child_root is None:
            continue
        root_id_by_target[_visit_key(node["bom_type"], node["target"])] = child_root

        # The child SPDX has its own CreationInfo / Person / Organization
        # / SpdxDocument / Bom which we don't want to duplicate. Skip them
        # but rebind every reference to those IDs in the rest of the
        # child's graph onto the parent's equivalents so the merged graph
        # stays referentially intact (otherwise SHACL flags the package's
        # originatedBy / suppliedBy as pointing at undeclared resources).
        rebind = {}
        child_creation_id = _creation_info_id(spdx_doc)
        if parent_creation_id and child_creation_id and child_creation_id != parent_creation_id:
            rebind[child_creation_id] = parent_creation_id
        child_person_id = _first_id_of_type(spdx_doc, "Person")
        if parent_person_id and child_person_id and child_person_id != parent_person_id:
            rebind[child_person_id] = parent_person_id
        child_org_id = _first_id_of_type(spdx_doc, "Organization")
        if parent_org_id and child_org_id and child_org_id != parent_org_id:
            rebind[child_org_id] = parent_org_id

        for elem in spdx_doc.get("@graph", []):
            t = elem.get("type")
            if t in {"CreationInfo", "Person", "Organization", "SpdxDocument", "Bom"}:
                continue
            sid = elem.get("spdxId") or elem.get("@id")
            if sid in seen_node_ids:
                continue
            seen_node_ids.add(sid)
            graph.append(_rebind_refs(elem, rebind))

    # Emit relationship objects for each parent->child edge.
    for node in recursive_result.get("generated", []):
        from_key = _visit_key("ai", node.get("parent", ""))
        from_id = root_id_by_target.get(from_key) or parent_root
        to_key = _visit_key(node["bom_type"], node["target"])
        to_id = root_id_by_target.get(to_key)
        if not from_id or not to_id:
            continue
        rel_id = f"urn:spdx:Relationship-{node['relationship_type']}-{_safe_id(node['target'])}-d{node['depth']}"
        if rel_id in seen_node_ids:
            continue
        seen_node_ids.add(rel_id)
        relationships.append({
            "type": "Relationship",
            "spdxId": rel_id,
            "creationInfo": parent_creation_id,
            "relationshipType": node["relationship_type"],
            "from": from_id,
            "to": [to_id],
            "description": (
                f"{node['relationship_type']} relationship from {node['parent']} "
                f"to {node['target']} (depth {node['depth']})"
            ),
        })

    return {
        "@context": parent_spdx.get("@context"),
        "@graph": graph + relationships,
    }


def linked_bundle_summary(
    linked_bundle: Dict[str, Any],
    recursive_result: Dict[str, Any],
) -> Dict[str, Any]:
    """AIkaBoOM-private sidecar metadata for a linked SPDX bundle.

    Kept out of the SPDX document itself because the official SPDX 3.0.1
    JSON Schema rejects unknown root keys. Use this for UI summaries,
    download manifests, and tests.
    """
    graph = linked_bundle.get("@graph", []) or []
    recursive_edges = [
        e for e in graph
        if isinstance(e, dict)
        and e.get("type") == "Relationship"
        and e.get("relationshipType") in {"trainedOn", "testedOn", "dependsOn"}
        and isinstance(e.get("spdxId"), str)
        and e.get("spdxId", "").startswith("urn:spdx:Relationship-")
        and re.search(r"-d\d+$", e.get("spdxId", "")) is not None
    ]
    return {
        "beta": True,
        "node_count": len(graph),
        "recursive_edge_count": len(recursive_edges),
        "deepest_level_reached": recursive_result.get("deepest_level_reached", 0),
        "tree_exhausted": recursive_result.get("tree_exhausted", True),
    }


def _root_package_id(spdx_doc: Dict[str, Any]) -> Optional[str]:
    for elem in spdx_doc.get("@graph", []):
        if elem.get("type") in {"ai_AIPackage", "dataset_DatasetPackage"}:
            return elem.get("spdxId") or elem.get("@id")
    return None


def _creation_info_id(spdx_doc: Dict[str, Any]) -> Optional[str]:
    for elem in spdx_doc.get("@graph", []):
        if elem.get("type") == "CreationInfo":
            return elem.get("spdxId") or elem.get("@id")
    return None


def _first_id_of_type(spdx_doc: Dict[str, Any], type_name: str) -> Optional[str]:
    for elem in spdx_doc.get("@graph", []):
        if elem.get("type") == type_name:
            return elem.get("spdxId") or elem.get("@id")
    return None


def _rebind_refs(elem: Dict[str, Any], rebind: Dict[str, str]) -> Dict[str, Any]:
    """Return a copy of ``elem`` with every string/list-of-strings value
    rewritten through ``rebind`` (other types are left alone).
    """
    if not rebind:
        return dict(elem)
    out: Dict[str, Any] = {}
    for k, v in elem.items():
        if isinstance(v, str) and v in rebind:
            out[k] = rebind[v]
        elif isinstance(v, list):
            out[k] = [rebind.get(item, item) if isinstance(item, str) else item for item in v]
        else:
            out[k] = v
    return out
