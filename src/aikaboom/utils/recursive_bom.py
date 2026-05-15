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
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from aikaboom.utils.pipeline_events import emit as _emit_event


# Order matters: discovery emits targets in this order and the global
# safety cap is applied first-come-first-served. modelLineage is listed
# first because the dependsOn edge is the only recursable one (it spawns
# AI children that walk deeper) and must never be starved by a large
# trainedOn/testedOn dataset fan-out — dataset edges are leaves.
AI_RELATIONSHIP_FIELDS = {
    "modelLineage": ("ai", "dependsOn"),
    "trainedOnDatasets": ("data", "trainedOn"),
    "testedOnDatasets": ("data", "testedOn"),
}

# Dataset BOMs walk their upstream lineage via ``sourceInfo`` (which captures
# parent / aggregated-from / derived-from datasets — see
# question_bank/data/sourceInfo.json). The relationship maps to SPDX
# ``dependsOn`` because a derived dataset depends on its upstream sources.
DATA_RELATIONSHIP_FIELDS = {
    "sourceInfo": ("data", "dependsOn"),
}


def _is_walkable_target(text: str) -> bool:
    """Filter out values that look like paper/URL references rather than
    walkable dataset/model identifiers.

    sourceInfo commonly mixes dataset names with arXiv paper refs
    (``arXiv:2108.07732``) and bare DOIs/IDs (``2108.07732``); those
    should not become child BOMs because they are not retrievable by the
    same enrichment pipeline.
    """
    t = text.strip().lower()
    if not t:
        return False
    if t.startswith(("arxiv:", "arxiv.org", "doi:", "http://", "https://")):
        return False
    # Bare arXiv-style IDs like "2108.07732" or "2108.07732v2".
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", t):
        return False
    return True


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


def _split_targets(value: Any, parent_identifier: Optional[str] = None) -> List[str]:
    """Parse a relationship-target value into deduplicated unique targets.

    Strings go through the shared :func:`aikaboom.utils.lineage.split_lineage_targets`
    helper (Phase 10 / Finding #17) so arrow-typed lineage strings
    (``"A -> B"``) split into separate targets the same way the SPDX
    builder does. List/tuple/dict inputs (rare; from upstream code that
    already pre-split) are walked element-by-element with the same nil-
    sentinel + dedupe + self-loop filter.
    """
    from aikaboom.utils.lineage import split_lineage_targets
    from aikaboom.utils.value_helpers import _is_nil_value

    value = _extract_value(value)
    if value is None:
        return []

    parent_lower = (parent_identifier or "").strip().lower()

    if isinstance(value, str):
        # Shared helper handles separators, nil-filter, dedupe, self-loop.
        return split_lineage_targets(value, parent_identifier)

    if isinstance(value, (list, tuple, set)):
        pieces: Iterable[Any] = value
    elif isinstance(value, dict):
        pieces = value.values()
    else:
        return split_lineage_targets(str(value), parent_identifier)

    targets: List[str] = []
    seen: set = set()
    for piece in pieces:
        text = str(_extract_value(piece) or "").strip()
        text = re.sub(r"^\s*[-*]\s*", "", text)
        low = text.lower()
        if not text or _is_nil_value(low) or low == parent_lower or low in seen:
            continue
        seen.add(low)
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
    """Find relationship targets that can seed child BOMs.

    Returns ``(targets, audit)``. Every target the parent reports is
    returned — including ones whose source field carries a conflict.
    Conflict-flagged targets are tagged with ``has_conflict=True`` and a
    ``conflict`` dict so downstream consumers (UI, CLI) can mark the edge
    without truncating the walk. ``audit['conflict_flagged']`` lists the
    fields that surfaced any conflicts (useful for summary text).

    This contract changed in the recursive-progress branch: the walker
    used to skip-and-continue on any conflict, which made it impossible
    to reach depth 3+ whenever a depth-2 model's ``modelLineage`` was
    auditor-flagged. Conflicts are now per-edge metadata, not a gate.
    """
    audit: Dict[str, Any] = {
        "conflict_flagged": [],
        # Back-compat alias for older callers (CLI, third-party readers
        # of the recursive result). The new walker no longer actually
        # skips on conflict — these are walked-with-conflict edges.
        "skipped_due_to_conflict": [],
        "considered": [],
    }
    if bom_type == "ai":
        relationship_map = AI_RELATIONSHIP_FIELDS
    elif bom_type == "data":
        relationship_map = DATA_RELATIONSHIP_FIELDS
    else:
        audit["reason"] = f"recursion not supported for bom_type={bom_type!r}"
        return [], audit

    # Dataset BOMs surface their RAG fields under ``rag_metadata`` (legacy
    # processor key) or ``rag_fields`` (modern key); accept both.
    rag_fields = metadata.get("rag_fields") or metadata.get("rag_metadata") or {}
    # Prefer ``repo_id`` (canonical ``owner/name`` form) over ``model_id``
    # (filename-safe slug ``owner_name``). modelLineage triplets surface in
    # canonical form, so the self-loop visit-key comparison fails when the
    # parent is recorded under the slug. For datasets, the analogous keys
    # are ``hf_url`` / ``dataset_id``.
    parent_id = (
        metadata.get("repo_id")
        or metadata.get("model_id")
        or metadata.get("dataset_id")
        or metadata.get("hf_url")
        or "parent-bom"
    )
    targets: List[Dict[str, Any]] = []

    for field, (child_bom_type, relationship_type) in relationship_map.items():
        triplet = rag_fields.get(field)
        audit["considered"].append(field)

        conflict = _conflict_of(triplet)
        if conflict is not None:
            flagged_entry = {
                "field": field,
                "relationship_type": relationship_type,
                "reason": "conflict-detected",
                "conflict": conflict,
            }
            audit["conflict_flagged"].append(flagged_entry)
            audit["skipped_due_to_conflict"].append(flagged_entry)

        for target in _split_targets(triplet, parent_identifier=parent_id):
            if not _is_walkable_target(target):
                continue
            targets.append({
                "source_field": field,
                "relationship_type": relationship_type,
                "target": target,
                "bom_type": child_bom_type,
                "parent": parent_id,
                "resolvable_hint": "/" in target and " " not in target,
                # Per-edge conflict marker so the UI can render a ⚠ badge
                # and downstream BOM consumers can audit the lineage.
                "has_conflict": conflict is not None,
                "conflict": conflict,
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


EXHAUST_DEPTH = sys.maxsize  # sentinel for "walk until the frontier empties"


def generate_recursive_boms(
    metadata: Dict[str, Any],
    bom_type: str = "ai",
    max_depth: int = 1,
    safety_cap: int = 50,
    validate_spdx: bool = True,
    strict_spdx: bool = False,
    enrich_fn: Optional[EnrichFn] = None,
) -> Dict[str, Any]:
    """Walk the dependency tree of an AI BOM and emit child BOMs.

    Args:
        metadata: Parent BOM metadata dict (with ``rag_fields``).
        bom_type: Parent BOM type. Recursion only descends through AI BOMs;
            ``data`` parents are leaves.
        max_depth: Maximum tree depth (1 = direct children only). Pass
            :data:`EXHAUST_DEPTH` to walk until the frontier empties or the
            ``safety_cap`` is reached. Recursion also stops naturally when
            the unique-target set is exhausted.
        safety_cap: Maximum number of child nodes to materialize before
            stopping. Prevents runaway walks under :data:`EXHAUST_DEPTH`.
            Default 50.
        validate_spdx: Validate each generated child SPDX export.
        strict_spdx: Use the SHACL strict pass (beta).
        enrich_fn: Optional callable ``(target_dict) -> metadata_dict`` that
            fetches full metadata for a discovered target. Without it,
            children carry only seed metadata and recursion typically stops
            after one level.
    """
    max_depth = max(0, int(max_depth or 0))
    safety_cap = max(0, int(safety_cap or 0))
    # Prefer ``repo_id`` (canonical ``owner/name`` form) over ``model_id``
    # (filename-safe slug ``owner_name``). modelLineage triplets surface in
    # canonical form, so the self-loop visit-key comparison fails when the
    # parent is recorded under the slug. Datasets surface under
    # ``dataset_id`` / ``hf_url``.
    parent_id = (
        metadata.get("repo_id")
        or metadata.get("model_id")
        or metadata.get("dataset_id")
        or metadata.get("hf_url")
        or "parent-bom"
    )
    visited: Set[Tuple[str, str]] = {_visit_key(bom_type, parent_id)}

    generated: List[Dict[str, Any]] = []
    # ``safety_capped`` is targets dropped because the cap was hit; the old
    # name ``skipped`` lumped these in with conflict-skipped fields. With
    # conflicts now walked-through, this list is cleanly cap-only.
    safety_capped: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []

    # Frontier of (parent_metadata, parent_target_label, parent_bom_type, current_depth)
    frontier: List[Tuple[Dict[str, Any], str, str, int]] = []
    if max_depth > 0 and bom_type in ("ai", "data"):
        frontier.append((metadata, parent_id, bom_type, 0))

    tree_exhausted = True

    # Signal the start of the recursive walk so the web UI can reveal the
    # Stage 2 card. The exact frontier size isn't known yet — children
    # appear via target.discovered events as parents complete.
    _recursive_t0 = time.time()
    _emit_event({
        "event": "recursive.start",
        "parent": parent_id,
        "bom_type": bom_type,
        "max_depth": max_depth,
        "safety_cap": safety_cap,
    })

    conflict_walked: List[Dict[str, Any]] = []

    while frontier:
        parent_meta, parent_label, parent_bom_type, depth = frontier.pop(0)
        if depth >= max_depth:
            # We could still discover more but are truncating
            targets, audit = discover_recursive_targets(parent_meta, bom_type=parent_bom_type)
            if targets:
                tree_exhausted = False
            for fld in audit.get("conflict_flagged", []):
                conflict_walked.append({**fld, "parent": parent_label, "depth": depth + 1, "truncated": True})
            continue

        targets, audit = discover_recursive_targets(parent_meta, bom_type=parent_bom_type)
        # Record fields that surfaced conflicts (walked anyway — kept here
        # for the summary/audit; no longer drives skip behaviour).
        for fld in audit.get("conflict_flagged", []):
            conflict_walked.append({**fld, "parent": parent_label, "depth": depth + 1})

        # Two-phase walk per parent:
        #   Phase 1 — discovery: classify every target (dup / cap / valid)
        #     and emit one event per target so the UI can render the full
        #     set of pending chips for this parent BEFORE any enrichment
        #     starts.
        #   Phase 2 — processing: enrich each valid target serially,
        #     emitting child.start before and child.done after.
        # Splitting these phases is what lets the Stage 2 tree show "0/3
        # processed" up front and then watch the chips turn amber → green
        # as each child resolves, instead of chips popping into existence
        # already-done.
        to_process: List[Dict[str, Any]] = []
        for t in targets:
            if len(generated) + len(to_process) >= safety_cap:
                # The cap is on the total *generated* count, but during
                # discovery we don't know which targets will actually be
                # enriched yet. Approximating with len(generated) + queued
                # to-process keeps the prior semantics intact while still
                # rejecting targets eagerly so the UI sees skipped chips
                # immediately rather than after all earlier siblings run.
                tree_exhausted = False
                safety_capped.append({
                    "target": t["target"],
                    "bom_type": t["bom_type"],
                    "relationship_type": t["relationship_type"],
                    "reason": "safety-cap-reached",
                    "parent": parent_label,
                    "depth": depth + 1,
                })
                _emit_event({
                    "event": "recursive.child.skipped",
                    "target": t["target"],
                    "bom_type": t["bom_type"],
                    "relationship_type": t["relationship_type"],
                    "parent": parent_label,
                    "depth": depth + 1,
                    "reason": "safety-cap-reached",
                    "has_conflict": t.get("has_conflict", False),
                })
                continue
            key = _visit_key(t["bom_type"], t["target"])
            if key in visited:
                duplicates.append({
                    "target": t["target"],
                    "bom_type": t["bom_type"],
                    "relationship_type": t["relationship_type"],
                    "parent": parent_label,
                    "depth": depth + 1,
                })
                _emit_event({
                    "event": "recursive.child.skipped",
                    "target": t["target"],
                    "bom_type": t["bom_type"],
                    "relationship_type": t["relationship_type"],
                    "parent": parent_label,
                    "depth": depth + 1,
                    "reason": "duplicate",
                    "has_conflict": t.get("has_conflict", False),
                })
                continue
            visited.add(key)
            # Emit the discovery event immediately so the UI's pending
            # chip for this target shows up before any of its siblings
            # start running.
            _emit_event({
                "event": "recursive.target.discovered",
                "target": t["target"],
                "bom_type": t["bom_type"],
                "relationship_type": t["relationship_type"],
                "parent": parent_label,
                "depth": depth + 1,
                "has_conflict": t.get("has_conflict", False),
            })
            to_process.append(t)

        # Phase 2 — process each valid target in BFS-sibling order.
        for t in to_process:
            t_with_parent = {**t, "parent": parent_label}
            _emit_event({
                "event": "recursive.child.start",
                "target": t["target"],
                "bom_type": t["bom_type"],
                "relationship_type": t["relationship_type"],
                "parent": parent_label,
                "depth": depth + 1,
                "has_conflict": t.get("has_conflict", False),
            })
            _child_t0 = time.time()

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

            _emit_event({
                "event": "recursive.child.done",
                "target": t["target"],
                "bom_type": t["bom_type"],
                "relationship_type": t["relationship_type"],
                "parent": parent_label,
                "depth": depth + 1,
                "enriched": enriched,
                "error": enrichment_error,
                "has_conflict": t.get("has_conflict", False),
                "duration_ms": int((time.time() - _child_t0) * 1000),
            })

            # Only AI children carry relationship fields worth descending into.
            if t["bom_type"] == "ai":
                frontier.append((child_metadata, t["target"], "ai", depth + 1))

    deepest = max((n["depth"] for n in generated), default=0)
    _emit_event({
        "event": "recursive.done",
        "parent": parent_id,
        "duration_ms": int((time.time() - _recursive_t0) * 1000),
        "generated_count": len(generated),
        "conflict_walked_count": len(conflict_walked),
        "safety_capped_count": len(safety_capped),
        "duplicate_count": len(duplicates),
        "deepest_level_reached": deepest,
        "tree_exhausted": tree_exhausted,
    })
    return {
        "beta": True,
        "enabled": True,
        "max_depth": max_depth,
        "deepest_level_reached": deepest,
        "tree_exhausted": tree_exhausted,
        "strategy": "conflict-tagged dependency-tree recursion",
        "generated_count": len(generated),
        "generated": generated,
        # Edges that were walked despite a parent-field conflict — UI shows
        # a ⚠ on these chips; consumers can audit the lineage. Kept under
        # the old key name (skipped_due_to_conflict) as an alias for
        # back-compat with the CLI summary line; semantically it now means
        # "conflict-flagged walked edges" rather than "skipped".
        "conflict_walked": conflict_walked,
        "skipped_due_to_conflict": conflict_walked,
        "safety_capped": safety_capped,
        "duplicates": duplicates,
        "visited": sorted(f"{bt}:{name}" for bt, name in visited),
        "warnings": [
            "Recursive BOM generation is beta.",
            "Each level walks the unique-target set: trainedOn/testedOn "
            "produce data BOM leaves; modelLineage produces AI BOM nodes "
            "that may themselves have dependencies.",
            "Edges from conflict-flagged parent fields are walked anyway "
            "and tagged with has_conflict=True so downstream consumers "
            "can flag them; resolve the parent conflict to lock the edge.",
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
    # The parent SPDX auto-emits a stub for every relationship target:
    # dataset_DatasetPackage stubs for trainedOn/testedOn and ai_AIPackage
    # stubs for modelLineage/dependsOn. Both kinds must be suppressed when
    # a recursive child covers the same target — but never the parent's
    # own root package, which is also an ai_AIPackage.
    parent_root_id = _root_package_id(parent_spdx)
    stub_ids_to_drop = set()
    for elem in parent_spdx.get("@graph", []):
        if elem.get("type") in ("dataset_DatasetPackage", "ai_AIPackage"):
            sid = elem.get("spdxId") or elem.get("@id")
            if sid == parent_root_id:
                continue
            name = str(elem.get("name") or "").strip().lower()
            if name in suppressed_names:
                stub_ids_to_drop.add(sid)

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
        parent_label = parent_metadata.get("repo_id") or parent_metadata.get("model_id") or "parent-bom"
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
