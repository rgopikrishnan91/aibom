"""Read-side graph queries for the worldofBOMs browser visualization.

Pure functions over a `BomStore`. `web/app.py` calls these; all graph
SPARQL lives here so the Flask layer stays thin.
"""

from __future__ import annotations

import re as _re
import warnings

from aikaboom.store import vocab
from aikaboom.store.store import _validate_sparql_iri

# Artifact-to-artifact edge predicates we render.
_EDGE_PREDICATES = {
    str(vocab.trainedOn): "trainedOn",
    str(vocab.testedOn): "testedOn",
    str(vocab.dependsOn): "dependsOn",
}

_KIND_BY_CLASS = {
    str(vocab.Model): "Model",
    str(vocab.Dataset): "Dataset",
    str(vocab.Paper): "Paper",
    str(vocab.CodeRepo): "CodeRepo",
}


def _node_rows(store) -> list[dict]:
    """Every Artifact as a node dict."""
    rows = store._backend.select(
        f"""
        SELECT ?artifact ?label ?placeholder WHERE {{
            ?artifact a <{vocab.Artifact}> .
            OPTIONAL {{ ?artifact <{vocab.canonicalLabel}> ?label . }}
            OPTIONAL {{ ?artifact <{vocab.isPlaceholder}> ?placeholder . }}
        }}
        """
    )
    out: dict[str, dict] = {}
    for row in rows:
        iri = str(row["artifact"])
        if iri in out:
            continue
        out[iri] = {
            "iri": iri,
            "label": str(row.get("label") or iri.rsplit("/", 1)[-1]),
            "is_placeholder": bool(row.get("placeholder")),
            "kind": _kind_for(store, iri),
            "claim_count": _claim_count(store, iri),
        }
    return list(out.values())


def _kind_for(store, artifact_iri: str) -> str:
    iri = _validate_sparql_iri(artifact_iri)
    for row in store._backend.select(f"SELECT ?c WHERE {{ <{iri}> a ?c }}"):
        kind = _KIND_BY_CLASS.get(str(row["c"]))
        if kind:
            return kind
    return "Artifact"


def _claim_count(store, artifact_iri: str) -> int:
    iri = _validate_sparql_iri(artifact_iri)
    rows = list(store._backend.select(
        f"""
        SELECT (COUNT(?claim) AS ?n) WHERE {{
            <{iri}> <{vocab.hasVersion}> ?v . ?v <{vocab.hasClaim}> ?claim .
        }}
        """
    ))
    return int(rows[0]["n"]) if rows and rows[0].get("n") is not None else 0


def _edge_rows(store) -> list[dict]:
    """Every artifact-to-artifact relationship edge."""
    out: list[dict] = []
    for pred_uri, pred_name in _EDGE_PREDICATES.items():
        for row in store._backend.select(
            f"SELECT ?s ?t WHERE {{ ?s <{pred_uri}> ?t }}"
        ):
            out.append({"source": str(row["s"]), "target": str(row["t"]),
                        "predicate": pred_name})
    return out


def edge_count(store) -> int:
    """Total number of artifact-to-artifact relationship edges."""
    return len(_edge_rows(store))


def full_graph(store) -> dict:
    """The whole graph: every artifact node and every relationship edge."""
    return {"nodes": _node_rows(store), "edges": _edge_rows(store)}


def ego_graph(store, artifact_iri: str, direction: str = "both",
              depth: int | None = None) -> dict:
    """Breadth-first ego subgraph around `artifact_iri`.

    direction: "up" follows edges forward (dependencies), "down" follows
    them backward (dependents), "both" does the union. `depth` caps the
    hop count; None means unlimited (full lineage).
    """
    focus = _validate_sparql_iri(artifact_iri)
    if direction not in ("up", "down", "both"):
        raise ValueError(
            f"direction must be 'up', 'down', or 'both', got {direction!r}"
        )
    all_nodes = {n["iri"]: n for n in _node_rows(store)}
    all_edges = _edge_rows(store)
    forward: dict[str, list[dict]] = {}
    backward: dict[str, list[dict]] = {}
    for e in all_edges:
        forward.setdefault(e["source"], []).append(e)
        backward.setdefault(e["target"], []).append(e)

    keep_nodes: set[str] = {focus}
    keep_edges: list[dict] = []
    seen_edges: set[tuple] = set()
    frontier = [focus]
    hops = 0
    while frontier and (depth is None or hops < depth):
        nxt: list[str] = []
        for node in frontier:
            steps: list[tuple[dict, str]] = []
            if direction in ("up", "both"):
                steps += [(e, e["target"]) for e in forward.get(node, [])]
            if direction in ("down", "both"):
                steps += [(e, e["source"]) for e in backward.get(node, [])]
            for edge, other in steps:
                key = (edge["source"], edge["predicate"], edge["target"])
                if key not in seen_edges:
                    seen_edges.add(key)
                    keep_edges.append(edge)
                if other not in keep_nodes:
                    keep_nodes.add(other)
                    nxt.append(other)
        frontier = nxt
        hops += 1

    nodes = [all_nodes[iri] for iri in keep_nodes if iri in all_nodes]
    return {"nodes": nodes, "edges": keep_edges, "focus": focus}


_PRESETS = ("licenses", "datasets", "models", "conflicts")


def lineage_query(store, artifact_iri: str, preset: str,
                  direction: str = "both") -> list[dict]:
    """Run one preset query over the ego node set of `artifact_iri`.

    presets: "licenses", "datasets", "models", "conflicts".
    """
    if preset not in _PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {_PRESETS}")
    ego = ego_graph(store, artifact_iri, direction=direction, depth=None)
    nodes = ego["nodes"]

    if preset == "datasets":
        return [{"iri": n["iri"], "label": n["label"]}
                for n in nodes if n["kind"] == "Dataset"]
    if preset == "models":
        return [{"iri": n["iri"], "label": n["label"]}
                for n in nodes if n["kind"] == "Model"]
    if preset == "licenses":
        out: list[dict] = []
        for n in nodes:
            for lic in _licenses_for(store, n["iri"]):
                out.append({"artifact": n["label"], "license": lic})
        return out
    # conflicts
    out = []
    for n in nodes:
        for kind in _conflicts_for(store, n["iri"]):
            out.append({"artifact": n["label"], "conflict": kind})
    return out


def _licenses_for(store, artifact_iri: str) -> list[str]:
    """License values from the artifact's canonical-claim field literals."""
    iri = _validate_sparql_iri(artifact_iri)
    rows = store._backend.select(
        f"""
        SELECT DISTINCT ?lic WHERE {{
            <{iri}> <{vocab.hasVersion}> ?v . ?v <{vocab.hasClaim}> ?claim .
            ?claim <{vocab.AIBOM}licenseName> ?lic .
        }}
        """
    )
    return [str(r["lic"]) for r in rows]


def _conflicts_for(store, artifact_iri: str) -> list[str]:
    """Conflict kinds annotated on the artifact's claims."""
    iri = _validate_sparql_iri(artifact_iri)
    rows = store._backend.select(
        f"""
        SELECT DISTINCT ?kind WHERE {{
            <{iri}> <{vocab.hasVersion}> ?v . ?v <{vocab.hasClaim}> ?claim .
            ?ann <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject> ?claim ;
                 <{vocab.conflictKind}> ?kind .
            FILTER(?kind != <{vocab.noConflict}>)
        }}
        """
    )
    return [str(r["kind"]).rsplit("#", 1)[-1] for r in rows]


def canonical_claim_iri(store, artifact_iri: str) -> str | None:
    """Return the IRI of the most-recent claim for an artifact, or None."""
    iri = _validate_sparql_iri(artifact_iri)
    rows = list(store._backend.select(
        f"""
        SELECT ?claim ?createdAt WHERE {{
            <{iri}> <{vocab.hasVersion}> ?v . ?v <{vocab.hasClaim}> ?claim .
            OPTIONAL {{ ?claim <{vocab.createdAt}> ?createdAt . }}
        }}
        ORDER BY DESC(?createdAt)
        """
    ))
    return str(rows[0]["claim"]) if rows else None


def _bom_type_for_kind(kind: str) -> str:
    """Map graph node kind to recursive_bom BOM type string."""
    if kind == "Dataset":
        return "data"
    # Model, Paper, CodeRepo, Artifact — treat as "ai"
    return "ai"


def _relationship_type_for_predicate(predicate: str) -> str:
    """Map SPDX edge predicate name to SPDX 3.0.1 relationship type."""
    mapping = {
        "trainedOn": "trainedOn",
        "testedOn": "testedOn",
        "dependsOn": "dependsOn",
    }
    return mapping.get(predicate, "dependsOn")


def _seed_bom_for_kind(label: str, kind: str) -> dict:
    """Build minimal seed BOM metadata that SPDXValidator can convert.

    For artifacts that have no stored claim (e.g. placeholders), we need
    just enough fields to produce a valid SPDX package element.
    """
    if kind == "Dataset":
        safe_id = _re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip()).strip("-") or "dataset"
        return {
            "dataset_id": safe_id,
            "direct_metadata": {
                "name": label,
                "license": "NOASSERTION",
            },
            "rag_metadata": {},
        }
    # AI / Model / other
    safe_id = _re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip()).strip("-") or "model"
    return {
        "model_id": safe_id,
        "repo_id": label,
        "direct_fields": {"license": "NOASSERTION"},
        "rag_fields": {"model_name": label},
    }


def ego_spdx_bundle(store, artifact_iri: str | None,
                    direction: str = "both") -> dict:
    """Assemble an SPDX 3.0.1 linked bundle for an ego view (or the whole graph).

    ``artifact_iri=None`` exports every artifact in the store. Otherwise
    exports the ego subgraph for the given direction. Each member's canonical
    BOM is reconstructed (or a seed BOM built for placeholders) and converted
    to SPDX, then merged by ``build_linked_spdx_bundle``.
    """
    from aikaboom.utils.recursive_bom import build_linked_spdx_bundle
    from aikaboom.utils.spdx_validator import SPDXValidator

    if artifact_iri is None:
        all_nodes = _node_rows(store)
        members = all_nodes
        focus_node = members[0] if members else None
        focus_iri = focus_node["iri"] if focus_node else None
    else:
        ego = ego_graph(store, artifact_iri, direction=direction, depth=None)
        # Build a dict keyed by IRI so we retain kind info
        all_nodes_map = {n["iri"]: n for n in _node_rows(store)}
        members = [all_nodes_map[n["iri"]] for n in ego["nodes"] if n["iri"] in all_nodes_map]
        focus_iri = artifact_iri
        focus_node = all_nodes_map.get(focus_iri)

    if not focus_iri or not focus_node:
        return {"@context": None, "@graph": []}

    # Build parent BOM metadata
    focus_claim = canonical_claim_iri(store, focus_iri)
    if focus_claim:
        parent_meta = store.reconstruct_bom(focus_claim)
        # Ensure repo_id is present for build_linked_spdx_bundle's parent label
        if not parent_meta.get("repo_id"):
            parent_meta["repo_id"] = focus_node["label"]
    else:
        parent_meta = _seed_bom_for_kind(focus_node["label"], focus_node["kind"])

    parent_label = parent_meta.get("repo_id") or parent_meta.get("model_id") or focus_node["label"]
    parent_bom_type = _bom_type_for_kind(focus_node["kind"])

    # Build ego edges to understand relationships between members
    if artifact_iri is None:
        ego_edges = _edge_rows(store)
    else:
        ego_edges = ego["edges"]

    # For each non-focus member, find the edge connecting it to its parent in
    # the ego graph and build a generated node with the required spdx_data.
    generated = []
    seen_targets: set[str] = set()

    # Build a map of edges: target_iri -> list of (source_iri, predicate)
    # and source_iri -> list of (target_iri, predicate)
    edge_map_fwd: dict[str, list[tuple[str, str]]] = {}  # source -> [(target, pred)]
    edge_map_bwd: dict[str, list[tuple[str, str]]] = {}  # target -> [(source, pred)]
    for e in ego_edges:
        edge_map_fwd.setdefault(e["source"], []).append((e["target"], e["predicate"]))
        edge_map_bwd.setdefault(e["target"], []).append((e["source"], e["predicate"]))

    # Build a node-kind lookup
    node_kind_map = {n["iri"]: n for n in members}

    for node in members:
        member_iri = node["iri"]
        if member_iri == focus_iri:
            continue
        member_label = node["label"]
        target_key = member_label.lower()
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)

        member_bom_type = _bom_type_for_kind(node["kind"])

        # Determine the edge relationship between focus and this member.
        # Look for a direct edge from focus to member (forward) or member to focus (backward).
        rel_type = "dependsOn"
        edge_parent_label = parent_label

        # Try forward: focus -> member
        for tgt_iri, pred in edge_map_fwd.get(focus_iri, []):
            if tgt_iri == member_iri:
                rel_type = _relationship_type_for_predicate(pred)
                break
        else:
            # Try backward: member -> focus
            for src_iri, pred in edge_map_bwd.get(focus_iri, []):
                if src_iri == member_iri:
                    rel_type = _relationship_type_for_predicate(pred)
                    edge_parent_label = member_label
                    break
            else:
                # No direct edge — find any edge involving this member in the ego graph
                for tgt_iri, pred in edge_map_fwd.get(member_iri, []):
                    if tgt_iri in node_kind_map:
                        rel_type = _relationship_type_for_predicate(pred)
                        break

        # Build or reconstruct child BOM metadata
        member_claim = canonical_claim_iri(store, member_iri)
        if member_claim:
            child_bom = store.reconstruct_bom(member_claim)
            if not child_bom.get("repo_id") and not child_bom.get("dataset_id"):
                child_bom["repo_id"] = member_label
        else:
            child_bom = _seed_bom_for_kind(member_label, node["kind"])

        # Convert to SPDX — this is the key field build_linked_spdx_bundle reads.
        try:
            spdx_data = SPDXValidator(bom_type=member_bom_type).validate_and_convert(child_bom)
        except Exception as e:
            warnings.warn(
                f"ego_spdx_bundle: SPDX conversion failed for {member_label!r}; "
                f"falling back to seed BOM. Error: {e}",
                stacklevel=2,
            )
            # Fallback: try seed BOM if reconstruction gave an unusable shape
            seed = _seed_bom_for_kind(member_label, node["kind"])
            spdx_data = SPDXValidator(bom_type=member_bom_type).validate_and_convert(seed)

        # Derive the "target" name that build_linked_spdx_bundle uses for
        # dedup and relationship description — must match the package name.
        child_target = (
            child_bom.get("repo_id")
            or child_bom.get("dataset_id")
            or child_bom.get("model_id")
            or member_label
        )

        generated.append({
            "bom_type": member_bom_type,
            "target": child_target,
            "parent": edge_parent_label,
            "depth": 1,
            "relationship_type": rel_type,
            "metadata": child_bom,
            "spdx_data": spdx_data,
        })

    recursive_result = {
        "generated": generated,
        "deepest_level_reached": 1 if generated else 0,
    }
    return build_linked_spdx_bundle(parent_meta, recursive_result, bom_type=parent_bom_type)


_MUTATION_KEYWORDS = _re.compile(
    r"(?<!\?)\b(INSERT|DELETE|LOAD|CLEAR|DROP|CREATE|ADD|MOVE|COPY)\b", _re.IGNORECASE
)


def raw_query(store, sparql: str) -> list[dict]:
    """Run a read-only SPARQL query. Rejects anything that can mutate the store."""
    if _MUTATION_KEYWORDS.search(sparql or ""):
        raise ValueError("only read-only SELECT/ASK queries are allowed")
    # Strip PREFIX/comment lines, then require SELECT or ASK as the first keyword.
    stripped = "\n".join(
        ln for ln in (sparql or "").splitlines()
        if ln.strip() and not ln.strip().upper().startswith(("PREFIX", "#", "BASE"))
    ).strip()
    if not _re.match(r"(SELECT|ASK)\b", stripped, _re.IGNORECASE):
        raise ValueError("query must start with SELECT or ASK")
    return [{k: str(v) for k, v in row.items()} for row in store._backend.select(sparql)]
