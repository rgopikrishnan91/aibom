"""Read-side graph queries for the worldofBOMs browser visualization.

Pure functions over a `BomStore`. `web/app.py` calls these; all graph
SPARQL lives here so the Flask layer stays thin.
"""

from __future__ import annotations

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

    nodes = [n for n in _node_rows(store) if n["iri"] in keep_nodes]
    return {"nodes": nodes, "edges": keep_edges, "focus": focus}
