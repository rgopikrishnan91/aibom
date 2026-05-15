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
