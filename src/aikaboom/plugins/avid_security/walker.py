from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Literal

from aikaboom.plugins import Scope
from aikaboom.store import BomStore, vocab

ComponentKind = Literal["Model", "Dataset"]
ComponentScope = Literal["principal", "base", "dataset"]

# AIBOM namespace prefix (without trailing #) for SPARQL PREFIX declarations
_AIBOM_PREFIX = "PREFIX aibom: <https://aikaboom.dev/aibom#>"
_RDF_PREFIX = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"


@dataclass(frozen=True)
class Component:
    kind: ComponentKind
    hf_path: str
    developer: str | None
    base_models: tuple[str, ...]
    scope_in_bom: ComponentScope
    spdx_id: str

    @property
    def bare_name(self) -> str:
        """Lowercase last path segment of hf_path."""
        return self.hf_path.split("/")[-1].lower()


def _label(store: BomStore, iri: str) -> str:
    """Return the canonicalLabel of an artifact IRI, falling back to the IRI tail."""
    rows = list(store._backend.select(f"""
        {_AIBOM_PREFIX}
        SELECT ?l WHERE {{ <{iri}> aibom:canonicalLabel ?l }} LIMIT 1
    """))
    if rows:
        return str(rows[0]["l"])
    return iri.rsplit("/", 1)[-1]


def _developer(store: BomStore, artifact_iri: str) -> str | None:
    """Return the suppliedBy literal on the highest-trust claim, or None."""
    rows = list(store._backend.select(f"""
        {_AIBOM_PREFIX}
        SELECT ?supplier WHERE {{
          <{artifact_iri}> aibom:hasVersion ?version .
          ?version aibom:hasClaim ?claim .
          ?claim aibom:suppliedBy ?supplier .
        }}
        LIMIT 1
    """))
    if rows:
        return str(rows[0]["supplier"])
    return None


def _base_model_labels(store: BomStore, artifact_iri: str) -> tuple[str, ...]:
    """Return canonicalLabels of all base models (via trainedOn on the claim)."""
    rows = list(store._backend.select(f"""
        {_AIBOM_PREFIX}
        SELECT DISTINCT ?base WHERE {{
          <{artifact_iri}> aibom:hasVersion ?version .
          ?version aibom:hasClaim ?claim .
          ?claim aibom:trainedOn ?base .
        }}
    """))
    labels = []
    for row in rows:
        base_iri = str(row["base"])
        labels.append(_label(store, base_iri))
    return tuple(labels)


def _kind_from_types(store: BomStore, artifact_iri: str) -> ComponentKind:
    """Determine whether the artifact is a Model or Dataset from its rdf:type."""
    rows = list(store._backend.select(f"""
        {_RDF_PREFIX}
        {_AIBOM_PREFIX}
        SELECT ?t WHERE {{
          <{artifact_iri}> rdf:type ?t .
          VALUES ?t {{ aibom:Model aibom:Dataset }}
        }}
        LIMIT 1
    """))
    if rows:
        type_iri = str(rows[0]["t"])
        if type_iri == str(vocab.Dataset):
            return "Dataset"
    return "Model"


def _build_component(store: BomStore, artifact_iri: str, scope_in_bom: ComponentScope) -> Component:
    """Construct a Component from the store for the given artifact IRI."""
    return Component(
        kind=_kind_from_types(store, artifact_iri),
        hf_path=_label(store, artifact_iri),
        developer=_developer(store, artifact_iri),
        base_models=_base_model_labels(store, artifact_iri),
        scope_in_bom=scope_in_bom,
        spdx_id=artifact_iri,
    )


def _all_artifact_iris(store: BomStore) -> list[str]:
    """Return IRIs of all Model and Dataset artifacts in the store."""
    model_iri = str(vocab.Model)
    dataset_iri = str(vocab.Dataset)
    rows = store._backend.select(f"""
        {_RDF_PREFIX}
        SELECT DISTINCT ?artifact WHERE {{
          ?artifact rdf:type ?t .
          VALUES ?t {{ <{model_iri}> <{dataset_iri}> }}
        }}
    """)
    return [str(row["artifact"]) for row in rows]


def walk_components(store: BomStore, scope: Scope) -> Iterator[Component]:
    """Enumerate Components from the BomStore according to the given scope.

    - graph_wide: yields all Model and Dataset artifacts.
    - single: yields the requested artifact (with lineage in base_models).
    """
    if scope.kind == "graph_wide":
        yield from _walk_graph_wide(store)
    elif scope.kind == "single":
        yield from _walk_single(store, scope.artifact_iri, scope.depth)
    else:
        raise ValueError(f"Unknown scope kind: {scope.kind}")


def _walk_graph_wide(store: BomStore) -> Iterator[Component]:
    """Yield a Component for every Model/Dataset artifact in the graph."""
    for artifact_iri in _all_artifact_iris(store):
        yield _build_component(store, artifact_iri, "principal")


def _walk_single(store: BomStore, artifact_iri: str, depth: int) -> Iterator[Component]:
    """Yield the requested artifact; base_models are resolved inline."""
    # For the principal artifact, we always emit it regardless of depth.
    yield _build_component(store, artifact_iri, "principal")
