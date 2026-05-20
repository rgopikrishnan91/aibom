"""Artifact-to-artifact relationship edges for the worldofBOMs graph.

A BOM's relationship fields (trainedOnDatasets, testedOnDatasets,
modelLineage, sourceInfo) name other artifacts. This module turns those
names into real `trainedOn` / `testedOn` / `dependsOn` edges between
Artifact nodes, so the stored graph is connected rather than a set of
disconnected stars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from aikaboom.store.store import BomStore

from rdflib import Literal, URIRef, XSD
from rdflib.namespace import RDF

from aikaboom.store import iris, vocab
from aikaboom.store.naming import Identifier, canonicalize
from aikaboom.store.store import _validate_sparql_iri
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

_BOMTYPE_TO_KIND = {"ai": "Model", "data": "Dataset"}
_FIELD_TO_KIND: dict[str, str] = {
    field: _BOMTYPE_TO_KIND.get(spec[0], "Artifact")
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


def canon_name(name: str) -> str:
    """Canonicalize a free-text artifact name for identity comparison.

    Reuses the identifier canonicalization pipeline with the `name-only`
    platform (lowercase, separator collapse) — the conservative identity
    layer, never the fuzzy supplier triage.
    """
    return canonicalize(Identifier("name-only", name)).value


def _find_artifact_by_label(store: "BomStore", name: str) -> str | None:
    """Return a non-placeholder Artifact IRI whose canonical label matches `name`."""
    target = canon_name(name)
    rows = store._backend.select(
        f"""
        SELECT ?artifact ?label WHERE {{
            ?artifact a <{vocab.Artifact}> ;
                      <{vocab.canonicalLabel}> ?label .
            FILTER NOT EXISTS {{ ?artifact <{vocab.isPlaceholder}> true }}
        }}
        """
    )
    # Label comparison is done in Python (reusing canon_name) rather than in
    # SPARQL so we get the same separator-collapse + lowercase normalisation.
    for row in rows:
        if canon_name(str(row["label"])) == target:
            return str(row["artifact"])
    return None


def _mint_placeholder(store: "BomStore", name: str, kind: str = "Artifact") -> str:
    """Create a flagged placeholder Artifact for an unresolved name; return its IRI.

    If `kind` is "Model" or "Dataset", also asserts the corresponding type triple
    so the placeholder is properly typed from the start.
    """
    ident = canonicalize(Identifier("name-only", name))
    art = iris.artifact_iri(ident)
    if store._backend.ask(f"ASK {{ <{art}> a <{vocab.Artifact}> }}"):
        return art  # already minted by an earlier edge
    quads = [
        (URIRef(art), RDF.type, URIRef(vocab.Artifact), None),
        (URIRef(art), URIRef(vocab.isPlaceholder),
         Literal(True, datatype=XSD.boolean), None),
        (URIRef(art), URIRef(vocab.canonicalLabel), Literal(name), None),
        (URIRef(art), URIRef(vocab.primaryIdentifier),
         Literal(f"name-only:{ident.value}"), None),
    ]
    ident_node = URIRef(f"{art}/id")
    quads += [
        (URIRef(art), URIRef(vocab.identifier), ident_node, None),
        (ident_node, URIRef(vocab.platform), Literal("name-only"), None),
        (ident_node, URIRef(vocab.value), Literal(ident.value), None),
    ]
    if kind != "Artifact":
        quads.append(
            (URIRef(art), RDF.type, URIRef(getattr(vocab, kind)), None)
        )
    store._backend.add_quads(quads)
    return art


def resolve_edge_target(store: "BomStore", name: str,
                        kind: str = "Artifact") -> tuple[str, bool]:
    """Resolve a relationship target name to an Artifact IRI.

    Resolution order (identity layer only — no fuzzy matching here):
      1. identifier match via `store.resolve` (the cache-hit path);
      2. exact name-label match against an existing non-placeholder artifact;
      3. mint a flagged placeholder typed by `kind`.

    Returns `(artifact_iri, minted_placeholder)`.
    """
    resolved = store.resolve([Identifier("name-only", name)])
    if resolved.existing_artifact:
        return resolved.existing_artifact, False
    by_label = _find_artifact_by_label(store, name)
    if by_label:
        return by_label, False
    placeholder = _mint_placeholder(store, name, kind=kind)
    _record_potential_duplicates(store, placeholder, name)
    return placeholder, True


def _record_potential_duplicates(store: "BomStore", placeholder_iri: str,
                                 name: str) -> None:
    """Soft-link a placeholder to confident-but-inexact existing artifacts.

    Uses the supplier-alias confidence triage (Jaro-Winkler tier included).
    Records `potentialDuplicateOf` — a hint for the UI — and never merges.
    """
    from aikaboom.utils.supplier_alias import default_alias_index

    index = default_alias_index()
    owner = name.partition("/")[0] if "/" in name else name
    rows = list(store._backend.select(
        f"""
        SELECT ?artifact ?label WHERE {{
            ?artifact a <{vocab.Artifact}> ;
                      <{vocab.canonicalLabel}> ?label .
            FILTER NOT EXISTS {{ ?artifact <{vocab.isPlaceholder}> true }}
        }}
        """
    ))
    src = _validate_sparql_iri(placeholder_iri)
    for row in rows:
        cand_label = str(row["label"])
        cand_owner = cand_label.partition("/")[0] if "/" in cand_label else cand_label
        if canon_name(cand_label) == canon_name(name):
            continue  # exact — handled by name-label match, not a "duplicate"
        if index.is_same_supplier(owner, cand_owner):
            tgt = _validate_sparql_iri(str(row["artifact"]))
            if not store._backend.ask(
                f"ASK {{ <{src}> <{vocab.potentialDuplicateOf}> <{tgt}> }}"
            ):
                store._backend.add_quads(
                    [(URIRef(src), URIRef(vocab.potentialDuplicateOf),
                      URIRef(tgt), None)]
                )


def promote_placeholders_for(store: "BomStore", real_artifact_iri: str,
                             label: str) -> list[str]:
    """Merge any name-only placeholders that match `label` into the real artifact.

    Exact canonical-name equality only — never fuzzy. Returns the list of
    placeholder IRIs that were merged away.
    """
    target = canon_name(label)
    real = _validate_sparql_iri(real_artifact_iri)
    merged: list[str] = []
    rows = list(store._backend.select(
        f"""
        SELECT ?artifact ?label WHERE {{
            ?artifact a <{vocab.Artifact}> ;
                      <{vocab.isPlaceholder}> true ;
                      <{vocab.canonicalLabel}> ?label .
        }}
        """
    ))
    for row in rows:
        placeholder = str(row["artifact"])
        if placeholder == real:
            continue
        if canon_name(str(row["label"])) == target:
            store.merge_artifacts(into=real, from_=placeholder)
            merged.append(placeholder)
    return merged


def add_relationship_edges(store: "BomStore", source_artifact_iri: str,
                           bom_json: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Persist `trainedOn`/`testedOn`/`dependsOn` edges from a saved BOM.

    For each relationship target in `bom_json`, resolves it to an Artifact
    IRI (Task A4) and writes one edge triple. Edge writes are idempotent —
    an `ASK` guard skips a triple that already exists.

    Returns the list of `(source, predicate, target)` edges added.
    """
    src = _validate_sparql_iri(source_artifact_iri)
    added: list[tuple[str, str, str]] = []
    for predicate, target_name, kind in extract_relationship_targets(bom_json):
        pred_uri = str(getattr(vocab, predicate))
        target_iri, _minted = resolve_edge_target(store, target_name, kind)
        tgt = _validate_sparql_iri(target_iri)
        if tgt == src:
            continue  # never self-loop
        if store._backend.ask(f"ASK {{ <{src}> <{pred_uri}> <{tgt}> }}"):
            continue
        store._backend.add_quads(
            [(URIRef(src), URIRef(pred_uri), URIRef(tgt), None)]
        )
        added.append((src, predicate, tgt))
    return added


def extract_relationship_targets(bom_json: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Return `(predicate, target_name, kind)` triples for every walkable edge target.

    `predicate` is one of "trainedOn" / "testedOn" / "dependsOn".
    `kind` is "Model", "Dataset", or "Artifact" derived from the BOM field's
    child bom-type so that minted placeholders are properly typed.

    Scans both ``direct_fields`` and ``rag_fields`` because the graph store
    may receive BOMs that store relationship fields in either section depending
    on how the extractor classified them (high-confidence direct extraction vs.
    RAG-assisted retrieval).
    """
    out: list[tuple[str, str, str]] = []
    for section in ("direct_fields", "rag_fields"):
        fields = bom_json.get(section) or {}
        if not isinstance(fields, Mapping):
            continue
        for field_name, predicate in _FIELD_TO_PREDICATE.items():
            triplet = fields.get(field_name)
            if not isinstance(triplet, Mapping):
                continue
            kind = _FIELD_TO_KIND[field_name]
            for target in _split_targets(triplet.get("value")):
                if _is_walkable_target(target):
                    out.append((predicate, target, kind))
    return out
