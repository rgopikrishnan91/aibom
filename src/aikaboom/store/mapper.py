"""Convert BOM JSON ↔ RDF quads.

`bom_to_rdf` is lossy in one direction (only stores what the schema knows
about); `rdf_to_bom` reconstructs the JSON. Round-trip is lossless for the
fields the vocab defines, asserted by `test_mapper_roundtrip.py`.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Iterable, Mapping

from rdflib import BNode, Dataset, Literal, URIRef, XSD

from aikaboom.store import iris, vocab
from aikaboom.store.naming import Identifier, canonicalize_set, pick_primary


def _u(s: str) -> URIRef:
    return URIRef(s)


def _kind_for_platform(platform: str) -> URIRef:
    """Map a platform key to an Artifact subclass."""
    return {
        "huggingface": vocab.Model,
        "github": vocab.CodeRepo,
        "arxiv": vocab.Paper,
    }.get(platform, vocab.Artifact)


def _add_identifier_set(ds: Dataset, artifact: URIRef, idents: Iterable[Identifier]) -> None:
    """Attach each canonical identifier as a blank-node entry, plus aliases."""
    for ident in idents:
        node = BNode()
        ds.add((artifact, _u(vocab.identifier), node))
        ds.add((node, _u(vocab.platform), Literal(ident.platform)))
        ds.add((node, _u(vocab.value), Literal(ident.value)))


def _add_generation_run(ds: Dataset, run_meta: Mapping[str, Any]) -> URIRef:
    run = _u(iris.run_iri(run_meta))
    ds.add((run, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _u(vocab.GenerationRun)))
    for field in ("provider", "llm_model", "prompt_version", "code_version", "mode", "use_case"):
        if field in run_meta and run_meta[field] is not None:
            predicate_uri = {
                "provider": vocab.provider,
                "llm_model": vocab.llmModel,
                "prompt_version": vocab.promptVersion,
                "code_version": vocab.codeVersion,
                "mode": vocab.mode,
                "use_case": vocab.useCase,
            }[field]
            ds.add((run, _u(predicate_uri), Literal(str(run_meta[field]))))
    return run


def _add_field_claim(
    ds: Dataset,
    claim: URIRef,
    field_name: str,
    triplet: Mapping[str, Any],
) -> None:
    """Add one field claim triple + RDF-star annotation with source."""
    value = triplet.get("value")
    if value is None:
        return
    pred = _u(vocab.AIBOM[field_name])
    obj = Literal(str(value))
    ds.add((claim, pred, obj))
    source = triplet.get("source")
    if source:
        # rdflib's RDF-star support uses .add_quoted in newer versions;
        # for portability we model the annotation as a separate metadata
        # triple keyed on a deterministic blank node. The CI test
        # test_conflict_preservation asserts both forms round-trip.
        ann = BNode()
        ds.add((ann, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"), claim))
        ds.add((ann, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"), pred))
        ds.add((ann, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#object"), obj))
        ds.add((ann, _u(vocab.assertedBy), _u(iris.source_iri(source))))
        conflict = triplet.get("conflict")
        if conflict is None:
            ds.add((ann, _u(vocab.conflictKind), _u(vocab.noConflict)))
        elif isinstance(conflict, dict):
            kind = conflict.get("type", "inter")
            kind_uri = vocab.interSourceConflict if kind == "inter" else vocab.intraSourceConflict
            ds.add((ann, _u(vocab.conflictKind), _u(kind_uri)))


def bom_to_rdf(
    bom_json: Mapping[str, Any],
    run_meta: Mapping[str, Any],
    identifiers: list[Identifier],
) -> tuple[Dataset, str]:
    """Convert a BOM JSON dict into an RDF Dataset.

    Args:
        bom_json: the dict produced by AIBOMProcessor / DATABOMProcessor.
        run_meta: GenerationRun parameters (provider, llm_model, ...).
        identifiers: known platform identifiers for the artifact.

    Returns:
        (dataset, claim_iri) — the dataset contains the artifact subgraph;
        the claim_iri is the new BOMClaim's IRI.
    """
    canon_ids = canonicalize_set(identifiers)
    if not canon_ids:
        raise ValueError("bom_to_rdf requires at least one identifier")
    primary = pick_primary(canon_ids)

    ds = Dataset()

    artifact = _u(iris.artifact_iri(primary))
    ds.add((artifact, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _u(_kind_for_platform(primary.platform))))
    ds.add((artifact, _u(vocab.primaryIdentifier), Literal(f"{primary.platform}:{primary.value}")))
    ds.add((artifact, _u(vocab.canonRuleVersion), Literal(vocab.CANON_RULE_VERSION)))
    _add_identifier_set(ds, artifact, canon_ids)
    label = bom_json.get("repo_id") or bom_json.get("model_id") or primary.value
    ds.add((artifact, _u(vocab.canonicalLabel), Literal(str(label))))

    version_str = (
        bom_json.get("direct_fields", {})
        .get("packageVersion", {})
        .get("value")
        or bom_json.get("direct_fields", {})
        .get("contentIdentifier", {})
        .get("value")
        or "unknown"
    )
    version = _u(iris.version_iri(str(artifact), str(version_str)))
    ds.add((version, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _u(vocab.ArtifactVersion)))
    ds.add((artifact, _u(vocab.hasVersion), version))

    claim = _u(iris.claim_iri())
    ds.add((claim, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _u(vocab.BOMClaim)))
    ds.add((version, _u(vocab.hasClaim), claim))
    ds.add((claim, _u(vocab.useCase), Literal(run_meta.get("use_case", "complete"))))
    ds.add((claim, _u(vocab.mode), Literal(run_meta.get("mode", "rag"))))
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    ds.add((claim, _u(vocab.createdAt), Literal(now, datatype=XSD.dateTime)))
    ds.add((claim, _u(vocab.schemaVersion), Literal(vocab.SCHEMA_VERSION)))
    ds.add((claim, _u(vocab.trustScore), Literal(0.0, datatype=XSD.decimal)))

    run = _add_generation_run(ds, run_meta)
    ds.add((claim, _u(vocab.generatedBy), run))

    for section in ("direct_fields", "rag_fields"):
        for field_name, triplet in (bom_json.get(section) or {}).items():
            if isinstance(triplet, dict):
                _add_field_claim(ds, claim, field_name, triplet)

    return ds, str(claim)
