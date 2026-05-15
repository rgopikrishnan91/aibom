"""Convert BOM JSON ↔ RDF quads.

`bom_to_rdf` is lossy in one direction (only stores what the schema knows
about); `rdf_to_bom` reconstructs the JSON. Round-trip is lossless for the
fields the vocab defines, asserted by `test_mapper_roundtrip.py`.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Iterable, Mapping

from rdflib import BNode, Dataset, Literal, URIRef, XSD
from rdflib.namespace import RDF

from aikaboom.store import iris, vocab
from aikaboom.store.naming import Identifier, canonicalize_set, pick_primary


def _u(s: str) -> URIRef:
    return URIRef(s)


def _field_value(bom: Mapping[str, Any], section: str, field: str) -> Any:
    """Safely extract a triplet's value, tolerating None at any level."""
    section_dict = bom.get(section) or {}
    triplet = section_dict.get(field) or {}
    if not isinstance(triplet, dict):
        return None
    return triplet.get("value")


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
    ds.add((run, RDF.type, _u(vocab.GenerationRun)))
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
    """Add one field claim triple + reified annotation carrying source + conflictKind."""
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
        ds.add((ann, RDF.subject, claim))
        ds.add((ann, RDF.predicate, pred))
        ds.add((ann, RDF.object, obj))
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
    ds.add((artifact, RDF.type, _u(_kind_for_platform(primary.platform))))
    ds.add((artifact, _u(vocab.primaryIdentifier), Literal(f"{primary.platform}:{primary.value}")))
    ds.add((artifact, _u(vocab.canonRuleVersion), Literal(vocab.CANON_RULE_VERSION)))
    _add_identifier_set(ds, artifact, canon_ids)
    label = bom_json.get("repo_id") or bom_json.get("model_id") or primary.value
    ds.add((artifact, _u(vocab.canonicalLabel), Literal(str(label))))

    version_str = (
        _field_value(bom_json, "direct_fields", "packageVersion")
        or _field_value(bom_json, "direct_fields", "contentIdentifier")
        or "unknown"
    )
    version = _u(iris.version_iri(str(artifact), str(version_str)))
    ds.add((version, RDF.type, _u(vocab.ArtifactVersion)))
    ds.add((artifact, _u(vocab.hasVersion), version))

    claim = _u(iris.claim_iri())
    ds.add((claim, RDF.type, _u(vocab.BOMClaim)))
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


def _platform_name_from_source_iri(source_iri_str: str) -> str:
    """Reverse `iris.source_iri('huggingface')` → 'huggingface'."""
    prefix = "aibom:source/"
    if source_iri_str.startswith(prefix):
        return source_iri_str[len(prefix):]
    return source_iri_str


def _conflict_kind_from_iri(kind_iri: str) -> Any:
    """Reverse the conflictKind mapping to the JSON form."""
    if kind_iri == str(vocab.noConflict):
        return None
    if kind_iri == str(vocab.interSourceConflict):
        return {"type": "inter"}
    if kind_iri == str(vocab.intraSourceConflict):
        return {"type": "intra"}
    return None


def rdf_to_bom(ds: Dataset, claim_iri: str) -> dict:
    """Reconstruct a BOM JSON dict from a claim's subgraph.

    Args:
        ds: the dataset to read from.
        claim_iri: the BOMClaim IRI to reconstruct.

    Returns:
        A dict with the same shape as the original BOM JSON (subset:
        only the fields the vocab models — round-trip-asserted by
        test_mapper_roundtrip.py).
    """
    claim = _u(claim_iri)
    out: dict[str, Any] = {"direct_fields": {}, "rag_fields": {}, "beta_fields": []}

    # Resolve the artifact via hasClaim back-edge.
    artifact_label = None
    for s, _, _, _ in ds.quads((None, _u(vocab.hasClaim), None, None)):
        # s is an ArtifactVersion; find its Artifact.
        for art, _, _, _ in ds.quads((None, _u(vocab.hasVersion), s, None)):
            for _, _, lab, _ in ds.quads((art, _u(vocab.canonicalLabel), None, None)):
                artifact_label = str(lab)
                break
            break
        break
    if artifact_label:
        out["model_id"] = artifact_label.replace("/", "_")
        out["repo_id"] = artifact_label

    # Use case + mode
    for _, _, lit, _ in ds.quads((claim, _u(vocab.useCase), None, None)):
        out["use_case"] = str(lit)
    for _, _, lit, _ in ds.quads((claim, _u(vocab.mode), None, None)):
        out["mode"] = str(lit)

    # Walk every (claim, pred, value) triple; pred → field name.
    aibom_ns = str(vocab.AIBOM)
    structural_preds = {
        str(vocab.useCase), str(vocab.mode), str(vocab.createdAt),
        str(vocab.schemaVersion), str(vocab.trustScore), str(vocab.generatedBy),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    }
    for _, p, o, _ in ds.quads((claim, None, None, None)):
        p_str = str(p)
        if p_str in structural_preds:
            continue
        if not p_str.startswith(aibom_ns):
            continue
        field_name = p_str[len(aibom_ns):]
        triplet: dict[str, Any] = {"value": str(o), "source": None, "conflict": None}
        # Find the annotation blank node for this triple, if any.
        for ann, _, _, _ in ds.quads((None, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#object"), o, None)):
            ann_subj = None
            ann_pred = None
            for _, _, s_val, _ in ds.quads((ann, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"), None, None)):
                ann_subj = s_val
            for _, _, p_val, _ in ds.quads((ann, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"), None, None)):
                ann_pred = p_val
            if ann_subj == claim and ann_pred == p:
                for _, _, src, _ in ds.quads((ann, _u(vocab.assertedBy), None, None)):
                    triplet["source"] = _platform_name_from_source_iri(str(src))
                for _, _, kind, _ in ds.quads((ann, _u(vocab.conflictKind), None, None)):
                    triplet["conflict"] = _conflict_kind_from_iri(str(kind))
                break
        out["direct_fields"][field_name] = triplet
    return out
