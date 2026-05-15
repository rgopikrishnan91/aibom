"""BomStore — public facade over the graph backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from aikaboom.store import iris, vocab
from aikaboom.store.backend import GraphBackend, open_backend
from aikaboom.store.mapper import bom_to_rdf, rdf_to_bom
from aikaboom.store.naming import Identifier, canonicalize_set, pick_primary


@dataclass
class ResolveResult:
    """Outcome of a resolve step.

    Attributes:
        existing_artifact: IRI of a matching Artifact, or None when no
            stored artifact shares any of the supplied identifiers.
        artifact_label: The Artifact's canonicalLabel, when known.
        matching_claims: BOMClaim rows already attached to the matched
            artifact (filtered by `use_case`/`mode` when supplied).
        collision_artifacts: Additional Artifact IRIs that also matched
            the identifier set — populated when the supplied identifiers
            straddle two previously-disconnected artifact nodes (cross-
            identifier collision). The primary is in `existing_artifact`.
    """
    existing_artifact: str | None
    artifact_label: str | None
    matching_claims: list[dict] = field(default_factory=list)
    collision_artifacts: list[str] = field(default_factory=list)


def _escape_sparql_literal(s: str) -> str:
    """Escape a string for safe interpolation into a SPARQL string literal.

    Handles backslash, double-quote, newline, and carriage return.
    The result is intended for embedding inside `"..."` in SPARQL.
    """
    return (
        s.replace("\\", "\\\\")
         .replace('"', '\\"')
         .replace("\n", "\\n")
         .replace("\r", "\\r")
    )


def _validate_sparql_iri(s: str) -> str:
    """Validate an IRI string for safe interpolation into a SPARQL `<...>` slot.

    Raises ValueError if the string contains characters that would break
    out of the `<>` bracket form. Returns the input unchanged on success.
    """
    if any(c in s for c in '<>"\\\n\r '):
        raise ValueError(f"unsafe IRI for SPARQL interpolation: {s!r}")
    return s


class BomStore:
    def __init__(self, backend: GraphBackend):
        self._backend = backend

    @classmethod
    def open(cls) -> "BomStore":
        return cls(backend=open_backend())

    def save_claim(
        self,
        bom_json: Mapping[str, Any],
        run_meta: Mapping[str, Any],
        identifiers: list[Identifier],
    ) -> str:
        """Convert and persist a BOM. Returns the new claim IRI."""
        ds, claim_iri = bom_to_rdf(bom_json, run_meta, identifiers=identifiers)
        quads = [(s, p, o, None) for s, p, o, _ in ds.quads()]
        self._backend.add_quads(quads)
        return claim_iri

    def find_claims_for(
        self,
        identifiers: list[Identifier],
        use_case: str | None = None,
        mode: str | None = None,
    ) -> list[dict]:
        """Find existing claims for the artifact whose primary derives from `identifiers`.

        Note: this resolves the artifact via priority-pick over the supplied set,
        not by cross-identifier lookup. For the cross-identifier flow, use
        `BomStore.resolve()`.
        """
        canon = canonicalize_set(identifiers)
        if not canon:
            return []
        primary = pick_primary(canon)
        artifact = iris.artifact_iri(primary)
        return self._find_claims_for_artifact(artifact, use_case=use_case, mode=mode)

    def _find_claims_for_artifact(
        self,
        artifact_iri: str,
        use_case: str | None = None,
        mode: str | None = None,
    ) -> list[dict]:
        """SPARQL query for claims under a specific artifact IRI."""
        artifact = _validate_sparql_iri(artifact_iri)
        filters = []
        if use_case is not None:
            filters.append(f'?claim <{vocab.useCase}> "{_escape_sparql_literal(use_case)}" .')
        if mode is not None:
            filters.append(f'?claim <{vocab.mode}> "{_escape_sparql_literal(mode)}" .')
        filter_clause = "\n".join(filters)

        q = f"""
        SELECT ?claim ?createdAt ?llmModel WHERE {{
            <{artifact}> <{vocab.hasVersion}> ?version .
            ?version <{vocab.hasClaim}> ?claim .
            {filter_clause}
            OPTIONAL {{ ?claim <{vocab.createdAt}> ?createdAt . }}
            OPTIONAL {{
                ?claim <{vocab.generatedBy}> ?run .
                ?run <{vocab.llmModel}> ?llmModel .
            }}
        }}
        ORDER BY DESC(?createdAt)
        """
        out = []
        for row in self._backend.select(q):
            out.append({
                "iri": str(row["claim"]),
                "created_at": str(row.get("createdAt", "")),
                "llm_model": str(row.get("llmModel", "")),
            })
        return out

    def resolve(
        self,
        identifiers: list[Identifier],
        use_case: str | None = None,
        mode: str | None = None,
    ) -> ResolveResult:
        """Find existing Artifact + claims via cross-identifier lookup.

        Unlike `find_claims_for`, which uses only the primary identifier,
        `resolve` walks every identifier in the input set so that an
        Artifact previously saved under (huggingface=X, arxiv=Y) is still
        found when only `arxiv=Y` is supplied.

        Returns a `ResolveResult`:
          * no matches    → existing_artifact=None, matching_claims=[]
          * one match     → existing_artifact set, matching_claims populated
          * multi-match   → existing_artifact set to the first hit, with
                            the rest in `collision_artifacts` (caller can
                            issue a `potentialDuplicateOf` link later).
        """
        canon = canonicalize_set(identifiers)
        if not canon:
            return ResolveResult(existing_artifact=None, artifact_label=None)

        # Build a VALUES clause across all provided identifiers. Both
        # platform and value are user-supplied, so escape each literal
        # to defeat SPARQL injection (see `_escape_sparql_literal`).
        values_rows = "\n".join(
            f'            ("{_escape_sparql_literal(ident.platform)}" '
            f'"{_escape_sparql_literal(ident.value)}")'
            for ident in canon
        )
        q = f"""
        SELECT DISTINCT ?artifact ?label WHERE {{
            ?artifact <{vocab.identifier}> ?id .
            ?id <{vocab.platform}> ?p ; <{vocab.value}> ?v .
            VALUES (?p ?v) {{
{values_rows}
            }}
            OPTIONAL {{ ?artifact <{vocab.canonicalLabel}> ?label . }}
        }}
        """
        matches: list[tuple[str, str]] = []
        for row in self._backend.select(q):
            matches.append((str(row["artifact"]), str(row.get("label") or "")))

        if not matches:
            return ResolveResult(existing_artifact=None, artifact_label=None)

        if len(matches) > 1:
            return ResolveResult(
                existing_artifact=matches[0][0],
                artifact_label=matches[0][1],
                collision_artifacts=[m[0] for m in matches[1:]],
                matching_claims=self._find_claims_for_artifact(
                    matches[0][0], use_case=use_case, mode=mode,
                ),
            )

        return ResolveResult(
            existing_artifact=matches[0][0],
            artifact_label=matches[0][1],
            matching_claims=self._find_claims_for_artifact(
                matches[0][0], use_case=use_case, mode=mode,
            ),
        )

    def stats(self) -> dict[str, int]:
        """Return node counts by class."""
        out = {}
        for label, cls in [
            ("artifacts", vocab.Artifact),
            ("versions", vocab.ArtifactVersion),
            ("claims", vocab.BOMClaim),
            ("votes", vocab.TrustVote),
        ]:
            rows = list(
                self._backend.select(
                    f"SELECT (COUNT(?s) AS ?n) WHERE {{ ?s a <{cls}> }}"
                )
            )
            out[label] = int(rows[0]["n"]) if rows else 0
        return out

    def reconstruct_bom(self, claim_iri: str) -> dict:
        """Rebuild a BOM JSON dict from a stored claim.

        Internally builds a small rdflib.Dataset by selecting every triple
        whose subject is `claim_iri` (the claim's own triples) plus every
        annotation blank node that references those triples, then hands
        the dataset to `rdf_to_bom`.
        """
        from rdflib import Dataset as _RDFDataset, URIRef as _URIRef, Literal as _Literal, BNode as _BNode

        ds = _RDFDataset()

        # Pull every (claim_iri, p, o) triple.
        q_claim = f"SELECT ?p ?o WHERE {{ <{_validate_sparql_iri(claim_iri)}> ?p ?o }}"
        for row in self._backend.select(q_claim):
            p = _URIRef(str(row["p"]))
            o_raw = row["o"]
            o = _URIRef(str(o_raw)) if str(o_raw).startswith(("http", "bom:", "aibom:", "_:")) else _Literal(str(o_raw))
            ds.add((_URIRef(claim_iri), p, o))

        # Pull annotation blank nodes that point at this claim.
        q_ann = f"""
        SELECT ?ann ?p ?subj ?pred ?obj ?asserted ?conflict WHERE {{
            ?ann <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject> <{_validate_sparql_iri(claim_iri)}> .
            ?ann <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate> ?pred .
            ?ann <http://www.w3.org/1999/02/22-rdf-syntax-ns#object> ?obj .
            OPTIONAL {{ ?ann <{vocab.assertedBy}> ?asserted . }}
            OPTIONAL {{ ?ann <{vocab.conflictKind}> ?conflict . }}
        }}
        """
        for row in self._backend.select(q_ann):
            ann = _BNode()
            ds.add((ann, _URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"), _URIRef(claim_iri)))
            ds.add((ann, _URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"), _URIRef(str(row["pred"]))))
            obj_raw = row["obj"]
            obj_val = _URIRef(str(obj_raw)) if str(obj_raw).startswith(("http", "bom:", "aibom:", "_:")) else _Literal(str(obj_raw))
            ds.add((ann, _URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#object"), obj_val))
            if row.get("asserted"):
                ds.add((ann, _URIRef(vocab.assertedBy), _URIRef(str(row["asserted"]))))
            if row.get("conflict"):
                ds.add((ann, _URIRef(vocab.conflictKind), _URIRef(str(row["conflict"]))))

        # Pull the artifact label via the hasClaim back-edge so rdf_to_bom can populate repo_id.
        q_label = f"""
        SELECT ?label WHERE {{
            ?version <{vocab.hasClaim}> <{_validate_sparql_iri(claim_iri)}> .
            ?artifact <{vocab.hasVersion}> ?version ;
                      <{vocab.canonicalLabel}> ?label .
        }}
        """
        for row in self._backend.select(q_label):
            # Add the back-edges into ds so rdf_to_bom finds them.
            v = _BNode()
            a = _BNode()
            ds.add((v, _URIRef(vocab.hasClaim), _URIRef(claim_iri)))
            ds.add((a, _URIRef(vocab.hasVersion), v))
            ds.add((a, _URIRef(vocab.canonicalLabel), _Literal(str(row["label"]))))
            break

        return rdf_to_bom(ds, claim_iri)

    def close(self) -> None:
        self._backend.close()
