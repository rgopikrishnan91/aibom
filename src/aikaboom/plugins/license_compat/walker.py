"""Graph walker: enumerate lineage edges, resolve trust-aware licenses, compute frequencies."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterator, Optional

from aikaboom.plugins import Scope
from aikaboom.plugins.license_compat.matrix import (
    LicenseMatrix,
    normalize_license_field,
    resolve_license,
)
from aikaboom.store import BomStore, vocab

LINEAGE_PREDICATES = (
    str(vocab.trainedOn),
    str(vocab.testedOn),
    str(vocab.dependsOn),
    str(vocab.hostedAt),
)


@dataclass(frozen=True)
class LineageEdge:
    downstream_iri: str
    downstream_label: str
    upstream_iri: str
    upstream_label: str
    predicate: str


@dataclass(frozen=True)
class ResolvedArtifact:
    iri: str
    label: str
    licenses: frozenset[str]
    source_claim_iri: Optional[str]
    has_unknown: bool
    has_missing: bool


def _label(store: BomStore, iri: str) -> str:
    rows = list(store._backend.select(f"""
        PREFIX aibom: <https://aikaboom.dev/aibom#>
        SELECT ?l WHERE {{ <{iri}> aibom:canonicalLabel ?l }} LIMIT 1
    """))
    if rows:
        return str(rows[0]["l"])
    return iri.rsplit("/", 1)[-1]


def enumerate_edges(store: BomStore, scope: Scope) -> Iterator[LineageEdge]:
    if scope.kind == "graph_wide":
        yield from _enumerate_all(store)
    elif scope.kind == "single":
        yield from _enumerate_from(store, scope.artifact_iri, scope.depth)
    else:
        raise ValueError(f"Unknown scope kind: {scope.kind}")


def _enumerate_all(store: BomStore) -> Iterator[LineageEdge]:
    values_clause = " ".join(f"<{p}>" for p in LINEAGE_PREDICATES)
    rows = store._backend.select(f"""
        PREFIX aibom: <https://aikaboom.dev/aibom#>
        SELECT ?downstream ?upstream ?p WHERE {{
          VALUES ?p {{ {values_clause} }}
          ?artifact aibom:hasVersion ?version .
          ?version aibom:hasClaim ?claim .
          BIND(?artifact AS ?downstream)
          ?claim ?p ?upstream .
        }}
    """)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row["downstream"]), str(row["upstream"]), str(row["p"]))
        if key in seen:
            continue
        seen.add(key)
        yield LineageEdge(
            downstream_iri=key[0],
            downstream_label=_label(store, key[0]),
            upstream_iri=key[1],
            upstream_label=_label(store, key[1]),
            predicate=key[2],
        )


def _direct_upstreams(store: BomStore, artifact_iri: str) -> list[tuple[str, str]]:
    values_clause = " ".join(f"<{p}>" for p in LINEAGE_PREDICATES)
    rows = store._backend.select(f"""
        PREFIX aibom: <https://aikaboom.dev/aibom#>
        SELECT DISTINCT ?upstream ?p WHERE {{
          VALUES ?p {{ {values_clause} }}
          <{artifact_iri}> aibom:hasVersion ?version .
          ?version aibom:hasClaim ?claim .
          ?claim ?p ?upstream .
        }}
    """)
    return [(str(row["upstream"]), str(row["p"])) for row in rows]


def _enumerate_from(store: BomStore, start_iri: str, depth: int) -> Iterator[LineageEdge]:
    visited_nodes: set[str] = set()
    yielded_edges: set[tuple[str, str, str]] = set()
    frontier: list[tuple[str, int]] = [(start_iri, 0)]
    while frontier:
        artifact, level = frontier.pop(0)
        if artifact in visited_nodes:
            continue
        visited_nodes.add(artifact)
        if level >= depth:
            continue
        for up_iri, predicate in _direct_upstreams(store, artifact):
            key = (artifact, up_iri, predicate)
            if key in yielded_edges:
                continue
            yielded_edges.add(key)
            yield LineageEdge(
                downstream_iri=artifact,
                downstream_label=_label(store, artifact),
                upstream_iri=up_iri,
                upstream_label=_label(store, up_iri),
                predicate=predicate,
            )
            frontier.append((up_iri, level + 1))


def resolve_artifact_license(
    store: BomStore,
    artifact_iri: str,
    matrix: LicenseMatrix,
) -> ResolvedArtifact:
    rows = list(store._backend.select(f"""
        PREFIX aibom: <https://aikaboom.dev/aibom#>
        SELECT ?claim ?lic ?trust ?created WHERE {{
          {{
            <{artifact_iri}> aibom:hasVersion ?version .
            ?version aibom:hasClaim ?claim .
          }} UNION {{
            <{artifact_iri}> aibom:canonicalClaim ?claim .
          }}
          ?claim aibom:hasLicense ?lic .
          OPTIONAL {{ ?claim aibom:trustScore ?trust }}
          OPTIONAL {{ ?claim aibom:createdAt ?created }}
        }}
        ORDER BY DESC(?trust) DESC(?created)
        LIMIT 1
    """))
    has_unknown = False
    has_missing = False
    licenses: set[str] = set()
    source_claim: Optional[str] = None
    if rows:
        row = rows[0]
        source_claim = str(row["claim"])
        for raw in normalize_license_field(str(row["lic"])):
            r = resolve_license(raw, matrix)
            if r.is_missing:
                has_missing = True
                continue
            if r.is_unknown:
                has_unknown = True
            if r.primary_name is not None:
                licenses.add(r.primary_name)

    if not licenses:
        # Fallback: artifact-level hasLicense triple.
        fb = list(store._backend.select(f"""
            PREFIX aibom: <https://aikaboom.dev/aibom#>
            SELECT ?lic WHERE {{ <{artifact_iri}> aibom:hasLicense ?lic }} LIMIT 5
        """))
        for row in fb:
            for raw in normalize_license_field(str(row["lic"])):
                r = resolve_license(raw, matrix)
                if r.primary_name and not r.is_unknown:
                    licenses.add(r.primary_name)

    return ResolvedArtifact(
        iri=artifact_iri,
        label=_label(store, artifact_iri),
        licenses=frozenset(licenses),
        source_claim_iri=source_claim,
        has_unknown=has_unknown,
        has_missing=has_missing,
    )


def compute_license_frequencies(store: BomStore, matrix: LicenseMatrix) -> Counter:
    rows = store._backend.select("""
        PREFIX aibom: <https://aikaboom.dev/aibom#>
        SELECT ?lic WHERE {
          ?s aibom:hasLicense ?lic .
        }
    """)
    counts: Counter = Counter()
    for row in rows:
        for raw in normalize_license_field(str(row["lic"])):
            r = resolve_license(raw, matrix)
            if r.primary_name is not None and not r.is_unknown:
                counts[r.primary_name] += 1
    return counts
