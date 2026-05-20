"""Pure license-compatibility engine.

No I/O, no graph. Inputs are LicenseMatrix values; outputs are dataclass
verdicts and recommendations. Mirrors LicenseRec.py's analytical primitives.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

from aikaboom.plugins.license_compat.matrix import LicenseMatrix

Status = Literal["compatible", "violation", "unknown_upstream", "unknown_downstream", "missing_data"]


@dataclass(frozen=True)
class CompatVerdict:
    downstream: Optional[str]
    upstreams: frozenset[str]
    status: Status
    incompatible_with: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Recommendation:
    by_category: dict[str, list[str]]
    is_solvable: bool


_CC_VERSION_RE = re.compile(r"^(cc-[a-z\-]+)-(\d\.\d)$", re.IGNORECASE)


def check_compat(downstream: Optional[str], upstreams: frozenset[str], matrix: LicenseMatrix) -> CompatVerdict:
    if downstream is None:
        return CompatVerdict(None, upstreams, "unknown_downstream")
    if "UNKNOWN" in upstreams:
        return CompatVerdict(downstream, upstreams, "unknown_upstream")
    if not upstreams:
        return CompatVerdict(downstream, upstreams, "compatible")
    if downstream not in matrix.details:
        return CompatVerdict(downstream, upstreams, "missing_data")
    blocked = frozenset(
        up for up in upstreams
        if downstream not in matrix.upstream_compat_index.get(up, frozenset())
    )
    if blocked:
        return CompatVerdict(downstream, upstreams, "violation", incompatible_with=blocked)
    return CompatVerdict(downstream, upstreams, "compatible")


def recommend(
    upstreams: frozenset[str],
    matrix: LicenseMatrix,
    frequencies: Counter,
    top_k_per_category: int = 5,
) -> Recommendation:
    if not upstreams:
        return Recommendation(by_category={}, is_solvable=False)

    compat_sets = [matrix.upstream_compat_index.get(up, frozenset()) for up in upstreams]
    if not compat_sets:
        return Recommendation(by_category={}, is_solvable=False)

    candidate_names = frozenset.intersection(*compat_sets) if len(compat_sets) > 1 else compat_sets[0]
    is_solvable = len(candidate_names) > 0
    if not is_solvable:
        return Recommendation(by_category={}, is_solvable=False)

    filtered = [matrix.details[n] for n in candidate_names if n.lower() in matrix.allowed_licenses and n in matrix.details]
    if not filtered:
        return Recommendation(by_category={}, is_solvable=is_solvable)

    grouped: dict[str, list[str]] = defaultdict(list)
    for entry in filtered:
        grouped[entry.get("category", "UNKNOWN")].append(entry["name"])

    for cat, lic_list in list(grouped.items()):
        processed = lic_list
        cc_versions = [m for l in lic_list if (m := _CC_VERSION_RE.match(l))]
        if cc_versions:
            bases_with_4_0 = {m.group(1).lower() for l in lic_list if (m := _CC_VERSION_RE.match(l)) and m.group(2) == "4.0"}
            processed = [
                l for l in lic_list
                if not (m := _CC_VERSION_RE.match(l)) or m.group(2) == "4.0" or m.group(1).lower() not in bases_with_4_0
            ]
        grouped[cat] = sorted(processed, key=lambda l: (-frequencies.get(l, 0), l.lower()))[:top_k_per_category]

    return Recommendation(by_category=dict(grouped), is_solvable=is_solvable)


@dataclass(frozen=True)
class Finding:
    downstream_iri: str
    downstream_label: str
    upstream_iri: str
    upstream_label: str
    predicate: str
    verdict: CompatVerdict
    downstream_license: Optional[str] = None
    upstream_licenses: frozenset[str] = field(default_factory=frozenset)
    recommendation: Optional[Recommendation] = None

    def is_violation(self) -> bool:
        return self.verdict.status == "violation"

    def is_compatible(self) -> bool:
        return self.verdict.status == "compatible"


@dataclass(frozen=True)
class CompatSubchain:
    artifacts: frozenset[str]
    edges: frozenset[tuple[str, str, str]]
    size: int
    root: str


@dataclass(frozen=True)
class BreakingNode:
    artifact_iri: str
    label: str
    license: Optional[str]
    blamed_in: int
    affected_downstream: frozenset[str]
    fix_recommendations: Recommendation


class Findings:
    """Iterable wrapper around list[Finding] with helpers."""

    def __init__(self, items: Iterable[Finding]):
        self._items: list[Finding] = list(items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def violations(self) -> list[Finding]:
        return [f for f in self._items if f.is_violation()]

    def to_dict(self) -> dict:
        return {
            "findings": [
                {
                    "downstream_iri": f.downstream_iri,
                    "upstream_iri": f.upstream_iri,
                    "predicate": f.predicate,
                    "status": f.verdict.status,
                    "downstream_license": f.downstream_license,
                    "upstream_licenses": sorted(f.upstream_licenses),
                    "incompatible_with": sorted(f.verdict.incompatible_with),
                    "recommendation": (
                        None if f.recommendation is None else {
                            "by_category": f.recommendation.by_category,
                            "is_solvable": f.recommendation.is_solvable,
                        }
                    ),
                }
                for f in self._items
            ],
        }


def find_compatible_subchains(findings: Findings) -> list[CompatSubchain]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    all_artifacts: set[str] = set()
    compat_edges: list[tuple[str, str, str]] = []
    for f in findings:
        all_artifacts.add(f.downstream_iri)
        all_artifacts.add(f.upstream_iri)
        if f.is_compatible():
            compat_edges.append((f.downstream_iri, f.upstream_iri, f.predicate))
            union(f.downstream_iri, f.upstream_iri)

    # Build components — every artifact that was *seen* gets a root, even
    # if it sits on no compatible edge (size-1 component).
    for a in all_artifacts:
        parent.setdefault(a, a)

    groups: dict[str, set[str]] = defaultdict(set)
    for a in all_artifacts:
        groups[find(a)].add(a)

    edges_by_root: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for s, u, p in compat_edges:
        edges_by_root[find(s)].add((s, u, p))

    chains = [
        CompatSubchain(
            artifacts=frozenset(members),
            edges=frozenset(edges_by_root.get(root, set())),
            size=len(members),
            root=root,
        )
        for root, members in groups.items()
    ]
    chains.sort(key=lambda c: (-c.size, c.root))
    return chains


def find_breaking_nodes(
    findings: Findings,
    matrix: LicenseMatrix,
    frequencies: Counter,
) -> list[BreakingNode]:
    blame: dict[str, list[Finding]] = defaultdict(list)
    upstream_license: dict[str, Optional[str]] = {}
    upstream_label: dict[str, str] = {}
    for f in findings.violations():
        if f.upstream_iri in f.verdict.incompatible_with or any(
            u == f.upstream_iri for u in f.verdict.incompatible_with
        ):
            # incompatible_with carries license names, not IRIs — match on the
            # finding's resolved upstream_licenses if they intersect.
            pass
        # If any of the upstream's licenses is in incompatible_with, the
        # upstream IRI is "blamed" for the violation.
        if f.upstream_licenses & f.verdict.incompatible_with:
            blame[f.upstream_iri].append(f)
            upstream_label[f.upstream_iri] = f.upstream_label
            if f.upstream_licenses:
                upstream_license[f.upstream_iri] = next(iter(f.upstream_licenses))

    nodes: list[BreakingNode] = []
    for iri, edges in blame.items():
        affected = frozenset(e.downstream_iri for e in edges)
        # contextual fix: union of downstream licenses that blame this node
        downstream_lic_union = frozenset(
            e.downstream_license for e in edges if e.downstream_license is not None
        )
        fix = (
            recommend(downstream_lic_union, matrix, frequencies)
            if downstream_lic_union
            else Recommendation(by_category={}, is_solvable=False)
        )
        nodes.append(
            BreakingNode(
                artifact_iri=iri,
                label=upstream_label.get(iri, iri),
                license=upstream_license.get(iri),
                blamed_in=len(edges),
                affected_downstream=affected,
                fix_recommendations=fix,
            )
        )

    nodes.sort(key=lambda n: (-n.blamed_in, n.artifact_iri))
    return nodes
