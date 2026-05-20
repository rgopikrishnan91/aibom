"""GraphOverlay payload builder: edge tinting + breaking-node rings."""
from __future__ import annotations

from collections import Counter

from aikaboom.plugins import GraphOverlay
from aikaboom.plugins.license_compat.engine import (
    Findings,
    find_compatible_subchains,
)

_COLORS = {
    "compatible": "#22c55e",
    "violation": "#ef4444",
    "unknown_upstream": "#94a3b8",
    "unknown_downstream": "#94a3b8",
    "missing_data": "#94a3b8",
}


def _edge_key(s: str, p: str, o: str) -> str:
    pred_short = p.rsplit("#", 1)[-1]
    return f"{s}|{pred_short}|{o}"


def build_overlay(findings: Findings, plugin_name: str) -> GraphOverlay:
    edge_attrs: dict[str, dict] = {}
    blame: Counter = Counter()

    for f in findings:
        edge_attrs[_edge_key(f.downstream_iri, f.predicate, f.upstream_iri)] = {
            "color": _COLORS.get(f.verdict.status, "#cccccc"),
            "label": f.verdict.status,
            "tooltip": (
                f"{f.downstream_license} {f.verdict.status} with "
                f"{sorted(f.upstream_licenses) or '—'} via "
                f"{f.predicate.rsplit('#', 1)[-1]}"
            ),
        }
        if f.is_violation() and f.upstream_licenses & f.verdict.incompatible_with:
            blame[f.upstream_iri] += 1

    node_attrs: dict[str, dict] = {}
    for iri, count in blame.items():
        node_attrs[iri] = {
            "ring_color": "#ef4444",
            "badge": str(count),
        }

    # Largest compatible subchain gets a faint halo on every member.
    subchains = find_compatible_subchains(findings)
    if subchains and subchains[0].size > 1:
        for iri in subchains[0].artifacts:
            node_attrs.setdefault(iri, {})["halo_color"] = "#bbf7d0"

    return GraphOverlay(
        plugin_name=plugin_name,
        edge_attrs=edge_attrs,
        node_attrs=node_attrs,
    )
