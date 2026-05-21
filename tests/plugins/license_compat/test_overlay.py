"""GraphOverlay payload shape + color rules."""
from __future__ import annotations

from aikaboom.plugins.license_compat.engine import (
    CompatVerdict,
    Finding,
    Findings,
)
from aikaboom.plugins.license_compat.overlay import build_overlay


def _f(d: str, u: str, status: str, predicate: str = "trainedOn") -> Finding:
    return Finding(
        downstream_iri=d, downstream_label=d,
        upstream_iri=u, upstream_label=u,
        predicate=predicate,
        downstream_license="mit",
        upstream_licenses=frozenset({"apache-2.0"}),
        verdict=CompatVerdict(
            downstream="mit", upstreams=frozenset({"apache-2.0"}),
            status=status,
            incompatible_with=frozenset({"apache-2.0"}) if status == "violation" else frozenset(),
        ),
        recommendation=None,
    )


def test_overlay_colors_compatible_edges_green():
    o = build_overlay(Findings([_f("A", "B", "compatible")]), plugin_name="license-compat")
    key = "A|trainedOn|B"
    assert key in o.edge_attrs
    assert o.edge_attrs[key]["color"] == "#22c55e"


def test_overlay_colors_violation_edges_red():
    o = build_overlay(Findings([_f("A", "B", "violation")]), plugin_name="license-compat")
    key = "A|trainedOn|B"
    assert o.edge_attrs[key]["color"] == "#ef4444"


def test_overlay_marks_breaking_nodes_with_ring():
    findings = Findings([_f("A", "B", "violation"), _f("C", "B", "violation")])
    o = build_overlay(findings, plugin_name="license-compat")
    assert "B" in o.node_attrs
    assert o.node_attrs["B"]["ring_color"] == "#ef4444"
    assert o.node_attrs["B"]["badge"] == "2"


def test_overlay_empty_findings_empty_payload():
    o = build_overlay(Findings([]), plugin_name="license-compat")
    assert o.edge_attrs == {}
    assert o.node_attrs == {}
