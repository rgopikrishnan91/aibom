"""GraphOverlay payload builder: component ring tinting by worst AVID tier."""
from __future__ import annotations

from aikaboom.plugins import GraphOverlay

# Mirrors license_compat/overlay.py palette — red/orange/yellow for tier 1/2/3.
_RING_BY_TIER = {1: "#ef4444", 2: "#e80000", 3: "#ec0000"}

# Use human-readable colours that match the intent described in the spec.
_RING_COLORS = {
    1: "#ef4444",  # red  — affected
    2: "#f97316",  # orange — under investigation (tier 2)
    3: "#eab308",  # yellow — family/prefix match (tier 3)
}


def build_overlay(findings, plugin_name: str) -> GraphOverlay:
    """Build a GraphOverlay colouring each component by its worst (lowest) tier."""
    worst: dict[str, int] = {}
    for f in findings:
        cur = worst.get(f.component_iri)
        if cur is None or f.tier < cur:  # lower tier number = worse
            worst[f.component_iri] = f.tier

    node_attrs = {
        iri: {"ring_color": _RING_COLORS[tier], "badge": "AVID"}
        for iri, tier in worst.items()
    }
    return GraphOverlay(plugin_name=plugin_name, node_attrs=node_attrs)
