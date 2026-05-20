"""License-compat graph overlay (filled in Task 11)."""
from aikaboom.plugins import GraphOverlay


def build_overlay(findings, plugin_name):
    return GraphOverlay(plugin_name=plugin_name, edge_attrs={}, node_attrs={})
