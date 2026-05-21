"""Per-plugin contract test: every hook exists and returns the documented type."""
from __future__ import annotations

import pytest

from aikaboom.plugins import (
    ConflictRecord,
    GraphOverlay,
    Plugin,
    Scope,
    TabSpec,
    all_plugins,
    get,
)
from aikaboom.plugins.license_compat.engine import Findings


@pytest.fixture
def empty_findings():
    return Findings([])


def test_plugin_is_protocol_compatible():
    for p in all_plugins():
        assert isinstance(p, Plugin)


def test_plugin_enabled_returns_bool():
    for p in all_plugins():
        assert isinstance(p.enabled(), bool)


def test_plugin_web_blueprint_returns_blueprint_or_none():
    from flask import Blueprint
    for p in all_plugins():
        bp = p.web_blueprint()
        assert bp is None or isinstance(bp, Blueprint)


def test_plugin_bom_viewer_tab_returns_tabspec_or_none():
    for p in all_plugins():
        tab = p.bom_viewer_tab()
        assert tab is None or isinstance(tab, TabSpec)


def test_plugin_graph_overlay_returns_overlay(empty_findings):
    for p in all_plugins():
        overlay = p.graph_overlay(empty_findings)
        assert isinstance(overlay, GraphOverlay)


def test_plugin_conflict_findings_returns_list(empty_findings):
    for p in all_plugins():
        out = p.conflict_findings(empty_findings)
        assert isinstance(out, list)
        for entry in out:
            assert isinstance(entry, ConflictRecord)


def test_license_compat_plugin_is_registered():
    p = get("license-compat")
    assert p is not None
    assert p.name == "license-compat"
