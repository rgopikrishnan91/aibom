"""Plugin registry contract tests."""
from __future__ import annotations

import pytest

from aikaboom.plugins import all_plugins, get, register
from aikaboom.plugins.base import Plugin


class _DummyPlugin:
    name = "dummy-test"

    def enabled(self) -> bool:
        return True

    def analyze(self, store, scope):
        return None

    def register_cli(self, parent_subparsers) -> None:
        return None

    def web_blueprint(self):
        return None

    def bom_viewer_tab(self):
        return None

    def spdx_annotations(self, claim_iri, findings):
        return []

    def graph_overlay(self, findings):
        return None

    def conflict_findings(self, findings):
        return []


@pytest.fixture(autouse=True)
def _isolate_registry(monkeypatch):
    """Each test gets a fresh registry."""
    from aikaboom.plugins import _Registry
    monkeypatch.setattr(_Registry, "_plugins", {})


def test_register_and_retrieve():
    p = _DummyPlugin()
    register(p)
    assert get("dummy-test") is p


def test_all_plugins_returns_registered_plugins():
    p1 = _DummyPlugin()
    p2 = _DummyPlugin()
    p2.name = "dummy-test-2"
    register(p1)
    register(p2)
    names = {p.name for p in all_plugins()}
    assert names == {"dummy-test", "dummy-test-2"}


def test_duplicate_registration_raises():
    register(_DummyPlugin())
    with pytest.raises(ValueError, match="already registered"):
        register(_DummyPlugin())


def test_get_unknown_plugin_returns_none():
    assert get("nope") is None


def test_protocol_recognises_dummy():
    # _DummyPlugin implements all eight hooks; this asserts that
    # @runtime_checkable Protocol accepts a duck-typed class with the full
    # surface (the canonical case for our real plugins).
    p = _DummyPlugin()
    assert isinstance(p, Plugin)


def test_all_plugins_preserves_insertion_order():
    first = _DummyPlugin()
    second = _DummyPlugin()
    second.name = "second"
    third = _DummyPlugin()
    third.name = "third"
    register(first)
    register(second)
    register(third)
    assert [p.name for p in all_plugins()] == ["dummy-test", "second", "third"]
