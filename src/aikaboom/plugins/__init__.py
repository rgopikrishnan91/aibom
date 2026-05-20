"""aibom plugin registry. Plugins self-register at import time."""
from __future__ import annotations

from aikaboom.plugins.base import (
    ConflictRecord,
    Findings,
    GraphOverlay,
    Plugin,
    Scope,
    TabSpec,
)


class _Registry:
    """Class wrapper (not a module-level dict) so tests can monkeypatch
    ._plugins to isolate the registry per-test.
    """
    _plugins: dict = {}


def register(plugin: Plugin) -> None:
    """Register a plugin instance. Raises ValueError if name collides."""
    if plugin.name in _Registry._plugins:
        raise ValueError(f"Plugin {plugin.name!r} already registered")
    _Registry._plugins[plugin.name] = plugin


def all_plugins() -> list[Plugin]:
    """Return every registered plugin in insertion order."""
    return list(_Registry._plugins.values())


def get(name: str) -> Plugin | None:
    """Look up a plugin by name."""
    return _Registry._plugins.get(name)


__all__ = [
    "ConflictRecord",
    "Findings",
    "GraphOverlay",
    "Plugin",
    "Scope",
    "TabSpec",
    "register",
    "all_plugins",
    "get",
]
