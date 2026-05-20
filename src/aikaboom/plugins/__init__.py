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


class _registry:
    """Mutable container so monkeypatching in tests is straightforward."""
    _plugins: dict = {}


def register(plugin: Plugin) -> None:
    """Register a plugin instance. Raises ValueError if name collides."""
    if plugin.name in _registry._plugins:
        raise ValueError(f"Plugin {plugin.name!r} already registered")
    _registry._plugins[plugin.name] = plugin


def all_plugins() -> list[Plugin]:
    """Return every registered plugin in insertion order."""
    return list(_registry._plugins.values())


def get(name: str) -> Plugin | None:
    """Look up a plugin by name."""
    return _registry._plugins.get(name)


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
