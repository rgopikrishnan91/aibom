"""Plugin Protocol and supporting dataclasses for the aibom plugin system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    import argparse
    from flask import Blueprint


@dataclass(frozen=True)
class Scope:
    """Analysis scope for a plugin run."""
    kind: str  # "single" | "graph_wide"
    artifact_iri: Optional[str] = None
    depth: int = 5

    @classmethod
    def single(cls, artifact_iri: str, depth: int = 5) -> "Scope":
        return cls(kind="single", artifact_iri=artifact_iri, depth=depth)

    @classmethod
    def graph_wide(cls) -> "Scope":
        return cls(kind="graph_wide")


@dataclass(frozen=True)
class TabSpec:
    """Descriptor for a tab a plugin contributes to the BOM viewer."""
    label: str
    url_template: str  # e.g. "/license-compat/{artifact_id}"
    sort_order: int = 100


@dataclass(frozen=True)
class ConflictRecord:
    """Entry the plugin contributes to the existing Conflicts tab."""
    category: str
    severity: str  # "high" | "medium" | "low" | "info"
    subject_iri: str
    title: str
    detail: str
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GraphOverlay:
    """Payload for the graph-view edge/node tinting overlay."""
    plugin_name: str
    edge_attrs: dict = field(default_factory=dict)  # (s, p, o) tuple-as-str -> {color, label, tooltip}
    node_attrs: dict = field(default_factory=dict)  # iri -> {badge, ring_color}


class Findings(Protocol):
    """Result type of plugin.analyze(). Implementations supply iteration helpers."""

    def to_dict(self) -> dict: ...
    def violations(self) -> list: ...


@runtime_checkable
class Plugin(Protocol):
    """All plugins implement this surface. Hooks return None or empty if not used."""

    name: str

    def enabled(self) -> bool: ...

    def analyze(self, store: Any, scope: Scope) -> Findings: ...

    def register_cli(self, parent_subparsers: "argparse._SubParsersAction") -> None: ...

    def web_blueprint(self) -> Optional["Blueprint"]: ...

    def bom_viewer_tab(self) -> Optional[TabSpec]: ...

    def spdx_annotations(self, claim_iri: str, findings: Findings) -> list[dict]: ...

    def graph_overlay(self, findings: Findings) -> GraphOverlay: ...

    def conflict_findings(self, findings: Findings) -> list[ConflictRecord]: ...
