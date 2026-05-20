"""GraphBackend Protocol + selection."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Protocol

_log = logging.getLogger(__name__)


class GraphBackend(Protocol):
    """Minimal interface every backend must implement."""

    def update(self, sparql: str) -> None:
        """Run a SPARQL UPDATE."""
        ...

    def ask(self, sparql: str) -> bool:
        """Run a SPARQL ASK and return the boolean result."""
        ...

    def select(self, sparql: str) -> Iterator[Mapping[str, object]]:
        """Run a SPARQL SELECT and yield row bindings."""
        ...

    def add_quads(self, quads: Iterable[tuple]) -> None:
        """Bulk-add triples or quads.

        Each element of `quads` may be a 3-tuple `(s, p, o)` — interpreted as
        a triple in the default graph — or a 4-tuple `(s, p, o, g)` where `g`
        is a named-graph term or `None` for the default graph.
        """
        ...

    def export(self, path: Path, fmt: str = "nquads") -> None:
        """Dump the entire store to a file."""
        ...

    def import_(self, path: Path, fmt: str = "nquads") -> None:
        """Merge a dump file into the store."""
        ...

    def close(self) -> None:
        """Release any resources."""
        ...


def _store_dir() -> Path:
    return Path(os.environ.get("AIKABOOM_GRAPH_DIR", str(Path.home() / ".aikaboom" / "graph")))


def open_backend() -> GraphBackend:
    """Open the configured backend, falling back to RDFLib if Oxigraph is unavailable."""
    requested = os.environ.get("AIKABOOM_GRAPH_BACKEND", "auto").lower()
    store_dir = _store_dir()
    store_dir.mkdir(parents=True, exist_ok=True)

    if requested in ("oxigraph", "auto"):
        try:
            from aikaboom.store.oxigraph_backend import OxigraphBackend

            return OxigraphBackend(store_dir)
        except ImportError as e:
            if requested == "oxigraph":
                raise
            _log.warning("Oxigraph unavailable (%s); falling back to RDFLib", e)

    from aikaboom.store.rdflib_backend import RDFLibBackend

    return RDFLibBackend(store_dir)
