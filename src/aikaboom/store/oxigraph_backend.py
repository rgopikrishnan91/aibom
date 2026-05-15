"""Oxigraph backend (default)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Mapping

try:
    import pyoxigraph as _ox
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    raise ImportError("pyoxigraph is not installed; pip install pyoxigraph") from e


def _format_for(fmt: str) -> "_ox.RdfFormat":
    """Map a friendly format string to a pyoxigraph RdfFormat enum."""
    key = fmt.lower().replace("-", "").replace("_", "")
    if key in ("nquads", "nq"):
        return _ox.RdfFormat.N_QUADS
    if key in ("ntriples", "nt"):
        return _ox.RdfFormat.N_TRIPLES
    if key in ("turtle", "ttl"):
        return _ox.RdfFormat.TURTLE
    if key in ("trig",):
        return _ox.RdfFormat.TRIG
    if key in ("jsonld", "ldjson"):
        return _ox.RdfFormat.JSON_LD
    if key in ("rdfxml", "xml"):
        return _ox.RdfFormat.RDF_XML
    raise ValueError(f"Unsupported RDF format: {fmt!r}")


def _unwrap(term: object) -> object:
    """Unwrap a pyoxigraph term so that ``str(term)`` yields the user-facing value.

    Literals stringify to their N-Triples form (``"value"^^datatype``) by default,
    which is unfriendly. We surface the literal's ``value`` instead. IRIs and
    blank nodes stringify acceptably so we return them unchanged.
    """
    if isinstance(term, _ox.Literal):
        return term.value
    return term


class OxigraphBackend:
    """Oxigraph-backed implementation of :class:`GraphBackend`."""

    def __init__(self, store_dir: Path):
        self._store_dir = Path(store_dir)
        self._store = _ox.Store(path=str(self._store_dir))

    def update(self, sparql: str) -> None:
        self._store.update(sparql)

    def ask(self, sparql: str) -> bool:
        return bool(self._store.query(sparql))

    def select(self, sparql: str) -> Iterator[Mapping[str, object]]:
        results = self._store.query(sparql)
        variables = [v.value for v in results.variables]
        for solution in results:
            row: dict[str, object] = {}
            for var in variables:
                term = solution[var]
                if term is None:
                    row[var] = None
                else:
                    row[var] = _unwrap(term)
            yield row

    def add_quads(self, quads: Iterable[tuple]) -> None:
        default_graph = _ox.DefaultGraph()
        for quad in quads:
            if len(quad) == 4:
                s, p, o, g = quad
            elif len(quad) == 3:
                s, p, o = quad
                g = None
            else:
                raise ValueError(f"Expected triple or quad tuple, got {len(quad)}-tuple")
            self._store.add(_ox.Quad(s, p, o, g if g is not None else default_graph))

    def export(self, path: Path, fmt: str = "nquads") -> None:
        rdf_fmt = _format_for(fmt)
        with open(path, "wb") as fh:
            self._store.dump(fh, rdf_fmt)

    def import_(self, path: Path, fmt: str = "nquads") -> None:
        rdf_fmt = _format_for(fmt)
        with open(path, "rb") as fh:
            self._store.bulk_load(fh, rdf_fmt)

    def close(self) -> None:
        # pyoxigraph Store has no explicit close; flush via reference drop.
        pass
