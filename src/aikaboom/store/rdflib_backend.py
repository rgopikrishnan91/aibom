"""RDFLib + N-Quads fallback backend."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from rdflib import Dataset


_NQ_FILE = "store.nq"


class RDFLibBackend:
    def __init__(self, store_dir: Path):
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._nq_path = self._store_dir / _NQ_FILE
        self._ds = Dataset()
        if self._nq_path.exists() and self._nq_path.stat().st_size > 0:
            self._ds.parse(self._nq_path, format="nquads")

    def _flush(self) -> None:
        """Atomically rewrite the N-Quads file.

        Writes to a tempfile in the same directory, then `os.replace`s it
        over the destination — same-filesystem rename is atomic on POSIX.
        On serialization failure, the tempfile is unlinked.
        """
        tmp = tempfile.NamedTemporaryFile(
            mode="wb", dir=self._store_dir, delete=False, suffix=".nq.tmp"
        )
        try:
            try:
                self._ds.serialize(destination=tmp, format="nquads")
            finally:
                tmp.close()
            os.replace(tmp.name, self._nq_path)
        except BaseException:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise

    def update(self, sparql: str) -> None:
        # rdflib 7.x's Dataset.update() crashes on INSERT DATA without a
        # GRAPH clause (3-tuple/4-tuple unpacking bug in evalInsertData).
        # Routing through the default graph sidesteps the broken path while
        # still letting GRAPH-qualified updates target named graphs.
        self._ds.default_graph.update(sparql)
        self._flush()

    def ask(self, sparql: str) -> bool:
        return bool(self._ds.query(sparql).askAnswer)

    def select(self, sparql: str) -> Iterator[Mapping[str, object]]:
        """Run SPARQL SELECT, yielding row bindings with Literals unwrapped to Python values."""
        from rdflib.term import Literal as _Literal
        def _unwrap(term):
            if isinstance(term, _Literal):
                return term.toPython()
            return term
        for row in self._ds.query(sparql):
            yield {str(var): _unwrap(row[var]) for var in row.labels}

    def add_quads(self, quads: Iterable[tuple]) -> None:
        """Bulk-add triples or quads (see GraphBackend Protocol)."""
        for quad in quads:
            if len(quad) == 4:
                s, p, o, g = quad
                if g is None:
                    self._ds.add((s, p, o))
                else:
                    self._ds.add((s, p, o, g))
            elif len(quad) == 3:
                self._ds.add(quad)
            else:
                raise ValueError(f"Expected triple or quad tuple, got {len(quad)}-tuple")
        self._flush()

    def export(self, path: Path, fmt: str = "nquads") -> None:
        fmt_map = {"nquads": "nquads", "jsonld": "json-ld"}
        self._ds.serialize(destination=str(path), format=fmt_map[fmt])

    def import_(self, path: Path, fmt: str = "nquads") -> None:
        fmt_map = {"nquads": "nquads", "jsonld": "json-ld"}
        self._ds.parse(str(path), format=fmt_map[fmt])
        self._flush()

    def close(self) -> None:
        self._flush()
