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
        """Atomically rewrite the N-Quads file."""
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=self._store_dir, delete=False, suffix=".nq.tmp"
        ) as tmp:
            self._ds.serialize(destination=tmp, format="nquads")
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, self._nq_path)

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
        for row in self._ds.query(sparql):
            yield {str(var): row[var] for var in row.labels}

    def add_quads(self, quads: Iterable[tuple]) -> None:
        for s, p, o, g in quads:
            self._ds.add((s, p, o, g))
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
