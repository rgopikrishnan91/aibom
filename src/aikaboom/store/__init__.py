"""worldofBOMs knowledge graph store.

Persists generated BOMs as RDF, dedupes by canonical artifact identity,
and accumulates trust signal silently. See docs/worldofboms/CONCEPT.md.
"""

from aikaboom.store.store import BomStore
from aikaboom.store.backend import GraphBackend

__all__ = ["BomStore", "GraphBackend"]
