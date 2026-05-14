"""worldofBOMs knowledge graph store.

Persists generated BOMs as RDF, dedupes by canonical artifact identity,
and accumulates trust signal silently. See docs/worldofboms/CONCEPT.md.
"""

__all__ = ["BomStore", "GraphBackend"]
