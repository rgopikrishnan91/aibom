# worldofBOMs — Python API Reference

## BomStore

The high-level facade. Open one per process; safe to share across threads
if the backend allows it (Oxigraph does; RDFLib's in-memory dataset does
not — guard externally if multi-threaded).

```python
from aikaboom.store import BomStore
from aikaboom.store.naming import Identifier

store = BomStore.open()
try:
    claim_iri = store.save_claim(
        bom_json,
        run_meta={
            "provider": "openrouter",
            "llm_model": "anthropic/claude-3-haiku",
            "prompt_version": "v12",
            "code_version": "abc1234",
            "mode": "rag",
            "use_case": "license",
        },
        identifiers=[
            Identifier("huggingface", "mistralai/Mistral-7B-v0.1"),
        ],
    )
finally:
    store.close()
```

### Methods

- `BomStore.open() -> BomStore` — open the configured backend.
- `save_claim(bom_json, run_meta, identifiers) -> str` — persist a BOM,
  return the new BOMClaim IRI.
- `find_claims_for(identifiers, use_case=None, mode=None) -> list[dict]` —
  find existing claims, newest first. Each dict has `iri`, `created_at`,
  `llm_model`.
- `stats() -> dict[str, int]` — counts of `artifacts`, `versions`,
  `claims`, `votes`.
- `reconstruct_bom(claim_iri) -> dict` — rebuild the BOM JSON dict from
  a stored claim. Used by the recursive walker to read cached BOMs.
- `close()` — flush and release resources.

## GraphBackend (Protocol)

You normally don't interact with this directly — `BomStore` is the front
door. If you're embedding the store in a different context, the Protocol
is:

```python
class GraphBackend(Protocol):
    def update(self, sparql: str) -> None: ...
    def ask(self, sparql: str) -> bool: ...
    def select(self, sparql: str) -> Iterator[Mapping[str, object]]: ...
    def add_quads(self, quads: Iterable[tuple]) -> None: ...
    def export(self, path: Path, fmt: str = "nquads") -> None: ...
    def import_(self, path: Path, fmt: str = "nquads") -> None: ...
    def close(self) -> None: ...
```

## BomMapper functions

- `bom_to_rdf(bom_json, run_meta, identifiers) -> (Dataset, claim_iri)`
- `rdf_to_bom(ds, claim_iri) -> dict`

Round-trip is asserted lossless by `tests/store/test_mapper_roundtrip.py`
for every BOM in `Golden_Set/` and `results/`.

## Naming helpers

- `Identifier(platform, value)` — a typed identifier dataclass.
- `canonicalize(ident) -> Identifier` — apply the canonicalization pipeline.
- `canonicalize_set(ids) -> list[Identifier]` — dedup-aware canonicalize.
- `pick_primary(ids) -> Identifier` — pick the highest-priority platform.

`PLATFORM_PRIORITY = ("huggingface", "github", "arxiv", "doi", "url")`.
