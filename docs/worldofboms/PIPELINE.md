# worldofBOMs — End-to-End Pipeline Walkthrough

A request enters at the CLI or web layer, flows through the resolver,
either short-circuits on a cache hit or runs the existing generator,
then writes its result into the graph. This doc traces a real
`aikaboom generate` invocation from input string to graph quad.

## 1. User invocation

```
$ aikaboom generate --type ai --repo mistralai/Mistral-7B-v0.1 --cache prompt
```

Argparse dispatches to `cmd_generate` (`src/aikaboom/cli.py:164`).

## 2. Identifier collection

`cmd_generate` collects whichever of `--repo`, `--arxiv`, `--github` are
present and builds `Identifier(platform, value)` tuples. For the example
above it produces:

```python
[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")]
```

## 3. Canonicalization

`BomStore.resolve` calls `canonicalize_set` (`src/aikaboom/store/naming.py`)
on each identifier:

```
"mistralai/Mistral-7B-v0.1"  →  lowercase  →  alias-resolve owner  →
"mistralai/mistral-7b-v0.1"
```

## 4. Cross-identifier lookup

The resolver runs a SPARQL VALUES query that asks "do any artifacts have
*any* of these (platform, value) pairs?" (`src/aikaboom/store/store.py`,
`_resolve` function). Three branches:

- **No matches** → return `ResolveResult(existing_artifact=None, …)`.
- **One match** → return it with the matching claims.
- **Multiple matches** → return the first with collision pointers for the
  others; the user can `aikaboom graph merge` later.

## 5. Cache decision

`cache_resolver.decide` (`src/aikaboom/store/cache_resolver.py`) maps
(ResolveResult, CachePolicy, interactive?) → "use" | "generate".

- `--cache use` / `--cache auto` → always use the most recent claim.
- `--cache regen` → always generate.
- `--cache prompt` (default in TTY) → render the two-option prompt.

## 6. On "use" — reconstruct + return

The cached BOM JSON is reconstructed via `rdf_to_bom` and returned. An
implicit-use vote is recorded silently (`BomStore.record_trust_vote`).

## 7. On "generate" — run the existing pipeline

The existing `AIBOMProcessor.process_ai_model` (or `DATABOMProcessor`)
runs unchanged. Result: a BOM JSON dict.

## 8. Mapping to RDF

`bom_to_rdf(bom_json, run_meta, identifiers)` produces an `rdflib.Dataset`:

- Mints `Artifact` / `ArtifactVersion` / `BOMClaim` / `GenerationRun` IRIs.
- Adds tier edges (`hasVersion`, `hasClaim`, `generatedBy`).
- Walks each direct/rag field and emits a triple + RDF-star annotation
  carrying `assertedBy` source and `conflictKind`.

## 9. Persistence

`BomStore.save_claim` flattens the dataset into quads and hands them to
the backend's `add_quads`. Oxigraph appends to its on-disk store;
RDFLib flushes to N-Quads atomically.

## 10. Existing exports still run

JSON / SPDX / CycloneDX output paths are unchanged. The graph write is an
additional sink, not a replacement.

## Key file references

| Step | File:line |
|---|---|
| Identifier collection | `src/aikaboom/cli.py:164–230` |
| Canonicalization | `src/aikaboom/store/naming.py` |
| Resolve | `src/aikaboom/store/store.py` (`_resolve`) |
| Cache decision | `src/aikaboom/store/cache_resolver.py` (`decide`) |
| Reconstruction | `src/aikaboom/store/mapper.py` (`rdf_to_bom`) |
| Existing generation | `src/aikaboom/core/processors.py:495` (`process_ai_model`) |
| Mapping | `src/aikaboom/store/mapper.py` (`bom_to_rdf`) |
| Persistence | `src/aikaboom/store/store.py` (`save_claim`) |
