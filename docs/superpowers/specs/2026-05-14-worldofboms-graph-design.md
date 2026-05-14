# worldofBOMs — Knowledge Graph + Graph Store Backend

**Status:** Draft for review
**Date:** 2026-05-14
**Author:** Gopi Krishnan Rajbahadur (with Claude)

## Problem

AIkaBoOM generates per-artifact BOMs (HuggingFace models, datasets, GitHub repos, arXiv papers) and writes them to disk as JSON, SPDX 3.0.1 JSON-LD, and CycloneDX 1.6. Each generation re-runs the full pipeline — metadata fetch, RAG/Direct LLM, conflict detection — even when the upstream artifact has not changed since the last run. There is no cross-session memory: two users (or the same user two days apart) pay the LLM cost twice for the same model at the same commit.

BOMs are already graph-shaped. A model BOM points to dataset BOMs (`trainedOn`), papers (`describedIn`), code repos (`hostedAt`), licenses, suppliers. The existing "recursive child BOM" feature traverses this graph implicitly. SPDX 3.0.1 is RDF. The triplet field model `{value, source, conflict}` is an attempt to represent provenance-bearing claims in JSON.

We can collapse all of this into a single persistent knowledge graph: store generated BOMs as RDF, cache by stable artifact identity, and let "world of BOMs" emerge as the union of every BOM anyone in the community has ever generated.

## Goals

1. **Persistent retrieval.** A generated BOM never has to be recomputed for the same `(artifact, version)`.
2. **Graph-native.** Relationships between artifacts (model → dataset, paper → model, supplier → model) are first-class queryable edges, not strings in JSON fields.
3. **Provenance-preserving.** The triplet `{value, source, conflict}` round-trips losslessly into RDF and back. Every claim retains the source that asserted it.
4. **Local-first, shareable.** Each install has a local graph. Two instances can sync by exchanging a dump file. No central registry required.
5. **Pluggable backend.** Default Oxigraph; RDFLib+SQLite as fallback for environments where Oxigraph wheels are unavailable (HF Spaces, restricted CI).

## Non-Goals (v1)

- Public registry or federation protocol.
- SPARQL HTTP endpoint.
- Browser graph visualizer.
- Schema migration tooling.
- Multi-tenant auth / ACLs.
- Replacing the existing JSON / SPDX / CycloneDX exports — those stay.

## Design Calls

| Decision | Choice | Rationale |
|---|---|---|
| Storage paradigm | RDF triplestore | SPDX 3.0.1 is already RDF; reusing standard vocab (SPDX, PROV-O, DCAT) avoids re-inventing ontology. |
| Default backend | Oxigraph (embedded) | Rust core, SPARQL 1.1, on-disk persistence, Python bindings, no server to run. |
| Fallback backend | RDFLib + SQLite | Pure-Python, ships in HF Spaces and restricted environments. |
| Provenance encoding | RDF-star quoted triples | Direct 1:1 mapping from `{value, source, conflict}` to triple + annotations. No reified `FieldClaim` nodes needed. |
| Identity scheme | `bom:<platform>/<id>@<version>` | Stable IRIs keyed by platform, repo id, and commit SHA (existing `packageVersion`). Versionless aliases redirect to latest snapshot. |
| Cache freshness | TTL (default 30 days), per-snapshot | Configurable. Forces eventual refresh even if `version` unchanged, since LLMs and prompts evolve. |
| Sync model | Export/import dump files | Local-first; no registry server in v1. Two laptops sync with `scp` + a CLI command. |

## Architecture

```
                          ┌────────────────────────┐
   CLI / Web request ──►  │  cmd_generate /        │
                          │  /api/generate route   │
                          └─────────┬──────────────┘
                                    │
                                    ▼
                          ┌────────────────────────┐
                          │ BomStore.lookup(...)   │ new
                          │   cache hit? return    │
                          └─────────┬──────────────┘
                                    │ miss
                                    ▼
                          ┌────────────────────────┐
                          │ AIBOMProcessor /       │   unchanged
                          │ DATABOMProcessor       │
                          └─────────┬──────────────┘
                                    │ BOM JSON
                                    ▼
                          ┌────────────────────────┐
                          │ BomMapper.to_rdf(bom)  │ new
                          └─────────┬──────────────┘
                                    │ RDF quads
                                    ▼
                          ┌────────────────────────┐
                          │ GraphBackend           │ new
                          │   Oxigraph (default)   │
                          │   RDFLib   (fallback)  │
                          └────────────────────────┘
                                    ▲
                                    │
                          ┌─────────┴──────────────┐
                          │ aikaboom graph CLI:    │
                          │   list / show / export │
                          │   import / stats /     │
                          │   query                │
                          └────────────────────────┘
```

## Components

New module under `src/aikaboom/store/`:

| File | Responsibility |
|---|---|
| `iris.py` | Mint stable IRIs from `(platform, id, version)`; one pure-function module so identity rules are auditable and unit-testable. |
| `vocab.py` | Local namespaces (`bom:`, `aibom:`). Reuses SPDX, PROV-O, DCAT vocab where they already cover a concept; only invents terms aikaboom needs. |
| `mapper.py` | `bom_to_rdf(bom_json) -> rdflib.Dataset` and `rdf_to_bom(iri) -> bom_json`. Round-trip is lossless for `direct_fields`, `rag_fields`, `beta_fields`, sources, and conflicts. |
| `backend.py` | `GraphBackend` Protocol — `add_quads`, `ask`, `select`, `construct`, `export`, `import_`. |
| `oxigraph_backend.py` | Default impl, on-disk store under `~/.aikaboom/graph/` (path configurable via `AIKABOOM_GRAPH_DIR`). |
| `rdflib_backend.py` | Fallback. RDFLib in-memory `Dataset` flushed to disk as N-Quads on every write (atomic rename). For typical use (10K–100K triples) this is fast enough and avoids a SQLite-store dependency that has uneven wheel coverage. |
| `store.py` | High-level `BomStore` facade — `lookup`, `save`, `find_by_license`, `find_by_dataset`, `list_snapshots`, `stats`. |
| `cli_graph.py` | `aikaboom graph` subcommands. |

Backend selection at startup: try Oxigraph; on `ImportError` or wheel-missing error, fall back to RDFLib and log once.

## Data Model

### Classes (RDF types)

- `aibom:Model` — a HuggingFace model (or equivalent)
- `aibom:Dataset` — a dataset artifact
- `aibom:Paper` — an arXiv paper
- `aibom:CodeRepo` — a GitHub repository
- `aibom:License` — reused from SPDX where possible (`spdx:License`)
- `aibom:Supplier` — org or individual; aliases the existing supplier index
- `aibom:Person` — author
- `aibom:Source` — one of `huggingface`, `github`, `arxiv` (the data source)
- `aibom:BOMSnapshot` — one generation event for one artifact
- `aibom:GenerationRun` — metadata about the LLM/provider/prompt used

### Edges

- `aibom:hasSnapshot` (Artifact → BOMSnapshot)
- `aibom:supersedes` (BOMSnapshot → BOMSnapshot)
- `aibom:generatedBy` (BOMSnapshot → GenerationRun)
- `aibom:trainedOn` (Model → Dataset)
- `aibom:describedIn` (Model → Paper)
- `aibom:hostedAt` (Model → CodeRepo)
- `aibom:hasLicense` (Artifact → License) — also `spdx:license`
- `aibom:suppliedBy` (Artifact → Supplier)
- `aibom:authoredBy` (Paper → Person)

### Field claims via RDF-star

A BOM field like:

```json
"license": {"value": "Apache-2.0", "source": "huggingface", "conflict": null}
```

becomes:

```
bom:snap/<uuid> spdx:license <https://spdx.org/licenses/Apache-2.0> .

<< bom:snap/<uuid> spdx:license <https://spdx.org/licenses/Apache-2.0> >>
    aibom:assertedBy aibom:source/huggingface ;
    aibom:conflictKind aibom:noConflict .
```

Inter-source conflict:

```
<< bom:snap/<uuid> spdx:license <...Apache-2.0> >>
    aibom:assertedBy aibom:source/huggingface ;
    aibom:conflictKind aibom:interSourceConflict ;
    aibom:conflictsWith << bom:snap/<uuid> spdx:license <...MIT> >> .
```

This collapses the triplet model 1:1 into RDF instead of inventing a reified `FieldClaim` node type. SPARQL `SELECT` over quoted triples lets queries ask "which fields had conflicts" naturally.

### Identity (IRIs)

| Artifact | IRI pattern | Example |
|---|---|---|
| HF model | `bom:hf/<owner>/<name>@<commit>` | `bom:hf/mistralai/Mistral-7B-v0.1@27d67f1b` |
| HF dataset | `bom:hf/ds/<owner>/<name>@<commit>` | `bom:hf/ds/rajpurkar/squad_v2@3ffb306f` |
| arXiv paper | `bom:arxiv/<id>@<version>` | `bom:arxiv/1911.00536@v1` |
| GitHub repo | `bom:gh/<owner>/<name>@<commit>` | `bom:gh/microsoft/DialoGPT@<sha>` |
| Snapshot | `bom:snap/<uuid4>` | — |
| Generation run | `bom:run/<uuid4>` | — |

Versionless aliases (`bom:hf/mistralai/Mistral-7B-v0.1`) exist as a separate IRI that owl:sameAs's to the *latest* snapshot's artifact IRI; updated on each new snapshot.

## Data Flow

### Cache hit path

1. Request arrives with `(platform, id)` and optional `version`.
2. If no `version`, do a single REST call to the platform API (HF / GitHub / arXiv) to resolve the current commit SHA / version. This is cheap and we already do it during generation.
3. `BomStore.lookup(artifact_iri, use_case, mode, max_age=TTL)` runs:
   ```sparql
   ASK {
     ?snap a aibom:BOMSnapshot ;
           aibom:subject <artifact_iri> ;
           aibom:useCase "<use_case>" ;
           aibom:mode "<mode>" ;
           aibom:generatedAt ?t .
     FILTER (?t > NOW() - "P30D"^^xsd:duration)
   }
   ```
4. On hit: `CONSTRUCT` the snapshot's subgraph, `rdf_to_bom` reconstructs the JSON, return. **Zero LLM calls.**
5. On miss: fall through to generation.

### Miss / generation path

1. Run the existing `process_ai_model` / `process_dataset` flow unchanged.
2. On success, `BomStore.save(bom, run_meta)`:
   - Mint a new snapshot IRI.
   - `bom_to_rdf` produces quads.
   - Single transactional write to the backend.
   - If a prior snapshot exists for the same `(artifact, use_case, mode)`, add `aibom:supersedes` edge.
3. Return the BOM JSON to the caller.

### Recursive children

`recursive_bom.generate_recursive_boms` walks the dependency tree. Each child generation goes through the same `BomStore.lookup` → cache check. Cross-model dataset sharing (two models both `trainedOn squad`) means the second model's recursive walk is free.

### Stale-but-present

If the resolved upstream version differs from any stored snapshot's version, lookup misses; a new snapshot is created and linked via `aibom:supersedes`. Old snapshots are retained for audit and never deleted automatically.

## Integration Points

| Existing code | Change |
|---|---|
| `cli.cmd_generate` (src/aikaboom/cli.py:164) | Wrap `process_ai_model` / `process_dataset` with `BomStore.lookup` check; add `--no-cache` flag. |
| Web `/api/generate` route (src/aikaboom/web/app.py) | Same wrap; honor existing `force_refresh` flag to bypass cache. |
| `recursive_bom.generate_recursive_boms` (src/aikaboom/utils/recursive_bom.py) | Each child generation goes through `BomStore.lookup` first. |
| SPDX validation/export (`src/aikaboom/utils/spdx_validator.py`, `src/aikaboom/schemas/`) | Unchanged. Mapper writes RDF into the graph store; the existing SPDX JSON-LD output is still produced from the in-memory BOM JSON. |
| CycloneDX export | Unchanged. |

The store is a **new optional layer**. If a user sets `AIKABOOM_GRAPH_DISABLE=1`, the system behaves exactly as today.

## CLI Surface

```
aikaboom graph stats              # node/edge counts, snapshot count, disk size
aikaboom graph list               # list all stored snapshots
aikaboom graph show <iri>         # pretty-print one snapshot
aikaboom graph export <file>      # dump entire graph as JSON-LD or N-Quads
aikaboom graph import <file>      # merge a dump into the local graph
aikaboom graph query <sparql>     # run an arbitrary SPARQL query
aikaboom graph rebuild            # rebuild graph from JSON files in results/
```

`aikaboom generate` gains `--no-cache` to force regeneration.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `AIKABOOM_GRAPH_DIR` | `~/.aikaboom/graph` | Where the on-disk store lives. |
| `AIKABOOM_GRAPH_BACKEND` | `auto` | `oxigraph`, `rdflib`, or `auto` (try oxigraph, fall back). |
| `AIKABOOM_GRAPH_TTL_DAYS` | `30` | Cache freshness window. |
| `AIKABOOM_GRAPH_DISABLE` | `0` | If `1`, store is bypassed entirely. |

## Error Handling

- **Backend init failure** (Oxigraph wheel missing): fall back to RDFLib, log once at startup. Generation never blocks on store availability.
- **Mapping failure on save**: log + still write the BOM JSON to disk as today. The graph is best-effort.
- **Mapping failure on load**: treat as cache miss; never return a partially-reconstructed BOM. Log the failing snapshot IRI.
- **Backend disk corruption**: `aikaboom graph rebuild` regenerates from the JSON files in `results/`. This is the recovery story.

## Testing

| Test | Asserts |
|---|---|
| `test_iris.py` | IRI minting is deterministic, URL-safe, and stable across runs. |
| `test_mapper_roundtrip.py` | `rdf_to_bom(bom_to_rdf(b)) == b` for every BOM in `Golden_Set/` and `results/`. Property test with hypothesis for generated BOMs. |
| `test_cache_hit.py` | Generate once with a stub LLM, generate again, assert zero LLM calls on second pass. |
| `test_conflict_preservation.py` | Load BOM with `inter` and `intra` conflicts, save, reload, both survive RDF-star annotation. |
| `test_export_import.py` | Export → fresh store → import → graph identity. |
| `test_backend_fallback.py` | Force Oxigraph unavailable, assert RDFLib backend takes over and tests still pass. |
| `test_recursive_cache.py` | Two models sharing a dataset: second model's recursive walk hits cache for the shared dataset. |

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Oxigraph wheels missing on some platforms (HF Spaces ARM, restricted CI) | RDFLib fallback with `AIKABOOM_GRAPH_BACKEND=auto`. |
| RDF-star support uneven across backends | Both Oxigraph and RDFLib 7+ support RDF-star. Pin minimum versions. |
| Schema evolution breaks old snapshots | Snapshots tagged with `aibom:schemaVersion`; mapper reads old versions; new versions only on write. Rebuild from JSON is always available. |
| Graph grows large (1000s of models, recursive children) | Oxigraph handles millions of triples easily. Provide `aikaboom graph stats` for visibility. |
| Cache returns stale BOM after prompt/model change | `GenerationRun.codeVersion` + `promptVersion` included in cache key. Bump prompt version → cache miss. |

## Open Questions

These are noted but not blocking for v1:

- **TTL default**: 30 days picked as a reasonable middle ground. Should this differ per artifact type (datasets change less than models)?
- **Schema versioning**: explicit `aibom:schemaVersion` on every snapshot, or implicit from `codeVersion`?
- **Anonymization for export**: should `aikaboom graph export` strip API keys / private repos by default?

## Out of Scope (v1)

- SPARQL HTTP endpoint (users can run SPARQL via `aikaboom graph query`).
- Public registry / federation protocol.
- Browser graph visualizer.
- Schema migration tooling beyond `rebuild`.
- Multi-user auth / ACLs.
- Replacing JSON / SPDX / CycloneDX exports.

## Success Criteria

1. Generating the same BOM twice (same artifact, same version, same use case, within TTL) makes zero LLM calls on the second run.
2. Round-trip JSON ↔ RDF is lossless for every BOM in `Golden_Set/`.
3. `aikaboom graph export | aikaboom graph import` on a fresh install reproduces the source graph exactly.
4. RDFLib fallback is exercised in CI and produces identical results to Oxigraph.
5. No regression in existing CLI/web tests with graph disabled (`AIKABOOM_GRAPH_DISABLE=1`).
