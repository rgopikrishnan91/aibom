# worldofBOMs — Knowledge Graph Visualization + Edge Connectivity

**Status:** Draft v1 for review
**Date:** 2026-05-15
**Author:** Gopi Krishnan Rajbahadur (with Claude)
**Builds on:** Draft PR #48 (`worldofboms-graph` branch) — the RDF/Oxigraph store.
This work is committed to that same branch.

## Problem

PR #48 added a persistent RDF knowledge graph: every generated BOM becomes an
`Artifact → ArtifactVersion → BOMClaim` subgraph, deduped by canonical artifact
identity. But two things are missing for it to be a *knowledge graph* a user can
actually see and use:

1. **The graph is disconnected.** `store/vocab.py` defines `trainedOn` but the
   mapper (`store/mapper.py`) never emits any artifact-to-artifact edge. Each
   saved BOM is an isolated star — its Artifact, that Artifact's versions and
   claims, and the claim's field literals. A model that was trained on a dataset
   already in the graph has no edge to it. `testedOn` and `dependsOn` are not
   even defined in the vocab.

2. **There is no way to see it.** The original worldofBOMs spec
   (`2026-05-14-worldofboms-graph-design.md`) explicitly lists "Browser graph
   visualizer" as a v1 non-goal. Today the only views are `aikaboom graph stats`
   / `list` / `show` and raw SPARQL.

This spec covers both: making the graph genuinely connected, and adding a
browser visualization that mirrors the existing recursive-BOM graph tab.

## Goals

1. **A connected graph.** Artifact-to-artifact edges (`trainedOn`, `testedOn`,
   `dependsOn`) are persisted, so the worldofBOMs graph is one connected
   structure rather than disconnected stars.
2. **It grows on every generation.** Edge creation runs at every BOM save, not
   only in recursive mode. Generating a model BOM that names a known dataset
   connects to that dataset's existing node — no duplicate node.
3. **A global view.** A new `worldofBOMs` tab renders the whole graph.
4. **An ego-centric view.** Clicking any node highlights that node and its
   dependency paths (upstream / downstream / both), dims the rest, and pops up
   that node's reconstructed BOM — mirroring the recursive-tab interaction.
5. **Lineage queries.** Preset queries answer "licenses across the lineage",
   "all datasets in the lineage", "all models in the lineage", "conflicts in the
   lineage", each scoped by direction. A raw SPARQL box covers power users.
6. **Downloadable.** The whole graph, or any ego view (upstream / downstream /
   both), exports as an SPDX 3.0.1 linked bundle.
7. **No collisions.** One node per artifact (store-level identity dedup
   guarantees this); layout spacing prevents visual overlap; placeholder
   artifacts are visually distinct; label clashes between distinct artifacts are
   disambiguated.

## Non-Goals (v1)

- Editing the graph from the browser (no node delete / merge UI — `graph merge`
  stays a CLI operation).
- Surfacing trust scores. Trust stays silent in v1, consistent with the original
  worldofBOMs spec — the visualization does not display `trustScore`.
- A SPARQL HTTP endpoint for external clients. The raw SPARQL box is an in-app
  convenience, server-mediated, not a public endpoint.
- Real-time multi-user graph streaming. The graph refreshes on tab activation
  and after a local generation finishes; no websockets.
- Graph-layout persistence across sessions.

## Part A — Graph Connectivity

### A.1 Vocab additions

`store/vocab.py` gains two predicates under the `AIBOM` namespace:

```python
testedOn  = AIBOM.testedOn   # Model  → Dataset
dependsOn = AIBOM.dependsOn   # Model → Model, Dataset → Dataset
```

`trainedOn` already exists. `describedIn` / `hostedAt` / `suppliedBy` /
`authoredBy` remain defined but unused — out of scope here. `docs/worldofboms/
SCHEMA.md` gains the two new rows in the same edit (the existing
`test_docs_schema_parity` test enforces vocab↔doc parity, so this is mandatory,
not optional).

### A.2 Relationship-field source

The mapping from BOM field to edge predicate already exists in
`utils/recursive_bom.py` and is reused verbatim — not duplicated:

| BOM field            | Edge predicate | Endpoint kinds        |
|----------------------|----------------|-----------------------|
| `trainedOnDatasets`  | `trainedOn`    | Model → Dataset       |
| `testedOnDatasets`   | `testedOn`     | Model → Dataset       |
| `modelLineage`       | `dependsOn`    | Model → Model         |
| `sourceInfo`         | `dependsOn`    | Dataset → Dataset     |

These constants (`AI_RELATIONSHIP_FIELDS`, `DATA_RELATIONSHIP_FIELDS`) are
imported from `recursive_bom.py` so there is a single source of truth. The
existing `_is_walkable_target` filter in `recursive_bom.py` — which already
drops arXiv/DOI noise from `sourceInfo` — is reused to decide which field values
become edges.

### A.3 Edge persistence at save time

A new function in `store/mapper.py` (or a small `store/edges.py` helper):

```
add_relationship_edges(store, source_artifact_iri, bom_json) -> list[Edge]
```

Called from `BomStore.save_claim` after the artifact subgraph is written. For
each relationship field present in `bom_json`:

1. Split the field value into individual targets (it may be a list or a
   delimited string), apply `_is_walkable_target`.
2. Canonicalize each target to an `Identifier`. If the value carries a
   platform-qualified handle (e.g. a HuggingFace dataset id), use that platform;
   otherwise fall back to `name-only`.
3. Resolve the target to an Artifact IRI, in order:
   a. `store.resolve([identifier])` — identifier match. This is the cache-hit
      path; it matches on platform handles only.
   b. If 3a misses and the target is `name-only`, a **name-label match**: look
      for an existing non-placeholder Artifact whose canonicalized
      `canonicalLabel` exactly equals the canonicalized target name. If found,
      link to it — no placeholder is minted.
   c. If 3a and 3b both miss, mint a placeholder Artifact (`isPlaceholder true`,
      `name-only` identifier), exactly as the recursive walker's
      unresolved-target path already does. Then run the confidence-triage check
      (A.6): if it reports a confident-but-inexact match against an existing
      artifact, add a soft `potentialDuplicateOf` edge — but do not merge.
4. Add the triple `(source_artifact, <predicate>, target_artifact)`.

Edge writes are idempotent — saving the same BOM twice does not create
duplicate triples (an `ASK` guard before `INSERT`, matching the store's existing
best-effort persistence style).

### A.4 "Connects to a dataset we already have a BOM for"

Connectivity is automatic in both directions. Note that the plain cache-hit
`resolve()` is **identifier**-based — on its own it does not match a free-text
`name-only` placeholder against a platform-handle artifact. The name-label match
(A.3 step 3b) is what bridges that gap.

- **A new model references a known dataset.** A.3 step 3 finds the dataset's
  existing node — by identifier (3a) or by name (3b) — and the `trainedOn` edge
  lands directly on it. No placeholder, no duplicate node.

- **A new dataset BOM arrives for an existing placeholder.** When an earlier
  model BOM minted a `name-only` placeholder for a dataset, and that dataset is
  later generated for real, `save_claim` runs the name-label match *in reverse*:
  it looks for placeholder Artifacts whose canonicalized name exactly equals the
  new real artifact's canonicalized label. Each match is **merged** into the new
  artifact. So yes — *if the name resolves, we merge*: the placeholder is
  promoted in place and every incoming edge that pointed at it (the model's
  `trainedOn`/`testedOn`/`dependsOn`) now points at the real dataset node.

The merge reuses the store's existing `graph merge` logic (`cmd_graph_merge`
already transfers all incoming edges then deletes the stale node). That logic is
extracted from `cli_graph.py` into a `BomStore.merge_artifacts(into, from_)`
method so both the CLI and `save_claim` call one implementation.

The name-label match is **exact equality on canonicalized identity names —
deliberately not fuzzy.** Auto-merging on a fuzzy match risks collapsing two
genuinely different artifacts. The fuzzy / confidence-based matching the
codebase already has is still used — but only to *suggest* a merge, never to
perform one. See A.6.

`aikaboom graph rebuild` already replays every `results/*.json`; because edge
creation lives inside `save_claim`, a rebuild reconstructs all edges
retroactively with no extra code.

### A.5 Recursive walker

Recursive child BOMs are persisted through the same `save_claim` path, so their
edges form automatically — no walker-specific edge code. The walker's
parent→child relationship is exactly the `trainedOn`/`testedOn`/`dependsOn` edge
that A.3 already writes from the parent's relationship fields.

### A.6 Name matching: identity vs confidence triage

The codebase already separates two name-matching layers, and worldofBOMs edge
resolution keeps them separate rather than collapsing them into one rule:

- **Identity layer** — `naming.canonicalize` / `BomStore.resolve`. Deterministic
  and conservative: URL stripping, lowercasing, separator collapse, and only
  *expanding* owner-alias rewrites. This decides node identity and the stable
  IRI. A.3 step 3 and A.4's reverse promotion use **only this layer** — an exact
  match here is the *only* thing that triggers an automatic merge, because a
  merge permanently collapses two nodes and a wrong merge is destructive.

- **Confidence-triage layer** — `SupplierAliasIndex.is_same_supplier`: a tiered
  resolver (curated cross-reference index → normalized-exact → Jaro-Winkler
  ≥ 0.85). `naming.py` already documents that this layer exists for "conflict
  checks and provenance, where one Artifact can be linked to several supplier
  handles *without losing identity*" — it is explicitly **not** an identity
  decision.

worldofBOMs reuses the confidence-triage layer but never lets it auto-merge.
When A.3 step 3 has minted a placeholder (the exact identity match missed) and
the triage layer reports a confident-but-inexact match against an existing
artifact, the store records a soft **`aibom:potentialDuplicateOf`** edge between
the two — the predicate already exists in the vocab, and the original
worldofBOMs spec already uses it for hard multi-match "soft collisions". It does
not merge.

In the UI these soft edges render as dashed links with a "possible duplicate of
<artifact>" hint on the placeholder node (C.6). The user confirms a genuine
duplicate with `aikaboom graph merge`, which routes through the same
`BomStore.merge_artifacts` as A.4. The rule, in one line:

> **Exact identity match → auto-merge. Confidence-triage match → a
> `potentialDuplicateOf` hint the user confirms. Never auto-merge on a fuzzy
> score.**

## Part B — Backend (read side)

### B.1 New module: `src/aikaboom/store/graph_view.py`

All graph-read logic lives here so `web/app.py` stays thin. Pure functions over
a `BomStore`:

- `full_graph(store) -> {nodes, edges}` — every Artifact as a node
  (`iri, label, kind, is_placeholder, claim_count`) and every relationship
  triple as an edge (`source, target, predicate`).
- `ego_graph(store, artifact_iri, direction, depth) -> {nodes, edges, focus}` —
  breadth-first edge traversal from `artifact_iri`. `direction ∈ {up, down,
  both}`; `up` follows edges *into* the node, `down` follows edges *out*.
  `depth` defaults to unlimited (full lineage); a finite cap is accepted.
- `lineage_query(store, artifact_iri, preset, direction) -> rows` — runs one of
  the four preset queries scoped to the ego set.
- `raw_query(store, sparql) -> rows` — SELECT-only; reuses the store's existing
  `_validate_sparql_iri` / injection guards; rejects UPDATE/DELETE/INSERT.
- `ego_spdx_bundle(store, artifact_iri, direction) -> dict` — assembles an SPDX
  3.0.1 linked bundle for the ego set by reconstructing each member artifact's
  canonical BOM and linking them with `Relationship` elements, reusing the
  recursive feature's existing linked-bundle builder.

### B.2 Flask routes (`web/app.py`)

| Route | Returns |
|---|---|
| `GET /worldofboms/graph` | `full_graph` JSON |
| `GET /worldofboms/ego/<path:artifact>` | `ego_graph` JSON; `?direction=&depth=` |
| `GET /worldofboms/bom/<path:artifact>` | canonical reconstructed BOM (`reconstruct_bom`) for the side panel |
| `POST /worldofboms/query` | `{preset, artifact, direction}` or `{sparql}` → result rows |
| `GET /worldofboms/export` | `?scope=full\|ego&artifact=&direction=` → SPDX 3.0.1 linked bundle download |
| `GET /worldofboms/stats` | `store.stats()` + edge count, for the header |

All routes degrade gracefully when the store is unavailable (same
best-effort pattern as the existing `_try_resolve_cache`): they return an empty
graph with a `store_unavailable` flag rather than a 500, so the tab shows a
clean empty state.

### B.3 Preset queries

Each preset is a parameterized SPARQL SELECT in `graph_view.py`, run over the
ego node set for the chosen direction:

- **Licenses across the lineage** — `?artifact aibom:hasLicense ?license` (falls
  back to the canonical claim's `licenseName` field literal when no license
  edge exists), grouped by artifact.
- **All datasets in the lineage** — ego nodes typed `aibom:Dataset`.
- **All models in the lineage** — ego nodes typed `aibom:Model`.
- **Conflicts anywhere in the lineage** — claims in the ego set whose field
  annotations carry `conflictKind ∈ {interSourceConflict, intraSourceConflict}`.

## Part C — Frontend (`web/templates/index.html`)

### C.1 New tab

A `worldofBOMs` tab is added beside `Recursive`, following the existing
`switchTab` pattern. It reuses the Cytoscape + `cytoscape-dagre` libraries
already loaded, and the `.graph-wrap` / `.graph-canvas` / `.graph-controls` /
`.side-panel` CSS and DOM machinery from the recursive tab.

### C.2 Global view

On tab activation the tab fetches `/worldofboms/graph` and renders all
artifacts. Layout: a non-hierarchical layout (`cose` or `concentric`) since the
global graph is not a tree. Node colour by kind (Model / Dataset / Paper /
CodeRepo). A header strip shows counts from `/worldofboms/stats`
(N artifacts · N datasets · N models · N edges). Empty state: "No BOMs in the
graph yet — generate one to start your worldofBOMs."

### C.3 Ego view

Clicking a node:
- switches the layout to `dagre` (directional) and re-renders the ego subgraph
  from `/worldofboms/ego/<artifact>`;
- highlights the focus node (amber ring, same as the recursive tab's root) and
  its dependency-path edges; dims nodes/edges outside the ego set;
- opens the side panel.

A `View: Global | Ego` control and a `Direction: Upstream | Downstream | Both`
control sit in a top strip. Changing direction re-fetches the ego graph.

### C.4 Side panel

Two tabs inside the existing `.side-panel`:

- **BOM** — the node's reconstructed canonical BOM from `/worldofboms/bom/...`,
  rendered with the existing `renderBOM` / `renderFlatBOM` + raw-JSON toggle
  (identical to the recursive side panel).
- **Lineage & queries** — the Direction control, the four preset-query buttons
  (results render as a compact table), and a collapsible **Advanced: SPARQL**
  box that posts to `/worldofboms/query` and shows the result table.

### C.5 Download

A `Download ▾` control offers an SPDX 3.0.1 linked bundle for:
- the whole graph, or
- the current ego view × {upstream, downstream, both}.

It hits `GET /worldofboms/export`. Wired to the existing download-dropdown
component (PR #45) for visual consistency.

### C.6 Collision & UX handling

- **Identity:** one Artifact IRI = one node. The store's cross-identifier
  `resolve()` already collapses platform-handle variants, so duplicates cannot
  reach the renderer.
- **Layout overlap:** `cose`/`dagre` run with explicit node spacing and
  `avoidOverlap`; an auto-fit on render keeps the graph in view (reusing the
  recursive tab's fit logic).
- **Placeholder artifacts:** rendered as dashed, hollow nodes (consistent with
  the "greyed unmaterialized chips" treatment from PR #46). A placeholder shows
  a "generate a BOM for this" affordance; any `potentialDuplicateOf` edge on it
  (A.6) renders as a dashed link with a "possible duplicate of <artifact>" hint
  the user can confirm.
- **Label clashes:** when two *distinct* artifact IRIs share a `canonicalLabel`,
  each node label gets a platform badge (`hf:` / `gh:` / `arxiv:`) so they read
  as separate.
- **Search:** a search box filters/locates a node by label and centres it.
- **Growth:** the tab refreshes after a generation completes (hooking the
  existing post-generation event) so the user sees the graph grow.
- Floating zoom / fit / reset / fullscreen controls and the TB/LR/concentric
  layout switch are reused from the recursive tab.

## Testing

TDD-first — a failing test precedes each production change.

**Part A (connectivity):**
- Saving a model BOM with `trainedOnDatasets` creates a `trainedOn` edge to a
  (placeholder or real) dataset Artifact.
- `testedOn` and `dependsOn` edges created from their respective fields.
- Saving a model BOM that names an *already-stored* dataset connects to the
  existing node — by identifier (3a) and, separately, by name-label match (3b);
  node count does not increase for the dataset.
- A `name-only` placeholder is merged into a later real BOM for the same
  canonicalized name: incoming `trainedOn`/`testedOn`/`dependsOn` edges are
  transferred to the real artifact and the placeholder node is removed.
- A non-exact name does *not* auto-merge — the placeholder survives, and a
  confidence-triage match instead records a `potentialDuplicateOf` edge
  (a Jaro-Winkler-only match must never produce a merge).
- Edge writes are idempotent across a re-save.
- `graph rebuild` reconstructs edges from `results/*.json`.
- Vocab↔SCHEMA.md parity (existing `test_docs_schema_parity` covers the new
  predicates).

**Part B (backend):**
- `full_graph` / `ego_graph` node+edge shape.
- `ego_graph` direction semantics: `up` vs `down` vs `both` return the correct
  node sets on a known fixture graph.
- Each of the four preset queries on a fixture.
- `raw_query` rejects UPDATE/DELETE/INSERT.
- `ego_spdx_bundle` output passes SPDX 3.0.1 validation.
- Routes return a clean empty graph (not 500) when the store is unavailable.

**Part C (frontend):**
- Extends `tests/test_web_ui_features.py`: the `worldofBOMs` tab exists, the
  routes are reachable, the empty state renders.

## Documentation

- `docs/worldofboms/` — remove "Browser graph visualizer" from the v1 non-goals
  in `CONCEPT.md` and the original spec's non-goal list.
- New `docs/worldofboms/VISUALIZATION.md` — global vs ego view, the direction
  control, preset queries, raw SPARQL, SPDX export, and the placeholder /
  collision behaviour.
- `docs/worldofboms/SCHEMA.md` — `testedOn` + `dependsOn` rows (A.1).

## Files Touched

| File | Change |
|---|---|
| `src/aikaboom/store/vocab.py` | + `testedOn`, `dependsOn` |
| `src/aikaboom/store/mapper.py` (or new `store/edges.py`) | `add_relationship_edges` |
| `src/aikaboom/store/store.py` | call edge creation from `save_claim`; new `merge_artifacts(into, from_)` (extracted from `cli_graph.py`); name-label match helper |
| `src/aikaboom/store/cli_graph.py` | `cmd_graph_merge` delegates to `BomStore.merge_artifacts` |
| `src/aikaboom/store/graph_view.py` | **new** — full/ego/query/export read logic |
| `src/aikaboom/web/app.py` | + 6 `/worldofboms/*` routes |
| `src/aikaboom/web/templates/index.html` | + `worldofBOMs` tab, JS, side-panel tab |
| `docs/worldofboms/{CONCEPT,SCHEMA}.md`, new `VISUALIZATION.md` | docs |
| `tests/store/`, `tests/test_web_ui_features.py` | new tests per section above |

## Open Risk

Large global graphs (hundreds of artifacts) may render slowly in Cytoscape. v1
renders everything; if it proves slow in practice, a follow-up adds a node cap
with "show more". Not pre-optimized — flagged so it is a conscious choice.
