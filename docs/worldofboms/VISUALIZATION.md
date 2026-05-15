# worldofBOMs — Visualization Reference

The worldofBOMs visualization is a browser-based graph explorer built into the
AIkaBoOM web UI. It shows every artifact the graph knows about and the
relationships between them, and lets you drill into any one artifact's lineage,
query it, and download a linked SPDX bundle.

The visualization is documented here; the underlying graph store and its RDF
schema are covered in `CONCEPT.md` and `SCHEMA.md`.

## The worldofBOMs tab

A `worldofBOMs` tab sits alongside the `Recursive` tab in the web UI. It
reuses the same Cytoscape graph canvas, side panel, and layout controls as the
recursive-BOM view, so the interaction model is familiar.

The tab loads the global view automatically on first activation and refreshes
after each local generation finishes, so the graph grows in place as you
generate more BOMs.

## Global view

The default view renders every artifact in the store as a node and every
relationship edge between them.

**Node colour by kind:**

| Kind | Colour |
|---|---|
| Model | Blue / indigo (`#e6ecff` background, `#3451b2` border) |
| Dataset | Amber / peach (`#fff1e5` background, `#f0b079` border) |
| Paper | Green |
| CodeRepo | Purple |

Nodes whose artifact is a *placeholder* — an unresolvable reference minted
automatically when a model BOM names a dataset or model the graph has not seen
yet — render as dashed, hollow circles to distinguish them from artifacts the
system has a real BOM for.

A stats header at the top of the tab shows the counts from
`GET /worldofboms/stats`: **N artifacts · N edges · N claims**.

Empty state: "No BOMs in the graph yet — generate one to start your
worldofBOMs."

## Ego view

Clicking any node switches the canvas to that node's *ego subgraph*:
a breadth-first subgraph of the node and everything connected to it
(direction-dependent; see below). The rest of the graph is dimmed.

The layout switches from the global `cose` arrangement to a
directional `dagre` layout. The focus node gains an amber ring, the same
treatment as the root node in the recursive tab.

A `View: Global | Ego` toggle in the top strip lets you return to the full
graph. Changing the Direction control (see below) re-fetches the ego subgraph
without navigating away.

## The Direction control

A `Direction: Upstream | Downstream | Both` control in the top toolbar scopes
every ego operation — the ego subgraph, preset queries, and the download.
The Lineage & queries pane reflects the active value via a "Scope:
\<direction\>" hint but does not host the control itself.

**Edge direction convention.** Relationship edges point from the *dependent*
to the *dependency*:

- `trainedOn` — the model points to the dataset it was trained on.
- `testedOn` — the model points to the evaluation dataset.
- `dependsOn` — the dependent artifact points to what it depends on.

**Upstream** follows edges *forward* out of the focus node — its dependencies
(the datasets, models, and repos it directly or transitively relies on).

**Downstream** follows edges *backward* into the focus node — its dependents
(things that list this artifact in their own relationship fields).

**Both** is the union of upstream and downstream.

## The side panel

Clicking a node opens a side panel with two panes:

- **BOM** — the node's reconstructed canonical BOM, fetched from
  `GET /worldofboms/bom/<artifact>`. Rendered via `renderBOM()` into a
  `.flat-bom` div — the same renderer the recursive tab uses, but without
  a raw-JSON toggle.

- **Lineage & queries** — the four preset-query buttons, a "Scope:
  \<direction\>" hint reflecting the current Direction control value, and an
  Advanced SPARQL box (see below).

## Preset lineage queries

Four preset queries run over the ego node set for the active direction:

| Preset | What it returns |
|---|---|
| **Licenses across the lineage** | One row per (artifact, license) pair found in the ego set. Each artifact's `licenseName` field value is read directly from its canonical claim. |
| **All datasets in the lineage** | Ego nodes typed `aibom:Dataset`. |
| **All models in the lineage** | Ego nodes typed `aibom:Model`. |
| **Conflicts anywhere in the lineage** | Claims in the ego set whose field annotations carry `conflictKind ∈ {interSourceConflict, intraSourceConflict}`. |

Results render as a compact table beneath the preset buttons.

## Advanced: raw SPARQL

A collapsible **Advanced: SPARQL** box in the Lineage & queries pane lets
power users run arbitrary read-only queries against the live store. It posts
to `POST /worldofboms/query`.

Only `SELECT` and `ASK` queries are accepted. Any query containing a mutation
keyword (`INSERT`, `DELETE`, `LOAD`, `CLEAR`, `DROP`, `CREATE`, `ADD`, `MOVE`,
`COPY`) is rejected with an error before it reaches the store. This is an
in-app convenience, not a public SPARQL HTTP endpoint.

For a library of pre-written queries, see `QUERIES.md`.

## Download

A `Download ▾` control exports an SPDX 3.0.1 linked bundle via
`GET /worldofboms/export`. Two scopes are offered:

- **Whole graph** (`?scope=full`) — every artifact in the store.
- **Ego view** (`?scope=ego&artifact=<iri>&direction=<up|down|both>`) —
  the current ego subgraph for the chosen direction.

Each member artifact's canonical BOM is reconstructed (or a minimal seed BOM
is used for placeholders) and linked with `Relationship` elements by
`build_linked_spdx_bundle`, the same builder used by the recursive-BOM
feature. The download filename is `worldofboms-graph.spdx.json` for the full
graph and `worldofboms-ego.spdx.json` for an ego view.

## How edges are created

Artifact-to-artifact relationship edges (`trainedOn`, `testedOn`, `dependsOn`)
are persisted to the graph when `BomStore.save_claim` is called — which happens
at every BOM generation, not only in recursive mode.

For each relationship field present in the BOM (`trainedOnDatasets`,
`testedOnDatasets`, `modelLineage`, `sourceInfo`), the store resolves the
target to an existing artifact node (by identifier match, then by
canonicalized-name match) or mints a placeholder. The graph therefore grows and
self-connects with every generation: a model BOM saved today automatically
links to a dataset node that was added yesterday, and vice versa.

When a placeholder is later replaced by a real BOM for the same canonicalized
name, the placeholder is merged into the real node and all incoming edges are
transferred. See sections A.3–A.6 of
`docs/superpowers/specs/2026-05-15-worldofboms-visualization-design.md` for
the full edge-creation, name-matching, and placeholder-promotion logic.

**Placeholder display.** Placeholder nodes are rendered as dashed, hollow
circles. If the confidence-triage layer found a probable-but-not-exact match
against an existing artifact, a dashed `potentialDuplicateOf` edge is shown
with a "possible duplicate of <artifact>" hint. The user confirms a genuine
duplicate with `aikaboom graph merge`; the visualization never auto-merges on a
fuzzy score.

**`graph rebuild`** replays every `results/*.json` through `save_claim`, so
edges are reconstructed retroactively if the store is ever reset.

## Developer reference

| Route | Returns |
|---|---|
| `GET /worldofboms/graph` | `full_graph` JSON — all nodes and edges |
| `GET /worldofboms/stats` | Artifact / edge / claim counts |
| `GET /worldofboms/ego/<path:artifact>` | Ego subgraph; `?direction=up\|down\|both&depth=N` |
| `GET /worldofboms/bom/<path:artifact>` | Reconstructed canonical BOM for the side panel |
| `POST /worldofboms/query` | `{preset, artifact, direction}` or `{sparql}` → result rows |
| `GET /worldofboms/export` | `?scope=full\|ego&artifact=&direction=` → SPDX 3.0.1 download |

All routes return a clean empty response (not a 500) when the graph store is
unavailable, so the tab shows an empty state rather than an error page.

The read-side logic lives in `src/aikaboom/store/graph_view.py`; the Flask
routes are in `src/aikaboom/web/app.py`.
