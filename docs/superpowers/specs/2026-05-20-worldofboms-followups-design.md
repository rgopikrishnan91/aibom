# worldofBOMs PR #48 — three followups

**Status:** design approved, plan pending
**Branch:** `worldofboms-graph` (all three followups commit here; PR #48 stays the single landing PR for this work)
**Date:** 2026-05-20

## Problem

PR #48 (worldofBOMs persistent graph + visualization) is in review. Three usability and verification gaps surfaced during late-stage testing:

1. **No bridge from the splash to past BOMs.** The first-load splash card shows the AIkaBoOM wordmark and 01/02/03 steps, but a returning user with prior BOMs sees no invitation to open one. Commit `fa532fe` left the History tab button visible from the empty state, but a tab button is not a navigation affordance — it reads as part of the chrome, not as "open your past work".
2. **No end-to-end evidence that the worldofBOMs graph reuses existing nodes when a new BOM is inserted.** The store layer has all the right mechanisms (deterministic artifact IRIs, cross-identifier `BomStore.resolve`, placeholder promotion, edge target dedup) and each is unit-tested. What is missing is a single test that drives the public web `/process` path twice and asserts that the second insert reuses the first artifact's IRI, producing exactly one artifact node with two claim rows.
3. **Raw SPARQL is the wrong surface.** The side-panel lineage pane offers four query presets plus an `<details>Advanced: SPARQL</details>` textarea. The textarea invites a level of graph literacy the target user does not have. What is wanted instead are more *action* buttons — operations against the focused node — sitting next to the presets.

## Non-goals

- Adding upstream traversal to the recursive BOM walker. Upstream is mentioned in the placeholder-generate chooser (per user direction) but the actual upstream/both walks are gated as "downstream-only today" until the pipeline grows that direction in a separate followup.
- Removing `graph_view.raw_query` from the Python store layer. It stays available for CLI and tests; only the web UI surface is removed.
- Schema or vocab changes to the RDF store.

## Design

### A. Splash → "Open a past BOM" affordance

**Location:** `src/aikaboom/web/templates/index.html`, the existing `#resultEmpty` splash card (lines ~3191–3260 on `worldofboms-graph`).

**Change:** One new line at the bottom of the splash, styled like `.splash-hint`, with a clickable anchor:

> *…or open one of your **N past BOMs** →*

- `N` reads from the already-loaded `History.rows` array (populated on `DOMContentLoaded` by `History.load()`).
- Hidden when `N === 0`.
- The link calls the existing `switchTabByName('history')`, which already toggles `is-history-open` on the empty pane so the History panel renders in place of the splash.

**JS:** A new `_refreshSplashHistoryHint()` that runs after `History.load()` and after every `History.upsert()` to keep the count fresh.

**Test:** Add to the existing Playwright UI smoke suite — a fresh page with one history row shows the hint with "1 past BOM", clicking it opens History. With zero rows, the hint is absent.

### B. End-to-end node-reuse verification

**Mechanisms already present (confirmed by reading `src/aikaboom/store/` on the branch):**

| Mechanism | Where | Effect |
|---|---|---|
| Deterministic artifact IRI | `iris.artifact_iri(pick_primary(canonicalize_set(idents)))` in `mapper.bom_to_rdf` | Same identifier set → same IRI → second BOM merges into the existing artifact subgraph |
| Cross-identifier resolve | `BomStore.resolve()` SPARQL `VALUES` across every supplied identifier | Artifact saved under `(hf=X, arxiv=Y)` still found by `arxiv=Y` alone |
| Placeholder promotion | `edges.promote_placeholders_for()` | Name-only stubs from a parent BOM repoint to the real artifact when a later BOM populates them |
| Relationship edge dedup | `edges.add_relationship_edges()` | `trainedOn` / `testedOn` / `dependsOn` targets resolved to existing IRIs before insert |

**New test file:** `tests/store/test_e2e_reuse_via_process.py`

The test exercises the public Flask `/process` route — the same path the browser hits — so it covers the full ingest pipeline, not just the store façade.

**Scenarios:**

1. **Identical-model reuse.** POST `/process` twice for the same model (using a corpus fixture so we don't hit the network). Assert:
   - `BomStore.stats()['artifacts'] == 1`
   - `BomStore.stats()['claims'] == 2`
   - Both claim IRIs are attached to the same artifact IRI via `hasVersion → hasClaim`.

2. **Cross-identifier reuse.** Save BOM 1 with `(hf=mistralai/Mistral-7B-v0.1, arxiv=2310.06825)`, then save BOM 2 with only `(arxiv=2310.06825)`. Assert one artifact, two claims.

3. **Edge target dedup.** POST BOM A for model M, then POST BOM B for model M' that lists M as `trained_on`. Assert the `trainedOn` edge points to M's existing artifact IRI (no duplicate artifact created for M).

**TDD-first.** Each scenario is written with a deliberately wrong initial expectation (`expected_artifacts = 2` for scenario 1) and watched to fail with the matching message, then flipped. Per standing rule: failing test first, watch the failure, then implement (or in this case, confirm production code already satisfies and the test is a regression guard).

**If a scenario fails on real code:** investigate and fix in the store layer before shipping. Most likely culprits, ranked by suspicion:
- Identifier set ordering not preserved across calls → different `pick_primary` outcomes → different IRIs (low risk: `canonicalize_set` sorts).
- The `/process` handler doesn't call `BomStore.save_claim` on every successful generation (audit it).
- Edge resolution doesn't run `BomStore.resolve` and falls back to creating a fresh artifact for the target.

### C. SPARQL textarea → action buttons

**Location:** `src/aikaboom/web/templates/index.html`, the `loadWorldLineage()` function (around line 7170) and its sibling event handlers.

**Removed:**
- The `<details>Advanced: SPARQL</details>` block, the `worldSparql` textarea, and the `worldSparqlRun` button.
- The `runWorldSparql()` JS function and its event-wire.
- The `sparql` branch of the `/worldofboms/query` POST handler in `src/aikaboom/web/app.py`. (`graph_view.raw_query` itself stays untouched — CLI and tests still need it.)

**Kept:**
- The four lineage presets (`licenses`, `datasets`, `models`, `conflicts`) and their backing `/worldofboms/query` POST.

**Added — four action buttons** in a new "Actions" sub-group below the existing four query presets:

| Button | Backend | Enabled when |
|---|---|---|
| Download SPDX bundle for this lineage | `GET /worldofboms/ego_spdx_bundle?artifact=<focusIri>&direction=<dir>` (route already exists) | A node is focused (`worldFocusIri` set) |
| Generate BOM for this placeholder (with scope chooser) | Pre-fills the main form, switches to the form pane, scrolls into view | Focused node has `claim_count === 0` *or* `kind === 'placeholder'` |
| Open this BOM in the main viewer | `GET /worldofboms/bom/<iri>` → rehydrate into the existing BOM/SPDX/CycloneDX viewers via `History`-style hydration | Focused node has `claim_count > 0` |
| Refresh / rebuild graph from history | `POST /worldofboms/rebuild` (new route, ~20 lines) — re-ingests every row in `bom-history/index.json`; idempotent because of section B's dedup | Always |

**Scope chooser for "Generate BOM for this placeholder":**
A small inline picker next to the button. User picks one of:
- **Just this node** — `recursive=off`
- **Downstream walk** — `recursive=on`, depth/breadth at form defaults
- **Upstream walk** — labelled `(downstream-only today — upstream walk lands in a followup)`, button disabled with a tooltip explaining why
- **Both** — same disabled state as Upstream

When enabled options are picked, the button pre-fills the form with `bom_type` (mapped from the node's `kind`: `model` → `ai`, `dataset` → `data`, others ambiguous → default to `ai` with a console warning), the subject ID (from `node.label`), and the recursive toggle, then switches to the form pane.

**New endpoint — `POST /worldofboms/rebuild`:**

```python
@app.route('/worldofboms/rebuild', methods=['POST'])
def worldofboms_rebuild():
    """Re-ingest every BOM in bom-history/index.json into the worldofBOMs
    graph. Idempotent — relies on store dedup (artifact IRI from canonical
    identifier set + BomStore.resolve cross-identifier lookup). Returns
    {processed, artifacts, claims} so the UI can show a toast."""
```

This is the only new server-side code in section C. It iterates `_history_load()`, opens each artifact JSON, reconstructs identifiers via the same path used by `/process`, and calls `BomStore.save_claim`. A counter tracks how many rows it walked.

**Tests for section C:**
- Backend: a new test in `tests/store/test_graph_view.py` (or sibling) hits `POST /worldofboms/rebuild` against a temp `bom-history/` with two fixture BOMs, then asserts `stats() == {artifacts: 2, claims: 2}` (or 1+2 if the fixtures share an identifier).
- Frontend (Playwright smoke): the lineage pane has four `[data-action]` buttons, no `#worldSparql` textarea, and clicking "Open this BOM" loads it into the main viewer (assert by checking `#flatViewerComplete` got populated content).

## File-level impact

| File | Change | Lines (est.) |
|---|---|---|
| `src/aikaboom/web/templates/index.html` | Section A splash hint (+ JS hook), Section C lineage pane rewrite | +120 / -55 |
| `src/aikaboom/web/app.py` | Drop sparql branch from `/worldofboms/query`; add `/worldofboms/rebuild` | +30 / -8 |
| `tests/store/test_e2e_reuse_via_process.py` | New file, section B | +180 |
| `tests/store/test_graph_view.py` | Add `/rebuild` test | +35 |
| `tests/ui/test_splash_history_hint.spec.ts` (or wherever Playwright tests live on this branch) | New, section A | +40 |
| `tests/ui/test_world_actions.spec.ts` | New, section C | +60 |

No store-layer code changes are expected. If section B uncovers a bug, the file impact grows.

## Order of work

1. **Section B test first.** Cheapest signal — passes mean we keep moving with confidence; fails surface the real bug to fix before shipping sections A/C.
2. **Section A.** Self-contained HTML/CSS/JS slice. One commit.
3. **Section C.** Template edit + JS rewrite of `loadWorldLineage` + one new Flask route + tests. One commit.

Three commits on `worldofboms-graph`, in this order, keeping PR #48's diff readable.

## Open risks

- **Form pre-fill semantics.** If the worldofBOMs side-panel knows only a node's label (and not its full identifier set), the form pre-fill in section C's Generate-BOM action may end up with only `repo_id`/`hf_repo_id` populated. That's the same input shape a fresh user types, so the existing `/process` pipeline handles it.
- **Bom-history rebuild contention.** `POST /worldofboms/rebuild` may run while a `/process` request is mid-save. Since both go through `BomStore.save_claim` and rdflib/oxigraph backends are serial-write, the worst case is a brief stall. No transactional rollback needed.
- **`kind === 'placeholder'` detection.** The frontend reads `kind` from `worldFullGraph.nodes`. Verify that placeholder nodes have a stable `kind` value before relying on it in the gating logic; fall back to `claim_count === 0` if not.
