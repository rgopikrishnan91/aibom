# Recursive BOM — depth / breadth / cap + visible tree

Date: 2026-05-15
Status: approved

## Problem

The recursive BOM walker conflates three concepts into two controls
(`max_depth`, `safety_cap`). There is no per-node fan-out limit, so a
large `testedOn` dataset list consumes the global cap and starves the
`modelLineage` (`dependsOn`) edge — the only recursable one. As a
result depth >1 never produces deeper nodes, and in the web UI the
Stage 2 live tree shows level 1 only. Discovered-but-not-generated
("greyed") nodes are not interactive.

## Three distinct controls

| Control | Meaning | Default |
|---|---|---|
| `max_depth` (depth) | tree levels to walk | 1 |
| `breadth` (NEW) | max children expanded per node | 10 |
| `safety_cap` | absolute ceiling on total generated nodes — runaway guard | 200 |

`discover_recursive_targets` returns targets lineage-first (`dependsOn`
before dataset edges). The walker expands only the first `breadth`
non-duplicate targets per parent, so `dependsOn` is always within
budget. Targets beyond `breadth` are recorded as `breadth_capped`.
`safety_cap` wraps the whole walk as the outer guard.

## Identified-but-not-generated nodes

A target that is discovered but not turned into a full node —
`breadth-cap`, `safety-cap`, or `enrichment-failed` / `unresolved` — is
recorded with a `reason` and emitted as a `recursive.child.skipped`
event so the UI greys it. Enrichment failure previously fell back to
seed metadata silently and still counted as generated; it now becomes a
greyed node with the reason and the branch stops there (siblings
continue).

## Depth >1

A depth-2 node appears only when its depth-1 parent enriched
successfully. A depth-1 node that failed to enrich is greyed with its
reason, so it is always visible *why* a branch stopped.

## UI (web — `templates/index.html`, Stage 2)

- The Stage 2 live tree consumes `recursive.*` SSE events; depth-2+
  nesting is verified/fixed so grandchildren render under their parent
  at any depth.
- `breadth-cap` / `enrichment-failed` reasons are added to greyed chips.
- Greyed chips get a `contextmenu` handler with two actions:
  - **Generate this BOM** — `POST /recursive-node` runs the enricher for
    that single target and returns the BOM; the chip turns green and the
    result is slotted in.
  - **Why was this skipped?** — popover with the reason and basic info.

## New endpoint

`POST /recursive-node` — body `{target, bom_type, parent, relationship_type}`.
Runs `build_enrich_fn(...)` for the one target, builds the node
(`spdx_data` / `cyclonedx_data`), returns it. Used by the right-click
"Generate this BOM" action.

## Wiring

- CLI: `--recursive-breadth` (default 10).
- Web form: a "breadth" input alongside depth and safety-cap.

## Out of scope

The Recursive-tab Cytoscape graph stays a final snapshot; the live
experience is the Stage 2 tree.

## Testing

TDD throughout: failing test first for breadth capping, enrich-fail
greying, depth-2 walk, and the new endpoint.
