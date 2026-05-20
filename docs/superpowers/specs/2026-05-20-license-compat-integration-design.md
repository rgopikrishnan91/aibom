# License-compatibility analysis as a plugin — design

**Status:** Approved for planning. Implementation plan to be authored next.
**Date:** 2026-05-20
**Author:** Gopi (with Claude)
**Target branch:** new branch off `main`, lands after `worldofboms-graph` merges
**Related:** `docs/worldofboms/CONCEPT.md`, `feedback_one_pr_per_feature.md`, `project_plugin_architecture.md`

## Problem

We have a working license-compatibility analyzer at `/mnt/d/LicenseRec/Processing/LicenseRec.py`. It loads a matrix of pairwise license compatibility verdicts (Yes / No / Same), resolves raw license strings to canonical names via aliases, checks whether each downstream artifact's license is compatible with all of its upstreams, and recommends alternative licenses for downstreams whose current choice violates compatibility. The current tool is a batch script over static HuggingFace + GitHub dumps with two hardcoded lineage shapes: Dataset → Model and Model → Repo.

We want this capability inside aibom, where the worldofBOMs graph already provides the substrate: an RDF knowledge graph of every generated BOM, deduped by artifact identity, with lineage predicates (`trainedOn`, `testedOn`, `dependsOn`, `hostedAt`) and `hasLicense` triples already in place. The lineage we want to analyze is exactly the lineage the graph already records.

## Goals

- Compatibility analysis runs over any lineage chain the graph knows about — model→model, dataset→model, model→repo, any directed lineage edge that makes semantic sense.
- Produces both **compliance verdicts** (per edge) and **relicensing recommendations** (for violating downstreams), mirroring LicenseRec.py's analytical scope.
- Surfaces structural insights derived from the per-edge verdicts: **maximal compatible subchains** and **breaking nodes** (upstream artifacts that cause the most violations).
- Available on two scopes: per-artifact (one BOM and its lineage) and graph-wide audit.
- Surfaces across four UIs: a new "License compatibility" tab in the BOM viewer, edge tinting on the WorldOfBOM graph view, SPDX Annotation Elements in exports, and category entries in the existing Conflicts tab.
- Ships as **plugin #1** under a new in-tree plugin architecture that future analytics (security advisories, attribution checks) will also use, without touching core call sites.

## Non-goals

- A registry server, multi-user identity, or remote SPARQL endpoint. The graph stays local-first per WorldOfBOM.
- A general "lineage policy" framework abstraction. We have one policy; the plugin Protocol is the framework. Generalize on the second concrete user.
- Embedding license-compat into `spdx_validator.py`. Compat is policy, not SPDX structural validity. Separate plugin, separate emitter.
- Retrofitting existing conflict-annotations into a plugin in this PR. Optional follow-up.

## Architecture

### Plugin substrate

A hybrid in-tree plugin system with a stable `Plugin` Protocol now and `importlib.metadata.entry_points` discovery later (one-line change).

```
src/aikaboom/plugins/
├── __init__.py            # register(plugin), all_plugins(), get(name)
└── base.py                # Plugin Protocol + Scope, Finding, GraphOverlay dataclasses
```

**Discovery flow on import of `aikaboom.plugins`:**
1. `plugins/__init__.py` eagerly imports each in-tree plugin subpackage.
2. Each plugin's `__init__.py` calls `register(InstanceOfPlugin())`.
3. Core code calls `all_plugins()` and iterates.
4. Later: replace eager-import with an `entry_points` discovery call.

**Plugin Protocol hook surface:**

```python
class Plugin(Protocol):
    name: str
    def enabled(self) -> bool: ...
    def analyze(self, store: BomStore, scope: Scope) -> Findings: ...
    def register_cli(self, parent_subparsers) -> None: ...
    def web_blueprint(self) -> Blueprint | None: ...
    def bom_viewer_tab(self) -> TabSpec | None: ...
    def spdx_annotations(self, claim_iri, findings) -> list[dict]: ...
    def graph_overlay(self, findings) -> GraphOverlay: ...
    def conflict_findings(self, findings) -> list[ConflictRecord]: ...
```

A plugin returns `None` or an empty list for any hook it doesn't implement; core treats absence as silence. Core call sites loop plugins and compose results — adding plugin #2 (security advisories per `project_plugin_architecture.md`) is a new directory under `plugins/`, no changes to core.

### License-compat plugin layout

```
src/aikaboom/plugins/license_compat/
├── __init__.py            # @register decorator binds the plugin
├── plugin.py              # LicenseCompatPlugin(Plugin) — wires the hooks
├── matrix.py              # LicenseMatrix loader, alias lookup, upstream→downstream index
├── engine.py              # Pure: check_compat, recommend, find_compatible_subchains, find_breaking_nodes
├── walker.py              # SPARQL enumeration of lineage edges + trust-aware license resolution
├── cli.py                 # license-check, license-audit subparsers
├── web.py                 # Flask Blueprint
├── spdx.py                # Annotation Element emitter (SPDX + CycloneDX parity)
├── overlay.py             # GraphOverlay payload for graph-view edge tinting
├── templates/license_compat/tab.html
└── data/
    ├── matrix.json
    ├── allowed_licenses.json
    └── missing.json
```

Bundled data loads via `importlib.resources` so the package is pip-installable. A `--matrix` CLI flag and env var override the bundled paths for custom matrices.

### Engine API (pure functions)

All inputs are values; no I/O inside engine functions. Tests are property-style: feed inputs, assert verdicts.

```python
@dataclass(frozen=True)
class LicenseMatrix:
    name_alias_lookup: dict[str, str]
    details: dict[str, dict]
    upstream_compat_index: dict[str, set[str]]
    allowed_licenses: set[str]
    missing_licenses: set[str]

def load_matrix(matrix_path=None, allowed_path=None, missing_path=None) -> LicenseMatrix

def resolve_license(raw: str, matrix: LicenseMatrix) -> ResolvedLicense

@dataclass(frozen=True)
class CompatVerdict:
    downstream: str
    upstreams: frozenset[str]
    status: Literal["compatible", "violation", "unknown_upstream", "unknown_downstream", "missing_data"]
    incompatible_with: frozenset[str]

def check_compat(downstream: str, upstreams: frozenset[str], matrix: LicenseMatrix) -> CompatVerdict

@dataclass(frozen=True)
class Recommendation:
    by_category: dict[str, list[str]]
    is_solvable: bool

def recommend(upstreams: frozenset[str], matrix: LicenseMatrix, frequencies: Counter[str], top_k_per_category: int = 5) -> Recommendation

@dataclass(frozen=True)
class CompatSubchain:
    artifacts: frozenset[URIRef]
    edges: frozenset[tuple[URIRef, URIRef, URIRef]]
    size: int
    root: URIRef

def find_compatible_subchains(findings: Findings) -> list[CompatSubchain]
# Connected components of the subgraph induced by compatible edges only.
# Sorted by size desc, then by root label. Isolated nodes appear as size-1 components.

@dataclass(frozen=True)
class BreakingNode:
    artifact_iri: URIRef
    label: str
    license: str
    blamed_in: int
    affected_downstream: frozenset[URIRef]
    fix_recommendations: Recommendation

def find_breaking_nodes(findings: Findings) -> list[BreakingNode]
# Upstream artifacts in CompatVerdict.incompatible_with for >=1 violation.
# fix_recommendations computed against the union of all downstream contexts that blame this node.
```

### Graph walker

Three responsibilities: enumerate lineage edges, resolve each artifact's license via the trust rule, compute frequencies for the recommender.

```python
LINEAGE_PREDICATES = (vocab.trainedOn, vocab.testedOn, vocab.dependsOn, vocab.hostedAt)

@dataclass(frozen=True)
class LineageEdge:
    downstream_iri: URIRef
    downstream_label: str
    upstream_iri: URIRef
    upstream_label: str
    predicate: URIRef

@dataclass(frozen=True)
class ResolvedArtifact:
    iri: URIRef
    label: str
    licenses: frozenset[str]
    source_claim_iri: URIRef | None
    has_unknown: bool
    has_missing: bool

def enumerate_edges(store: BomStore, scope: Scope) -> Iterator[LineageEdge]
def resolve_artifact_license(store: BomStore, artifact_iri: URIRef, matrix: LicenseMatrix) -> ResolvedArtifact
def compute_license_frequencies(store: BomStore, matrix: LicenseMatrix) -> Counter[str]
```

**Trust-aware license resolution:**

For each artifact, the walker picks the BOMClaim with the highest `trustScore`, tie-breaking to `canonicalClaim`, then to most recent `createdAt`. SPARQL:

```sparql
SELECT ?claim ?lic ?trust WHERE {
  ?version aibom:hasClaim ?claim .
  ?claim aibom:trustScore ?trust ;
         aibom:hasLicense ?lic .
  { ?artifact aibom:hasVersion ?version } UNION { ?artifact aibom:canonicalClaim ?claim }
  FILTER (?artifact = <iri>)
}
ORDER BY DESC(?trust) DESC(?createdAt) LIMIT 1
```

If no claim-attached license resolves, fall back to artifact-level `hasLicense` triples. If nothing resolves, the artifact's `licenses` is the empty set and a downstream verdict marks it `unknown_downstream`.

**Scope handling:**
- `Scope.single(iri)` — BFS along `LINEAGE_PREDICATES` from the artifact, cycle-safe (visited set), depth-bounded (default 5).
- `Scope.graph_wide()` — single SPARQL stream of all triples with a predicate in `LINEAGE_PREDICATES`.

**Frequencies:**
For `recommend()`, ranked by usage. Computed once per audit from a SPARQL over all `?artifact aibom:hasLicense ?lic` triples in the graph, normalized through the matrix's alias lookup, returned as `Counter[primary_name]`. This replaces LicenseRec.py's static HF-dump counts — the graph becomes the corpus.

### CLI

Two subcommands, both owned by the plugin (only the plugin file touches argparse for this feature).

```python
def register_cli(parent_subparsers):
    p_check = parent_subparsers.add_parser("license-check", ...)
    p_check.add_argument("artifact")
    p_check.add_argument("--depth", type=int, default=5)
    p_check.add_argument("--format", choices=["text", "json", "jsonl"], default="text")
    p_check.add_argument("--matrix", type=Path)
    p_check.add_argument("--violations-only", action="store_true")

    p_audit = parent_subparsers.add_parser("license-audit", ...)
    p_audit.add_argument("--format", choices=["text", "json", "jsonl"], default="text")
    p_audit.add_argument("--matrix", type=Path)
    p_audit.add_argument("--out", type=Path)
```

**`artifact` resolution:** IRI (verbatim) → platform-id (`hf:org/name`, `gh:org/repo`, `arxiv:NNNN`) through existing `BomStore.resolve()` → fuzzy label via `canonicalLabel` matching. Ambiguous matches list candidates and exit 3.

**Exit codes:** 0 no violations, 2 violations found, 3 unresolved artifact / no data, 1 unexpected error.

**Text output includes Compatible-subchains and Breaking-nodes sections in addition to per-edge verdicts.** Subchains shown as horizontal pill rows with size badges; breaking nodes ranked by blame count with contextual fix recommendations. JSON/JSONL output adds top-level `compatible_subchains` and `breaking_nodes` keys.

When a breaking node's license is *more permissive* than the downstream's, the text output surfaces a "relicense the downstream instead" hint, since touching a permissive upstream is rarely the right move.

### UI surfaces

All four surfaces consume the same `Findings` object from one `analyze()` call. Web request handler computes once, feeds all four emitters.

**SPDX export — Annotation Elements:** Two annotation kinds, both `annotationType="review"`, structured JSON in `comment`:
- Per violation, attached to downstream artifact's Element.
- Per breaking node, attached to the breaking node's Element with blame count and downstream consumers.

JSON-LD output gets the same triples via the SPDX 3 vocabulary. CycloneDX exporter gets parity emission through the existing `cyclonedx_exporter.py` plugin hook.

**Web — "License compatibility" tab:** Three stacked sections in the BOM viewer:
1. Lineage view (collapsible tree, license chips colored by status).
2. Compatible subchains (collapsed by default, horizontal pill rows).
3. Breaking nodes (sortable table, side-panel with full recommendation list).

A new `bom_viewer_tab()` plugin hook returns `{label, url_template, sort_order}`; the index template iterates the list to render the tab strip. Templates live in `src/aikaboom/plugins/license_compat/templates/license_compat/`.

**Graph view — edge tinting overlay:**
- Edges: green compatible, red violation, grey unknown.
- Nodes: red ring + corner badge on breaking nodes; faint green halo on largest compatible subchain.
- Tooltip on hover; click opens side-panel with verdict + recommendation.

Front-end: one new toggle in the existing graph view's legend, route `/license-compat/<artifact_id>/overlay.json` returns the GraphOverlay-as-JSON. Multiple plugins' overlays compose by namespace.

**Conflicts tab — new conflict category:** Plugin hook `conflict_findings()` emits `ConflictRecord`s with `category="license-compat"`. Existing Conflicts-tab grouping renders them with no template changes.

### Cross-surface guarantees

- Caching: `analyze(store, Scope.single(iri))` results cached per `(artifact_version_iri, matrix_version)` in the BomStore, TTL ~1h or invalidated when a new BOMClaim lands on any artifact in the lineage.
- Graceful disable: `enabled() -> False` removes the tab, overlay toggle, annotation emission, and conflict entries. Core paths stay live.
- Deletion: removing `src/aikaboom/plugins/license_compat/` removes the feature with no other code changes.

## Data model — bundled files

```
src/aikaboom/plugins/license_compat/data/
├── matrix.json             # vendored from /mnt/d/LicenseRec/Data/Matrixes/matrix.json
├── allowed_licenses.json   # vendored
└── missing.json            # vendored
```

Loaded via `importlib.resources.files("aikaboom.plugins.license_compat.data").joinpath("matrix.json")`. Versioned with the codebase. `--matrix` CLI flag and `AIKABOOM_LICENSE_MATRIX` env var override.

The matrix schema (per LicenseRec.py):
```json
{
  "timestamp": "2025-05-06T04:36:58+0000",
  "licenses": [
    {
      "name": "apache-2.0",
      "aliases": ["apache 2", "apache 2.0", "..."],
      "category": "PERMISSIVE",
      "compatibilities": [
        {"name": "apache-2.0", "compatibility": "Yes"},
        {"name": "gpl-3.0", "compatibility": "No"}
      ]
    }
  ]
}
```

Compatibility values: `"Yes"` and `"Same"` count as compatible; everything else (`"No"`, `""`, missing) is treated as incompatible. This matches LicenseRec.py's `upstream_compat_index` construction.

## Testing strategy

Layered to match the architecture:

| Tier | Scope | Speed | Inputs |
|------|-------|-------|--------|
| 1 | Pure engine (matrix, check_compat, recommend, subchains, breaking-nodes) | <500ms total | tiny_matrix fixture only |
| 2 | Graph walker | seconds | rdflib in-memory store + small TTL fixtures |
| 3 | Plugin contract / registry | ms | mocks |
| 4 | Surface emitters (SPDX, web, overlay, conflicts) | seconds | canned Findings fixture |
| 5 | CLI | seconds | subprocess against tmpdir BomStore |
| 6 | End-to-end smoke | ~10s | one ~5-artifact fixture |

**Coverage target:** engine modules 100%, walker 90%+, surfaces 80%+. Enforced via `coverage report --include=src/aikaboom/plugins/license_compat/*` in CI.

**TDD ordering (per `feedback_tdd_first.md`):** every change starts with a failing test that displays the expected message before the implementation is written. Implementation plan to flag explicitly per item; ask before skipping any.

**Backend parity:** walker tests run against `RDFLibBackend` in CI. One additional skipped-if-unavailable test runs the same suite against `OxigraphBackend` to catch drift.

## Risk and trade-offs

**Bundled matrix versioning.** The matrix is a vendored snapshot at point in time. As licenses evolve (new variants, new compatibility judgments), the bundle drifts from reality. Mitigation: bundle includes a `timestamp` field surfaced in the CLI output and tab UI ("matrix dated 2025-05-06"); a future "refresh-matrix" subcommand can pull updates without code changes. Out of scope for this PR.

**Trust-tie-break ordering.** Picking the highest-trust claim's license loses information about disagreement between claims. The Conflicts tab already surfaces this via field-level conflict annotations, so users who care can see the disagreement there. The license-compat tab notes the source claim in the UI so the choice is traceable.

**Walker depth limit.** Default depth 5 may miss long chains (e.g., a paper that cites a model that was trained on a dataset derived from another dataset). Configurable via `--depth`. Graph-wide audit doesn't use depth — it sees every edge.

**Pure-permissive upstreams flagged as breaking.** A breaking-node is *any* upstream whose license blocks a downstream. If the downstream is GPL and the upstream is MIT, MIT is technically the "blame" target but it's not the one we want to relicense. The "relicense the downstream instead" hint addresses this in CLI/UI output, but the underlying `BreakingNode` dataclass still records the structural cause. This is intentional — emitters can choose how to present it; the analytical primitive shouldn't pre-judge.

**Recommendation frequencies bootstrap problem.** Until the graph has a meaningful population, frequency ranking will be noisy or empty. Fallback: when `Counter` is empty for a candidate, sort lexicographically. Document in CLI help.

**Plugin import-time cost.** Eager-import on every CLI invocation pays a small startup penalty per plugin. With one plugin this is invisible; with ten it would matter. Mitigation: lazy import inside `register()` if the per-plugin import is heavy. Not needed yet.

## Open questions

None at design time. All clarifying answers captured above.

## Sequencing

1. Land worldofboms-graph (independent of this work).
2. Branch off `main` once worldofboms-graph is merged.
3. Implement in this order, each its own commit, all TDD-first:
   1. Plugin substrate (`plugins/__init__.py`, `plugins/base.py`, registry tests).
   2. Engine + matrix loader (`matrix.py`, `engine.py`, all Tier 1 tests).
   3. Graph walker (`walker.py`, Tier 2 tests).
   4. Plugin wiring (`plugin.py`, `__init__.py`, contract tests).
   5. CLI (`cli.py`, Tier 5 tests, golden snapshots).
   6. SPDX emitter (`spdx.py`, Tier 4 SPDX tests).
   7. Conflicts integration (`plugin.conflict_findings`, Tier 4 conflicts test).
   8. Web tab (`web.py`, templates, Tier 4 web tests).
   9. Graph overlay (`overlay.py`, frontend toggle, Tier 4 overlay tests).
   10. End-to-end smoke (Tier 6).
4. Bundle matrix data in step 2 (engine needs it).
5. Open PR.

Each step is independently mergeable in principle; in practice they ship as one PR for review coherence.

## Out of scope (deferred)

- Plugin entry_points discovery (defer until first third-party plugin).
- Retrofitting existing conflict-annotations as a plugin.
- "Refresh matrix" subcommand to pull updated compatibility data.
- Multi-matrix support (running the same lineage against `matrix.json` and `final_matrix.json` for diff).
- License-compat results as RDF triples in the graph itself (currently they're computed on demand, not persisted).
- Per-jurisdiction policy variants (US vs EU interpretations of attribution requirements).
- Attribution-requirements check, security-advisory check — those are plugin #2+.
