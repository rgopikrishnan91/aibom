# AVID + SPDX 3.0 Security Profile Plugin — Design

Status: Draft for review
Author: gopi (with Claude)
Date: 2026-05-20

## 1. Summary

Add a new AIBOM plugin, `avid_security`, that:

1. Walks the components of a generated BOM (principal + base models + training datasets) and looks each up against a locally-cached snapshot of the [AVID-DB](https://github.com/avidml/avid-db) AI vulnerability database.
2. Emits SPDX 3.0.1 Security Profile elements (`Vulnerability` + Core `Relationship(hasAssociatedVulnerability)` + `Vex*VulnAssessmentRelationship`) into the existing SpdxDocument for each matched (AVID report × component) pair.
3. Surfaces matches in a new Security web tab, on the existing graph view (node tinting), and in the Conflicts tab (high-confidence findings only), plus a `aikaboom security ...` CLI.
4. Ships the feature as a self-contained plugin under `src/aikaboom/plugins/avid_security/`, conforming to the hybrid in-tree plugin architecture (see `project_plugin_architecture` memory and the parallel `license_compat` plugin work).

The MIT AI Risk Repository (AIR) is **out of scope for v1** — it has no per-model records; deferring to v1.1 keeps matching deterministic and defensible.

The user-stated primary value is **practical risk surfacing** (recall-leaning, with VEX statuses for triage).

## 2. Goals & non-goals

### Goals

- Identify AVID-known issues against models and datasets in the BOM **without false-precision noise** (LLM-driven matching is rejected for v1; matcher is deterministic).
- Produce a single SPDX 3.0.1 JSON-LD document that passes SHACL validation including the Security profile shapes.
- Preserve BOM reproducibility: every Vulnerability element records the AVID snapshot SHA used at generation time.
- Be a true plugin — `src/aikaboom/plugins/avid_security/` is a drop-in directory; deleting it removes the feature with no other code changes.
- Survive the next batched main-golden-set run with no special handling (per `feedback_golden_set_runs` — feature ships on its own AVID golden set; main rerun consumes it passively).

### Non-goals (v1)

- No MIT AIR integration (deferred to v1.1).
- No CVSS / EPSS / SSVC scoring synthesis from SEP codes.
- No LLM-driven semantic matching or applicability confirmation.
- No suppression / "mark as not applicable" UI (would require a user/auth model).
- No live-network test coverage; CI runs offline against vendored fixtures.

## 3. Architecture

### 3.1 Module layout

```
src/aikaboom/plugins/avid_security/
├── __init__.py        # @register(AvidSecurityPlugin()) binds the plugin
├── plugin.py          # AvidSecurityPlugin(Plugin) — implements protocol hooks
├── snapshot.py        # AvidSnapshot — clone/refresh ~/.cache/aikaboom/avid-db/,
│                      #   SQLite index keyed by (bare_name, family_prefix, developer,
│                      #   sep_view, risk_domain, lifecycle, artifact_kind),
│                      #   10-day TTL, snapshot SHA persisted in marker file
├── matcher.py         # ComponentMatcher — tier 1 exact / tier 2 base-model lineage /
│                      #   tier 3 same-family + developer; emits Match records with
│                      #   confidence + evidence dicts
├── engine.py          # Pure: tier_to_vex_status, build_vulnerability,
│                      #   build_action_statement, build_status_notes
├── walker.py          # SPARQL queries against BomStore: components-by-purl,
│                      #   base-model lineage, dataset triples
├── cli.py             # Subparser builder (cli_subparser hook target)
├── web.py             # Flask Blueprint at /security (web_blueprint hook target)
├── spdx.py            # spdx_elements hook: Vulnerability + Relationship +
│                      #   Vex*VulnAssessmentRelationship emitter
├── overlay.py         # graph_overlay hook: per-node VEX tint payload
└── data/
    ├── sep_to_label.json   # Static: SEP code → human label (used in actionStatement)
    └── README.md           # Notes; bundled reports live in runtime cache, not here
```

### 3.2 Invariants

- The plugin directory is a self-contained drop-in.
- `data/` holds static config only. The 1784 AVID reports live in `~/.cache/aikaboom/avid-db/`, refreshed independently of the package version.
- `walker.py` uses the same `BomStore` SPARQL interface as `license_compat`. No reach into BOM internals.
- Core call sites (`web/app.py`, `cli.py`, `utils/spdx_exporter.py`) are touched **once** to add the plugin loop; never touched again as more plugins land.
- New JSON-LD `@context` aliases and SHACL Security shapes are merged into the existing bundled `spdx-context.jsonld` and `spdx-model.ttl` from the official SPDX 3.0.1 model — they are core extensions, not plugin code, because the SPDX viewer and validator must understand the Security namespace independent of which plugin emitted it.

### 3.3 Data flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    existing AIBOM pipeline                          │
│  Fetch ─► RAG/Direct extract ─► LLM ─► Provenance BOM ─► (conflict) │
└──────────────────────────────────────┬──────────────────────────────┘
                                       │ BomStore (RDF) + recursive child BOMs
                                       ▼
                       plugin loop (in utils/spdx_exporter.py)
                                       │
                                       ▼
              ┌── AvidSecurityPlugin.analyze(store, scope) ─────────┐
              │   1. walker.components() → List[Component]          │
              │   2. for each: matcher.match(component) → [Match]   │
              │   3. group matches by avid_report_id → [Finding]    │
              │   4. attach snapshot_sha to each Finding            │
              └─────────────────────────────────────────────────────┘
                                       │ Findings
              ┌─── AvidSecurityPlugin.spdx_elements(claim_iri, findings) ──┐
              │   - 1 security_Vulnerability per unique report_id          │
              │   - 1 hasAssociatedVulnerability per (component, report)   │
              │   - 1 Vex*Assessment per (component, report) — status      │
              │     depends on match tier                                  │
              │   - Returns List[dict] appended to SpdxDocument @graph     │
              └────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                    existing SHACL validator + SPDX viewer
                          + new Security/Vulnerabilities section
```

## 4. Plugin contract additions

This work depends on a foundational plugin scaffolding PR (PR-0) that introduces:

- `src/aikaboom/plugins/__init__.py` — registry: `register(plugin)`, `all_plugins()`, `get(name)`
- `src/aikaboom/plugins/base.py` — `Plugin` Protocol + `BomStore`, `Scope`, `Finding`, `GraphOverlay` dataclasses

PR-0 is being designed in parallel by the `license_compat` work. The AVID/Security plugin requires **two additive items** in PR-0 (both small, non-breaking):

### 4.1 New hook: `spdx_elements`

```python
class Plugin(Protocol):
    # ... existing hooks
    def spdx_annotations(self, claim_iri: str, findings: list[Finding]) -> list[dict]: ...

    # NEW — additive
    def spdx_elements(self, claim_iri: str, findings: list[Finding]) -> list[dict]:
        """Return any SPDX 3.0 element/relationship JSON-LD nodes (not just Annotations).

        The SPDX exporter appends these to the document's @graph after spdx_annotations.
        Plugins that only need Annotations leave this defaulted to []. AVID/security uses
        this for Vulnerability + VexAssessmentRelationship emission. Future plugins
        (supply-chain advisories, attribution) reuse the same generic hook.
        """
        return []
```

License-compat continues to use `spdx_annotations` unchanged. The exporter loop runs both hooks per plugin and concatenates results.

### 4.2 `Finding.snapshot_info: SnapshotInfo | None`

```python
@dataclass(frozen=True)
class SnapshotInfo:
    source: str            # e.g. "avid-db", "spdx-license-list", "ortelius-matrix"
    sha: str               # commit SHA or version identifier
    fetched_at: datetime   # when this snapshot was acquired locally

@dataclass(frozen=True)
class Finding:
    # ... existing fields (kind, severity, message, component_iri, etc.)
    snapshot_info: SnapshotInfo | None = None  # NEW
```

Optional field. AVID populates it with the avid-db SHA + fetch time. License-compat can populate it with the matrix version. The SPDX exporter writes any unique `snapshot_info` values into the document's `CreationInfo.comment` as a structured JSON blob — preserving reproducibility across plugins without each plugin inventing its own provenance schema.

## 5. Matcher tiers & VEX mapping

### 5.1 Component input

A `Component` is produced by `walker.components()` from the `BomStore`:

```python
@dataclass
class Component:
    kind: Literal["Model", "Dataset"]
    hf_path: str               # e.g. "google/gemma-3n-E4B-it"
    developer: str | None      # normalised via supplier_alias
    base_models: list[str]     # HF paths of upstream models in the lineage
    scope_in_bom: Literal["principal", "base", "dataset"]
    spdx_id: str               # the URN this component has in the SpdxDocument
```

### 5.2 AVID index keys (built once per snapshot in SQLite)

| Key | Definition |
|---|---|
| `bare_name` | lowercased `affects.artifacts[].name` (no org prefix) |
| `family_prefix` (set) | every hyphenated prefix of `bare_name` with ≥2 tokens, excluding `bare_name` itself. Example: `gemma-3n-E4B-it` yields the set `{gemma-3n-E4B, gemma-3n}`; `bert-base-uncased` yields `{bert-base}`; `gemma-3n` yields `∅`. Tier-3 fires when **any** prefix in the component's set intersects **any** prefix in the AVID record's set, and developer matches. |
| `developer` | normalised via `supplier_alias.py` aliases (`Google` ↔ `google-deepmind` ↔ `google`) |
| `artifact_kind` | `Model` / `Dataset` / `System` from `affects.artifacts[].type` |

### 5.3 Tier rules

| Tier | Trigger | Confidence | VEX status | VEX field populated |
|---|---|---|---|---|
| **1 — Exact** | `bare_name(component) == bare_name(avid)` AND `artifact_kind` matches | **high** | `affects` (Affected) | `actionStatement` (SHACL-required for VexAffected) |
| **2 — Base-model lineage** | A `base_models[*]` matches Tier-1 against AVID | **medium** | `underInvestigationFor` (UnderInvestigation) | `statusNotes` |
| **3 — Family + developer** | `family_prefix(component)` matches AVID `bare_name` prefix AND `developer` matches via alias | **low** | `underInvestigationFor` (UnderInvestigation) | `statusNotes` (different text from Tier 2) |

### 5.4 Statement templates

- **Tier 1 actionStatement:** `Mitigation: review AVID report {report_id} ({sep_view[0]}). No upstream fix recorded; apply {risk_domain[0]}-category guardrails appropriate to the deployment context.` AVID stores `sep_view` entries as `"E0101: Group fairness"` — code + label combined, so no separate label lookup is needed. `data/sep_to_label.json` exists for fallback when an AVID record's `sep_view` is missing the label.
- **Tier 2 statusNotes:** `Inherited from base model \`{base.hf_path}\`; downstream fine-tune may preserve or mask the issue. Re-evaluate against the AVID metric.`
- **Tier 3 statusNotes:** `Same family as AVID artifact \`{avid.bare_name}\` (developer \`{developer}\`); could impact this component — manual review needed to confirm applicability.`

### 5.5 Dedup rules

- A report matched against the same component through multiple tiers retains only the highest tier (no duplicate relationships).
- A report matched against multiple components produces **one shared** `Vulnerability` element + **N** `hasAssociatedVulnerability` Relationships + **N** `Vex*` relationships (one per component, status reflects per-component tier).
- Multiple AVID reports describing the same SEP code on the same component produce separate `Vulnerability` elements (each report is a distinct finding).

### 5.6 Confidence as plugin metadata

Each `Finding` carries `confidence: Literal["high", "medium", "low"]` and `evidence: dict[str, Any]` (matched keys, AVID report URL, snapshot SHA, matched_via). The plugin's web tab and graph overlay read these; external SPDX consumers see only the VEX status + statusNotes/actionStatement text.

## 6. SPDX emission

### 6.1 Tier-1 example output

For `bert-base-uncased` matched Tier-1 against `AVID-2022-R0001`:

```json
{
  "@graph": [
    {
      "type": "security_Vulnerability",
      "spdxId": "urn:aibom:vuln:avid-2022-r0001",
      "creationInfo": "_:creationinfo",
      "summary": "Gender Bias in Sentence Completion (HONEST) — bert-base-uncased",
      "description": "Sentence completions by bert-base-uncased were found to be significantly biased for one lexical category as defined by the HONEST hurtful sentence completion framework.",
      "publishedTime": "2022-11-09T00:00:00Z",
      "externalIdentifier": [{
        "type": "ExternalIdentifier",
        "externalIdentifierType": "securityOther",
        "identifier": "AVID-2022-R0001",
        "identifierLocator": ["https://avidml.org/database/AVID-2022-R0001"],
        "issuingAuthority": "urn:aibom:agent:avid-ml"
      }],
      "externalRef": [
        {"type": "ExternalRef", "externalRefType": "securityAdvisory",
         "locator": "https://github.com/avidml/avid-db/blob/{snapshot_sha}/reports/2022/AVID-2022-R0001.json"},
        {"type": "ExternalRef", "externalRefType": "securityOther",
         "locator": "https://huggingface.co/bert-base-uncased"}
      ]
    },
    {
      "type": "Relationship",
      "spdxId": "urn:aibom:rel:vuln-link-bert-base-uncased-avid-2022-r0001",
      "creationInfo": "_:creationinfo",
      "relationshipType": "hasAssociatedVulnerability",
      "from": "urn:aibom:pkg:bert-base-uncased",
      "to": ["urn:aibom:vuln:avid-2022-r0001"]
    },
    {
      "type": "security_VexAffectedVulnAssessmentRelationship",
      "spdxId": "urn:aibom:vex:bert-base-uncased-avid-2022-r0001",
      "creationInfo": "_:creationinfo",
      "relationshipType": "affects",
      "from": "urn:aibom:vuln:avid-2022-r0001",
      "to": ["urn:aibom:pkg:bert-base-uncased"],
      "security_assessedElement": "urn:aibom:pkg:bert-base-uncased",
      "security_actionStatement": "Mitigation: review AVID report AVID-2022-R0001 (E0101: Group fairness). No upstream fix recorded; apply Ethics-category guardrails appropriate to the deployment context.",
      "suppliedBy": ["urn:aibom:agent:aibom-avid-plugin"],
      "publishedTime": "{generation_time}"
    }
  ]
}
```

### 6.2 Tier-2/3 substitution

The third node becomes a `VexUnderInvestigationVulnAssessmentRelationship` with `relationshipType: "underInvestigationFor"` and a `security_statusNotes` field (per template in §5.4) replacing `security_actionStatement`.

### 6.3 SpdxId scheme

- `component_slug` is defined as: lowercase `hf_path` with `/` replaced by `__` and any character outside `[a-z0-9._-]` replaced by `-`. Example: `google/gemma-3n-E4B-it` → `google__gemma-3n-e4b-it`.
- `avid_id_lower` is the AVID report ID lowercased (`AVID-2026-R0478` → `avid-2026-r0478`).
- Vulnerability: `urn:aibom:vuln:{avid_id_lower}` — guarantees uniqueness across documents.
- hasAssociatedVulnerability Relationship: `urn:aibom:rel:vuln-link-{component_slug}-{avid_id_lower}`
- Vex* Relationship: `urn:aibom:vex:{component_slug}-{avid_id_lower}`
- Plugin Agent: `urn:aibom:agent:aibom-avid-plugin` (one per document, referenced by `suppliedBy`)

### 6.4 Snapshot SHA in CreationInfo

The SPDX exporter aggregates all unique `snapshot_info` values from plugin Findings and writes them into the document's `CreationInfo.comment` as JSON:

```json
{
  "plugin_snapshots": [
    {"source": "avid-db", "sha": "3f2a91c…", "fetched_at": "2026-05-12T10:14:00Z"}
  ]
}
```

This is the agreed plugin-agnostic provenance channel (§4.2). Reproducibility: regenerating with the same snapshot SHA produces **structurally identical** Vulnerability/VEX elements — same SpdxIds, same fields, same statement text. The Vex relationship's `publishedTime` is the BOM generation time (semantically: when the assessment was made) and will differ across runs. The `Vulnerability` element's `publishedTime` comes from the AVID report and is stable across runs.

### 6.5 JSON-LD / SHACL touchpoints (core change, not plugin)

- Extend bundled `schemas/spdx-context.jsonld` with the Security namespace aliases (`security_Vulnerability`, `security_VexAffectedVulnAssessmentRelationship`, `security_VexUnderInvestigationVulnAssessmentRelationship`, `security_actionStatement`, `security_assessedElement`, `security_statusNotes`) from the official SPDX 3.0.1 context at the same revision your `spdx-model.ttl` is pinned to.
- Extend bundled `schemas/spdx-model.ttl` with the Security profile shapes from the same model revision (the existing TTL is the Core+Software subset; we add Security).
- Ensure `utils/spdx_validator.py` constraints know about Security shapes — likely no code change, just the TTL extension.

## 7. Data acquisition (AVID snapshot)

### 7.1 Storage

- Path: `~/.cache/aikaboom/avid-db/` (overridable via `AIKABOOM_AVID_CACHE` env var).
- Contents: shallow `git clone --depth 1 https://github.com/avidml/avid-db.git` plus a sibling `snapshot.json` marker:

```json
{
  "sha": "3f2a91c…",
  "fetched_at": "2026-05-12T10:14:00Z",
  "ttl_days": 10
}
```

- SQLite index `~/.cache/aikaboom/avid-db.sqlite` built once per snapshot. Schema:

```sql
CREATE TABLE avid_report (
  report_id TEXT PRIMARY KEY,
  bare_name TEXT NOT NULL,
  family_prefix TEXT,
  developer TEXT,
  artifact_kind TEXT,
  sep_view TEXT,          -- JSON array
  risk_domain TEXT,       -- JSON array
  lifecycle_view TEXT,    -- JSON array
  published_date TEXT,
  source_path TEXT NOT NULL,  -- relative path inside the clone
  raw_json TEXT NOT NULL      -- the entire AVID JSON, for on-demand inflation
);
CREATE INDEX idx_bare_name ON avid_report(bare_name);
CREATE INDEX idx_family_prefix ON avid_report(family_prefix);
CREATE INDEX idx_developer_kind ON avid_report(developer, artifact_kind);
```

### 7.2 Refresh policy

- **TTL check** on every `analyze()` call. If `(now - fetched_at) > ttl_days`, refresh transparently before lookup. Default TTL: 10 days.
- **CLI**: `aikaboom security refresh-avid` forces immediate refresh regardless of TTL.
- **First-run**: if cache directory doesn't exist, clone synchronously on the first lookup. Future improvement (out of scope for v1): a background-refresh thread to avoid blocking.
- **Offline-safe**: if refresh fails (no network, GitHub down), log a warning and proceed with the existing cache. The BOM still records the existing snapshot SHA.

### 7.3 What's indexed vs. left in the JSON

- Top-level fields (`report_id`, `bare_name`, `developer`, `risk_domain`, `sep_view`, etc.) go into SQLite columns for fast lookup.
- The full report JSON is also stored in `raw_json` to avoid disk reads during description/reference extraction.

## 8. UX surfaces

### 8.1 Web Security tab (`web_blueprint()` → Flask Blueprint at `/security`)

- **Snapshot status bar** (top): `AVID snapshot {sha}  ·  fetched {date} ({N} days ago, auto-refresh in {M} days)  ·  [Refresh now]`
- **Filter chips**: VEX status (Affected / UnderInvestigation) · Confidence (high / medium / low) · Risk domain (Security / Ethics / Performance) · Scope (principal / base / dataset)
- **Finding card** (one per matched component × AVID report):
  - Component HF path + scope chip
  - VEX status badge + confidence chip + AVID report ID link
  - SEP code + label + lifecycle stage
  - actionStatement (Tier 1) or statusNotes (Tier 2/3)
  - Links: View AVID report, View in SPDX
- **Sort**: Affected first → UnderInvestigation; within each by scope (principal → base → dataset) → SEP code
- **Empty state**: explicitly names the snapshot SHA so absence is unambiguous

### 8.2 Graph overlay (`graph_overlay(findings) -> dict`)

Node tint by worst finding (per `findings_for_node`):

| Color | Trigger |
|---|---|
| 🔴 red | any Affected (Tier 1) |
| 🟠 orange | UnderInvestigation medium (Tier 2), no Affected |
| 🟡 yellow | only UnderInvestigation low (Tier 3) |
| no tint | no findings |

Click tinted node → side panel listing findings (with links back to Security tab cards). Toggle in graph header (`[ ] Show security findings`), off by default, persisted in session storage.

### 8.3 Conflicts tab integration (`conflict_findings(findings) -> list[dict]`)

Inject **Tier-1 / Affected only**. Form: `[Security · AVID]  {component_hf_path}  ·  {sep_code} {sep_label}  ·  {avid_report_id}`. UnderInvestigation findings stay in the Security tab only — keeps the Conflicts tab's color-count badge high-signal.

### 8.4 CLI (`cli_subparser(parent)`)

```
aikaboom security scan <bom.json>       # run analysis against an existing BOM, print table
aikaboom security refresh-avid          # force snapshot refresh
aikaboom security status                # snapshot SHA, fetched-at, TTL remaining
aikaboom generate --no-security         # opt-out flag on existing generate
```

`generate` runs security analysis by default once the plugin is installed (same default-on pattern as SPDX validation). Toggleable via `[security] enabled=false` in `.env`.

### 8.5 SPDX viewer extension (core, not plugin)

The existing viewer (`web/templates/spdx_viewer.html` + walker) renders `Annotation` distinctly (commits `bad7450`, `e8631e3`). Add:

- A "Vulnerabilities" section at the top of the document view, collapsed by default if >5 entries.
- `security_Vex*VulnAssessmentRelationship` rendered inline under each affected Package with the status badge + a "Show details" toggle for the actionStatement / statusNotes.

~50-line change in the viewer template + walker. Lives outside the plugin because the viewer must understand any plugin's emitted security elements without naming the plugin. Ships in the same PR as the plugin for self-contained delivery.

## 9. Testing & AVID Golden Set

### 9.1 Unit tests (fast, mocked, no network)

| File | What it tests |
|---|---|
| `tests/plugins/avid_security/test_matcher.py` | Table-driven (~30 cases). Inputs: a `Component` + in-memory `AvidIndex` fixture. Asserts tier/confidence/evidence. Covers: case-insensitive match, developer alias (`Google` ↔ `google-deepmind`), family_prefix tokenization (≥3 tokens), kind mismatch (Model vs Dataset) → no match, base-model lineage chain depth ≥3. |
| `tests/plugins/avid_security/test_engine.py` | Pure functions: `tier_to_vex_status`, `build_action_statement`, `build_status_notes`. No I/O. |
| `tests/plugins/avid_security/test_walker.py` | SPARQL queries against an rdflib in-memory `BomStore` fixture. Asserts walker uses only the documented `BomStore` interface. |
| `tests/plugins/avid_security/test_snapshot.py` | TTL logic with `tmp_path` + `freezegun`. Covers first-run clone, ≤10 days no refresh, >10 days refresh, manual refresh resets TTL, SHA persisted in marker. Network mocked. |
| `tests/plugins/avid_security/test_spdx_emitter.py` | Given Findings + claim IRI, asserts emitted JSON-LD shape per tier. Verifies shared-Vulnerability dedup (1 Vulnerability per AVID report_id across N components). |

TDD-first per `feedback_tdd_first`: write the failing test, watch it fail with the expected message, then implement.

### 9.2 AVID Golden Set

A new evaluation artifact dedicated to this feature, decoupled from the main 10-BOM golden set.

**Location:** `Golden_Set/AVID_Security/`

```
Golden_Set/AVID_Security/
├── README.md                # curation methodology + extension instructions
├── snapshot_sha.txt         # pinned avid-db SHA used during curation
├── cases.csv                # 10 curated cases (schema below)
└── avid_fixtures/           # bundled subset of avid-db reports referenced by cases.csv
    ├── AVID-2022-R0001.json
    ├── AVID-2026-R0478.json
    └── ...                  # ~10–20 files, ~50 KB total
```

**`cases.csv` schema:**

| Column | Purpose |
|---|---|
| `case_id` | Short tag e.g. `T1-bert-exact`, `T2-bert-finetune`, `neg-dev-mismatch` |
| `avid_report_id` | Real AVID ID — must exist in `avid_fixtures/` |
| `component_kind` | `Model` or `Dataset` |
| `component_hf_path` | e.g. `bert-base-uncased`, `google/gemma-3n-E4B-it` |
| `component_developer` | e.g. `Google`, `Hugging Face` (alias-normalised) |
| `component_base_models` | `;`-separated HF paths for lineage tests; empty otherwise |
| `component_scope` | `principal` / `base` / `dataset` |
| `expected_tier` | `1` / `2` / `3` / `no_match` |
| `expected_vex_status` | `affects` / `underInvestigationFor` / `none` |
| `expected_confidence` | `high` / `medium` / `low` / `none` |
| `expected_statement_contains` | Substring that must appear in actionStatement / statusNotes |
| `rationale` | One line — why this case is in the set |

**Curation target — 10 cases, distribution:**

| Count | Type | Coverage requirement |
|---|---|---|
| 4 | Tier 1 exact | At least 1 Model + 1 Dataset; spread across Security/Ethics/Performance risk domains |
| 2 | Tier 2 base-model lineage | Real HF fine-tune chains (e.g. `dslim/bert-base-NER` → base `bert-base-uncased` → matches `AVID-2022-R0001`) |
| 2 | Tier 3 family + developer | Real same-family models (e.g. `google/gemma-3n-E2B-it` matches `AVID-2026-R0478` via family_prefix + developer Google) |
| 2 | Negative controls | (a) bare-name collision with different developer; (b) kind mismatch (model name appearing in a dataset entry) |

**Curation methodology** (committed in `README.md`):
1. Pin to a recent avid-db SHA; write to `snapshot_sha.txt`.
2. Pick cases by walking **real** fine-tune lineages on HF — never synthesize them. Tier 2 chains must reference real `base_model` relationships from HF model cards. Tier 3 cases must use real same-family models, not invented strings.
3. For each chosen case, vendor only the referenced AVID JSON into `avid_fixtures/`.
4. The author manually walks the matcher rules for each row before adding it. CSV is ground truth.

**Example rows** (3 of 10; the rest filled in during implementation):

| case_id | avid_report_id | hf_path | developer | base_models | scope | tier | vex_status | conf | statement_contains | rationale |
|---|---|---|---|---|---|---|---|---|---|---|
| `T1-bert-honest` | AVID-2022-R0001 | `bert-base-uncased` | `Hugging Face` | | principal | 1 | `affects` | high | `E0101` | Exact bare-name + Model kind. Canonical Tier-1. |
| `T2-bert-ner-finetune` | AVID-2022-R0001 | `dslim/bert-base-NER` | `dslim` | `bert-base-uncased` | principal | 2 | `underInvestigationFor` | medium | `Inherited from base model` | Real HF fine-tune; tests base-model lineage triggers Tier 2. |
| `neg-gemma-other-dev` | AVID-2026-R0478 | `someone-else/gemma-3n-E4B-it` | `IndependentResearcher` | | principal | no_match | none | none | (n/a) | Same bare name, different developer → developer-disambiguation gates the false-positive. |

### 9.3 Test harness — `tests/plugins/avid_security/test_golden_set.py`

A single parametrised pytest that reads `Golden_Set/AVID_Security/cases.csv`, mounts `avid_fixtures/` as the AVID index, and per row asserts tier, confidence, VEX status, and statement substring. Failures point at a specific `case_id`. Runs in <1s with zero network.

### 9.4 Integration tests (per-plugin, vendored fixtures)

| Test | Verifies |
|---|---|
| `test_end_to_end_bert.py` | Generate AI BOM for `bert-base-uncased` → SPDX output contains expected Vulnerability + Relationship + VexAffected for curated fixtures. Snapshot SHA recorded in `CreationInfo.comment`. |
| `test_shacl_valid.py` | Full SHACL validation passes against extended `spdx-model.ttl` including Security shapes. |
| `test_jsonld_expand.py` | `rdflib` expand → no unresolved IRIs in Security namespace. |
| `test_recursive_lineage.py` | BOM with principal + base + dataset → one shared Vulnerability across matches, distinct VEX per component. |
| `test_plugin_isolation.py` | Plugin disabled via config → no security elements emitted, no Security namespace in `@context`, no `/security` blueprint mounted. Confirms the drop-in invariant. |

### 9.5 Acceptance criteria

- 100% of AVID golden-set cases pass. CSV is the regression baseline.
- PR description headlines: `4/4 Tier 1, 2/2 Tier 2, 2/2 Tier 3, 2/2 negative controls`.
- All unit + integration tests green.
- SHACL validation green against the extended Security shapes.
- Plugin-isolation test green (drop-in invariant preserved).

### 9.6 What's not tested in v1

- No live network in `pytest`.
- No LLM-in-the-loop matcher tests (matcher is deterministic).
- No headless-browser test of the SPDX viewer Security section. The viewer extension gets a small fixture-driven HTML-snapshot test, separate from the matcher golden set.
- No matcher false-positive rate against the full 1784-report avid-db. That would require manual triage at scale; deferred.

## 10. PR sequencing

Per `feedback_one_pr_per_feature`: one PR per feature, fresh branch off main.

- **PR-0 — foundation (someone else's work, in flight):** `src/aikaboom/plugins/{__init__.py, base.py}` — registry, decorator, Plugin Protocol, `BomStore`/`Scope`/`Finding`/`GraphOverlay` dataclasses. Includes the **two additive items** from §4 (the `spdx_elements` hook and `Finding.snapshot_info` field).
- **PR-1 — license_compat plugin (separate work):** First consumer of PR-0.
- **PR-2 — avid_security plugin (this design):** Off main; depends on PR-0 having landed. Contains:
  - `src/aikaboom/plugins/avid_security/*`
  - Core touch-once edits: `web/app.py` plugin loop (+ overlay hook), `cli.py` plugin loop, `utils/spdx_exporter.py` to call `spdx_elements` per plugin
  - Core schema extensions: `schemas/spdx-context.jsonld` and `schemas/spdx-model.ttl` extended with Security profile
  - Core viewer extension: `web/templates/spdx_viewer.html` Security section + walker change
  - `Golden_Set/AVID_Security/` with 10 curated cases + `avid_fixtures/` + `README.md` + `snapshot_sha.txt`
  - All tests from §9

PR-2 strictly depends on PR-0 — it does not ship until PR-0 has merged. The two additive items in §4 (the `spdx_elements` hook and `Finding.snapshot_info` field) are coordinated with the license_compat author **before** PR-0 lands, so PR-0 includes them from the start. This avoids any PR-2 fallback that would bundle scaffolding with feature work in violation of `feedback_one_pr_per_feature`.

## 11. Out of scope — v1.1 backlog

- MIT AI Risk Repository integration: static SEP→AIR-domain crosswalk lookup table, AIR tags attached to Vulnerability elements (one extra field) so users can filter by AIR domain in the Security tab.
- CVSS / EPSS / SSVC synthesis from SEP code → severity heuristic (only if defensible per-domain; otherwise stays out).
- Suppression UI ("mark as not applicable") emitting `VexNotAffected` with reviewer credit; requires a user/auth model.
- Standalone SBoVD JSON-LD export as a beta toggle (parallel to current CycloneDX/recursive_bom toggles). Default stays merged.
- Background-refresh thread for the AVID snapshot so first-after-TTL `analyze()` doesn't block on network.
- Headless-browser regression of the SPDX viewer Security section.

## 12. Risks & open questions

| Risk | Mitigation |
|---|---|
| AVID `affects.artifacts[].name` is informal — `bert-base-uncased` vs `BERT base uncased` vs `bert_base_uncased` | Index applies normalisation: lowercase, replace `_` and spaces with `-`, strip trailing version qualifiers. Tested in `test_matcher.py`. |
| Developer alias coverage is incomplete | Reuse and extend `supplier_alias.py`; new aliases are added as golden-set cases surface them. |
| Tier 2 lineage depends on BOM walker actually recording base-model relationships | Existing recursive BOM generation does this; `test_recursive_lineage.py` verifies the integration. |
| Tier 3 family_prefix may over-match across unrelated lineages (`bert-base` is also a prefix of many unrelated models) | Required `developer` match is the precision gate; negative-control case `neg-dev-mismatch` enforces it. |
| AVID-DB upstream schema changes | `snapshot.py` clones from a pinned SHA in `snapshot_sha.txt` for curated cases. Production snapshot follows `main` but is bounded by the TTL refresh — if a schema break lands, surfaces immediately in unit tests. |
| SHACL shapes for SPDX 3.0 Security may not be in the official model at the same revision we pinned | Verified during implementation; if a gap exists, we extend `spdx-model.ttl` from the source repo at HEAD and document the deviation. |

## 13. Definition of done

- All §9 acceptance criteria met.
- Spec doc (this file) committed and reviewed.
- Implementation plan (next step via `superpowers:writing-plans`) committed.
- Plugin landed as PR-2 off main, dependent on PR-0.
- AVID Golden Set lives at `Golden_Set/AVID_Security/` with 10 curated cases and a working test harness.
- Main BOM regression: zero impact when the plugin is disabled (plugin-isolation test green).
