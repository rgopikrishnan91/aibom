# AVID Security Golden Set

10 curated cases used as the regression baseline for the `avid_security` plugin matcher.

## File layout

- `snapshot_sha.txt` — pinned avid-db SHA used during curation
- `cases.csv` — 10 curated cases (schema below)
- `avid_fixtures/` — only the AVID JSON reports referenced by `cases.csv`

## CSV schema

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

## Curation methodology

1. Pin to a recent avid-db SHA; write to `snapshot_sha.txt`.
2. Pick cases by walking **real** fine-tune lineages on HuggingFace — never synthesize them.
   Tier 2 chains must reference real `base_model` relationships from HF model cards.
   Tier 3 cases must use real same-family models, not invented strings.
3. For each chosen case, vendor only the referenced AVID JSON into `avid_fixtures/`.
4. The author manually walks the matcher rules for each row before adding it. The CSV is ground truth —
   if the matcher disagrees with a row, either the matcher needs fixing or the row was curated wrong.

## Distribution target (10 cases)

| Count | Type | Coverage requirement |
|---|---|---|
| 4 | Tier 1 exact | At least 1 Model + 1 Dataset; spread across Security/Ethics/Performance risk domains |
| 2 | Tier 2 base-model lineage | Real HF fine-tune chains |
| 2 | Tier 3 family + developer | Real same-family models |
| 2 | Negative controls | (a) bare-name collision with different developer; (b) kind mismatch |

Cases are curated in Task 6.2; this task only bootstraps the structure.
