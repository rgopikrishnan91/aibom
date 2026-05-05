# Field Resolution Strategies

Every AIkaBoOM BOM field is either **Direct** (single structured signal from
the HuggingFace and GitHub APIs) or **Inferred** (RAG fusion over HF tags,
GH topics, READMEs, and arXiv text). Source ranking lives in
[`src/aikaboom/config/source_priority.json`](../src/aikaboom/config/source_priority.json)
and can be overridden per field via the `AIKABOOM_SOURCE_PRIORITY` env var.
Direct fields use HF + GH only; arXiv stays content-only and only enters via
the RAG pipeline.

Priorities reflect AIkaBoOM's shipped defaults — a working-group choice —
and are configurable via the JSON file above or the env var. The tables
below track the shipped defaults; tests in
[`tests/test_source_priority.py`](../tests/test_source_priority.py) lock
the runtime behaviour to the config (the file IS the spec).

Each row records the resolution **mode** (how the chosen value is picked),
the **normalisation** applied before comparison, the **conflict criterion**
that flags disagreement, and the **shape it takes when emitted to SPDX 3.0.1
or CycloneDX 1.7**. Date-merge `latest`/`earliest` picks the most-recent /
oldest date across sources. Priority + majority picks the priority winner
unless 2-of-3 sources agree on a normalised value. RAG runs the LangGraph
workflow (top-K retrieval → LLM conflict detection → priority-filtered
answer).

> **Coerced for export.** The Provenance BOM keeps the raw human-readable
> LLM answer (e.g. `"publicly downloadable"`, `"text-generation system"`).
> Coercion to SPDX-shaped values runs at export time inside
> `spdx_validator._normalize_enum` / `_normalize_enum_list` /
> `_dictionary_entries` / `_as_list` / `_coerce_dataset_size_bytes`.
> CycloneDX's coercion is beta and may diverge in edge cases.

## AI BOM

| Field | Class | Sources | Priority | Mode | Normalisation | Conflict | Coerced for export |
|---|---|---|---|---|---|---|---|
| `releaseTime` | Direct | HF + GH | n/a | date-merge `latest` | `parse_date` | sources differ by > 7 days | ISO 8601 timestamp |
| `suppliedBy` | Direct | HF + GH | HF > GH | priority + majority | lowercase + org-alias map (empty default) | values disagree after normalisation | str (Organization name) |
| `downloadLocation` | Direct | HF + GH | HF > GH | priority | URL normaliser (lowercase host, strip `www.`, drop trailing `/`, drop fragment) | values disagree after normalisation | str (URL) |
| `packageVersion` | Direct | HF + GH | HF > GH | priority | version normaliser (strip leading `v`, drop build metadata) | values disagree after normalisation | str |
| `license` | Inferred | HF tags + GH + README + arXiv | HF > GH > arXiv | RAG + post-process | SPDX alias map (`normalize_license`) + difflib ≥ 0.8 | RAG external/internal flag | str (SPDX expression) |
| `primaryPurpose` | Inferred | HF + GH + arXiv + README | HF > arXiv > GH | RAG (no post-process) | — | RAG flag | enum (SPDX `software_primaryPurpose`) |
| `autonomyType` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | — | RAG flag | enum (yes/no/noAssertion) |
| `domain` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | — | RAG flag | list[str] |
| `energyConsumption` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | — | RAG flag | str |
| `hyperparameter` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | — | RAG flag | DictionaryEntry list |
| `informationAboutApplication` | Inferred | HF + GH + arXiv | GH > HF > arXiv | RAG | — | RAG flag | str |
| `informationAboutTraining` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | — | RAG flag | str |
| `limitation` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | — | RAG flag | str |
| `metric` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | — | RAG flag | DictionaryEntry list |
| `metricDecisionThreshold` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | — | RAG flag | DictionaryEntry list |
| `modelDataPreprocessing` | Inferred | HF + GH + arXiv | arXiv > GH > HF | RAG | — | RAG flag | list[str] |
| `modelExplainability` | Inferred | HF + GH + arXiv | arXiv > GH > HF | RAG | — | RAG flag | list[str] |
| `safetyRiskAssessment` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | — | RAG flag | enum (low/medium/high/serious) |
| `standardCompliance` | Inferred | HF + GH + arXiv | arXiv > GH > HF | RAG | — | RAG flag | list[str] |
| `typeOfModel` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | — | RAG flag | list[str] |
| `useSensitivePersonalInformation` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | — | RAG flag | enum (yes/no/noAssertion) |

## Dataset BOM

| Field | Class | Sources | Priority | Mode | Normalisation | Conflict | Coerced for export |
|---|---|---|---|---|---|---|---|
| `builtTime` | Direct | HF + GH | n/a | date-merge `earliest` | `parse_date` | sources differ by > 7 days | ISO 8601 timestamp |
| `originatedBy` | Direct | HF + GH | HF > GH | priority + majority | lowercase + org-alias map | values disagree after normalisation | str (Organization name) |
| `releaseTime` | Direct | HF + GH | n/a | date-merge `latest` | `parse_date` | sources differ by > 7 days | ISO 8601 timestamp |
| `downloadLocation` | Direct | HF + GH | HF > GH | priority | URL normaliser | values disagree after normalisation | str (URL) |
| `contentIdentifier` | Direct | HF + GH | GH > HF | priority | lowercase SHA | hashes differ | str (hex SHA) |
| `license` | Inferred | HF tags + GH + README + arXiv | HF > GH > arXiv | RAG + post-process | SPDX alias map + difflib ≥ 0.8 | RAG flag | str (SPDX expression) |
| `primaryPurpose` | Inferred | HF + GH + arXiv + README | HF > arXiv > GH | RAG (no post-process) | — | RAG flag | enum (SPDX `software_primaryPurpose`) |
| `datasetAvailability` | Inferred | HF + GH + arXiv + README | HF > GH > arXiv | RAG (no post-process) | — | RAG flag | enum {`directDownload`, `clickthrough`, `query`, `registration`, `scrapingScript`, `noAssertion`} |
| `description` | Inferred | arXiv + HF + GH | arXiv > HF > GH | RAG + post-process | `collapse_whitespace` | RAG flag | str |
| `sourceInfo` | Inferred | HF model-tree + READMEs + arXiv | arXiv > HF > GH | RAG + post-process | `dedupe_named_entities` → list[str] | RAG flag | list[str] |
| `anonymizationMethodUsed` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | — | RAG flag | list[str] |
| `confidentialityLevel` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | — | RAG flag | enum (amber/clear/green/red) |
| `dataCollectionProcess` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | — | RAG flag | str |
| `dataPreprocessing` | Inferred | HF + GH + arXiv | GH > HF > arXiv | RAG | — | RAG flag | list[str] |
| `datasetNoise` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | — | RAG flag | str |
| `datasetSize` | Inferred | HF + GH + arXiv | HF > GH > arXiv | RAG | — | RAG flag | int (bytes); property omitted on no-assertion |
| `datasetType` | Inferred | HF + GH + arXiv | HF > GH > arXiv | RAG | — | RAG flag | enum-list (SPDX `dataset_datasetType`) |
| `datasetUpdateMechanism` | Inferred | HF + GH + arXiv | GH > HF > arXiv | RAG | — | RAG flag | str |
| `hasSensitivePersonalInformation` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | — | RAG flag | enum (yes/no/noAssertion) |
| `intendedUse` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | — | RAG flag | str |
| `knownBias` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | — | RAG flag | list[str] |
| `sensorUsed` | Inferred | HF + GH + arXiv | arXiv > GH > HF | RAG | — | RAG flag | DictionaryEntry list |

## Inferred relationships (recursive-BOM only; not SPDX 3.0.1 properties)

> These are AIkaBoOM-internal relationship targets, **not** SPDX 3.0.1
> properties on the package itself. They drive the recursive-BOM walker
> and are emitted as SPDX 3.0.1 `trainedOn` / `testedOn` / `dependsOn`
> *relationship types* between `ai_AIPackage` / `dataset_DatasetPackage`
> elements at SPDX export time. CycloneDX 1.7 represents them via
> `pedigree.ancestors` and `modelCard.datasets`.

| Field | Sources | Priority | Mode | Normalisation | Conflict | Coerced for export |
|---|---|---|---|---|---|---|
| `trainedOnDatasets` | HF + GH + arXiv | HF > arXiv > GH | RAG | — | RAG flag | SPDX `trainedOn` Relationship → DatasetPackage |
| `testedOnDatasets` | HF + GH + arXiv | arXiv > HF > GH | RAG | — | RAG flag | SPDX `testedOn` Relationship → DatasetPackage |
| `modelLineage` | HF + GH + arXiv | HF > arXiv > GH | RAG | — | RAG flag | SPDX `dependsOn` Relationship → AIPackage |

## Status

The tables above match the implementation as of this commit:

- ✅ `license`, `primaryPurpose`, `datasetAvailability`, `sourceInfo`,
  `description` are all in the RAG pipeline; synthetic structured chunks
  for HuggingFace and GitHub feed their tags / topics / license / model-tree
  / dataset siblings into the RAG retriever alongside README and arXiv text.
- ✅ `contentIdentifier` is implemented for Dataset BOMs (HF
  `repo_info.sha`, GH default-branch HEAD SHA).
- ✅ `releaseTime` / `builtTime` raise a 7-day-window inter-source
  conflict.
- ✅ URL, version, and org-name normalisers are wired into the direct
  resolution path. The org-name alias map ships empty; populate
  `_ORG_ALIASES` in `aikaboom.utils.normalise` to add canonical mappings.
- ✅ The Provenance BOM keeps human-readable RAG answers; SPDX/CycloneDX
  emitters apply enum / list / DictionaryEntry coercion at export time.
- ✅ `datasetSize` consumes a precise byte count from HF (sum of sibling
  file sizes) and GH (`repo.size * 1024`) via the synthetic structured
  chunk; arXiv free text is the fallback. Unparseable byte counts cause
  the SPDX emitter to omit the `dataset_datasetSize` property
  (no-assertion), avoiding a misleading `0`.
- ✅ Source priorities are a configurable design choice — the test
  `tests/test_source_priority.py::test_config_is_canonical_design_choice`
  locks the runtime behaviour to whatever
  `config/source_priority.json` declares.
- ✅ The RAG question bank lives as one JSON per field under
  `src/aikaboom/question_bank/<bom_type>/<field>.json`. Each file
  carries the question prompt, retrieval keywords, SPDX-citing
  description, and `post_process` name. Priority is layered on at
  module load from `config/source_priority.json`. Edit any field's
  prompt independently of the rest — the loader picks up the change
  at next process start.
