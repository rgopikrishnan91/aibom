# Field Resolution Strategies

Every AIkaBoOM BOM field is either **Direct** (single structured signal from
the HuggingFace and GitHub APIs) or **Inferred** (RAG fusion over HF tags,
GH topics, READMEs, and arXiv text). Source ranking lives in
[`src/aikaboom/config/source_priority.json`](../src/aikaboom/config/source_priority.json)
and can be overridden per field via the `AIKABOOM_SOURCE_PRIORITY` env var.
Direct fields use HF + GH only; arXiv stays content-only and only enters via
the RAG pipeline.

Each row records the resolution **mode** (how the chosen value is picked),
the **normalisation** applied before comparison, and the **conflict
criterion** that flags disagreement. "Date-merge `latest`/`earliest`" picks
the most-recent / oldest date across sources. "Priority + majority" picks
the priority winner unless 2-of-3 sources agree on a normalised value.
"RAG" runs the LangGraph workflow (top-K retrieval → LLM conflict detection
→ priority-filtered answer).

## AI BOM

| Field | Class | Sources | Priority | Mode | Normalisation | Conflict |
|---|---|---|---|---|---|---|
| `releaseTime` | Direct | HF + GH | n/a | date-merge `latest` | `parse_date` | sources differ by > 7 days |
| `suppliedBy` | Direct | HF + GH | HF > GH | priority + majority | lowercase + org-alias map (empty default) | values disagree after normalisation |
| `downloadLocation` | Direct | HF + GH | HF > GH | priority | URL normaliser (lowercase host, strip `www.`, drop trailing `/`, drop fragment) | values disagree after normalisation |
| `packageVersion` | Direct | HF + GH | HF > GH | priority | version normaliser (strip leading `v`, drop build metadata) | values disagree after normalisation |
| `license` | Inferred | HF tags + GH + README + arXiv | HF > GH > arXiv | RAG | SPDX alias map + difflib ratio ≥ 0.8 (post-process) | RAG external/internal flag |
| `primaryPurpose` | Inferred | HF + GH + arXiv + README | HF > arXiv > GH | RAG | Jaccard ≥ 0.5 + SPDX `software_primaryPurpose` enum coercion | RAG flag |
| `autonomyType` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | SPDX presence enum coercion | RAG flag |
| `domain` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | list-of-strings | RAG flag |
| `energyConsumption` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | strip + collapse | RAG flag |
| `hyperparameter` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | DictionaryEntry coercion | RAG flag |
| `informationAboutApplication` | Inferred | HF + GH + arXiv | GH > HF > arXiv | RAG | strip + collapse | RAG flag |
| `informationAboutTraining` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | strip + collapse | RAG flag |
| `limitation` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | strip + collapse | RAG flag |
| `metric` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | DictionaryEntry coercion | RAG flag |
| `metricDecisionThreshold` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | DictionaryEntry coercion | RAG flag |
| `modelDataPreprocessing` | Inferred | HF + GH + arXiv | arXiv > GH > HF | RAG | list-of-strings | RAG flag |
| `modelExplainability` | Inferred | HF + GH + arXiv | arXiv > GH > HF | RAG | list-of-strings | RAG flag |
| `safetyRiskAssessment` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | SPDX safety-risk enum coercion | RAG flag |
| `standardCompliance` | Inferred | HF + GH + arXiv | arXiv > GH > HF | RAG | list-of-strings | RAG flag |
| `typeOfModel` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | list-of-strings | RAG flag |
| `useSensitivePersonalInformation` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | SPDX presence enum coercion | RAG flag |
| `trainedOnDatasets` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | dedupe-named-entities → list[str] | RAG flag |
| `testedOnDatasets` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | dedupe-named-entities → list[str] | RAG flag |
| `modelLineage` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | dedupe-named-entities → list[str] | RAG flag |

## Dataset BOM

| Field | Class | Sources | Priority | Mode | Normalisation | Conflict |
|---|---|---|---|---|---|---|
| `builtTime` | Direct | HF + GH | n/a | date-merge `earliest` | `parse_date` | sources differ by > 7 days |
| `originatedBy` | Direct | HF + GH | HF > GH | priority + majority | lowercase + org-alias map (empty default) | values disagree after normalisation |
| `releaseTime` | Direct | HF + GH | n/a | date-merge `latest` | `parse_date` | sources differ by > 7 days |
| `downloadLocation` | Direct | HF + GH | HF > GH | priority | URL normaliser | values disagree after normalisation |
| `contentIdentifier` | Direct | HF + GH | GH > HF | priority | lowercase SHA | hashes differ |
| `license` | Inferred | HF tags + GH + README + arXiv | HF > GH > arXiv | RAG | SPDX alias map + difflib ratio ≥ 0.8 | RAG external/internal flag |
| `primaryPurpose` | Inferred | HF + GH + arXiv + README | HF > arXiv > GH | RAG | Jaccard ≥ 0.5 + SPDX enum coercion | RAG flag |
| `datasetAvailability` | Inferred | HF + GH + arXiv + README | HF > GH > arXiv | RAG | enum coercion to {`directDownload`, `clickthrough`, `query`, `registration`, `scrapingScript`, `noAssertion`} | RAG flag |
| `description` | Inferred | arXiv + HF + GH | arXiv > HF > GH | RAG | strip + collapse whitespace | RAG flag |
| `sourceInfo` | Inferred | HF model-tree + READMEs + arXiv | arXiv > HF > GH | RAG | dedupe-named-entities → list[str] | RAG flag |
| `anonymizationMethodUsed` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | list-of-strings | RAG flag |
| `confidentialityLevel` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | SPDX confidentiality enum coercion | RAG flag |
| `dataCollectionProcess` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | strip + collapse | RAG flag |
| `dataPreprocessing` | Inferred | HF + GH + arXiv | GH > HF > arXiv | RAG | list-of-strings | RAG flag |
| `datasetNoise` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | strip + collapse | RAG flag |
| `datasetSize` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | int coercion (bytes) | RAG flag |
| `datasetType` | Inferred | HF + GH + arXiv | HF > GH > arXiv | RAG | SPDX dataset-type enum list | RAG flag |
| `datasetUpdateMechanism` | Inferred | HF + GH + arXiv | GH > HF > arXiv | RAG | strip + collapse | RAG flag |
| `hasSensitivePersonalInformation` | Inferred | HF + GH + arXiv | HF > arXiv > GH | RAG | SPDX presence enum coercion | RAG flag |
| `intendedUse` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | strip + collapse | RAG flag |
| `knownBias` | Inferred | HF + GH + arXiv | arXiv > HF > GH | RAG | list-of-strings | RAG flag |
| `sensorUsed` | Inferred | HF + GH + arXiv | arXiv > GH > HF | RAG | DictionaryEntry coercion | RAG flag |

## Status

The tables above match the implementation as of this commit:

- ✅ `license`, `primaryPurpose`, `datasetAvailability`, `sourceInfo`,
  `description` are all in the RAG pipeline; the synthetic structured
  chunks for HuggingFace and GitHub feed their tags / topics / license-
  header / model-tree into the RAG retriever alongside the README and
  arXiv text.
- ✅ `contentIdentifier` is implemented for Dataset BOMs (HuggingFace
  `repo_info.sha`, GitHub default-branch HEAD SHA).
- ✅ `releaseTime` and `builtTime` raise a 7-day-window conflict.
- ✅ URL, version, and org-name normalisers are wired into the direct
  resolution path. The org-name alias map ships empty; populate
  `_ORG_ALIASES` in `aikaboom.utils.normalise` to add canonical mappings.
- ✅ Per-question RAG post-processors (`normalize_license`,
  `normalize_purpose_enum`, `normalize_availability_enum`,
  `dedupe_named_entities`, `collapse_whitespace`) canonicalise LLM
  answers before they land in the triplet.

Remaining priority drift between this document and
[`config/source_priority.json`](../src/aikaboom/config/source_priority.json)
for inferred fields (e.g. `informationAboutApplication`,
`hasSensitivePersonalInformation`, `intendedUse`, `sensorUsed`) is
deliberate — those priorities are working-group choices that the config
exposes for community editing rather than a missing implementation.
