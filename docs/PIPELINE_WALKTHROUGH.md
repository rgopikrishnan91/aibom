# AIkaBoOM — Pipeline Walkthrough

**The canonical "what happens when I paste a link" guide.** This document
traces a single BOM-generation request through the entire pipeline, with
file:line references to the code that runs at each step. Where the README
shows you *how to use* the tool, this doc shows you *how the tool works*.

> If you only have five minutes, read §1 (the one-screen map) and §6 (the
> field-by-field reference table). Everything else is detail behind those
> two.

---

## Contents

1. [One-screen map of the pipeline](#1-one-screen-map-of-the-pipeline)
2. [Entry points: how a request actually gets in](#2-entry-points-how-a-request-actually-gets-in)
3. [Source ingestion: HuggingFace, GitHub, arXiv](#3-source-ingestion-huggingface-github-arxiv)
4. [The RAG pipeline in detail](#4-the-rag-pipeline-in-detail)
5. [Direct-field resolution](#5-direct-field-resolution)
6. [Field reference table — every BOM field, its sources, and how conflicts work](#6-field-reference-table)
7. [Recursive walker — depth-1 and beyond](#7-recursive-walker--depth-1-and-beyond)
8. [Validation — SPDX 3.0.1 and CycloneDX 1.6](#8-validation--spdx-301-and-cyclonedx-16)
9. [Output formats and where each lives](#9-output-formats-and-where-each-lives)
10. [End-to-end worked example](#10-end-to-end-worked-example)

---

## 1. One-screen map of the pipeline

```
            ┌──── User pastes link / runs CLI / hits /process ────┐
            │                                                       │
            ▼                                                       │
┌──────────────────────────────┐                                    │
│  Entry router                │                                    │
│  cli.py:cmd_generate         │                                    │
│  web/app.py:process          │                                    │
└──────────────┬───────────────┘                                    │
               │                                                    │
               ▼                                                    │
┌──────────────────────────────┐    Link Fallback Agent (optional)  │
│ (Optional) auto-fill missing │ ── uses Gemini API to discover     │
│ HF / arXiv / GitHub links    │    the missing sources             │
│ utils/link_fallback.py       │                                    │
└──────────────┬───────────────┘                                    │
               │                                                    │
               ▼                                                    │
┌─────────────────────────────────────────────────────────────────┐ │
│ AIBOMProcessor.process_ai_model  OR  DATABOMProcessor.process_dataset│
│ core/processors.py:495 / :842                                   │ │
└──────────────┬──────────────────────────────────────────────────┘ │
               │                                                    │
   ┌───────────┴───────────┐                                        │
   │                       │                                        │
   ▼                       ▼                                        │
┌──────────┐   ┌─────────────────────────────────────────────────┐  │
│ Fetch    │   │ Build structured chunks                         │  │
│ source   │   │ (HF tags, GH topics, model-tree, license)       │  │
│ objects  │   │ utils/metadata_fetcher.py:huggingface/github_   │  │
│ once     │   │   structured_chunk                              │  │
│ :447     │   └─────────────────────────────────────────────────┘  │
└──────────┘                                                        │
   │                                                                │
   ▼                                                                │
┌─────────────────────────────────────────────────────────────────┐ │
│ TWO parallel resolution paths                                   │ │
├─────────────────────────────────────────────────────────────────┤ │
│  DIRECT FIELDS                  │  RAG FIELDS                   │ │
│  releaseTime, suppliedBy,       │  All other fields (license,   │ │
│  downloadLocation,              │  domain, hyperparameter,      │ │
│  packageVersion, license, …     │  limitation, …)               │ │
│                                 │                               │ │
│  Read directly from HF / GH     │  For each field:              │ │
│  API objects. Per-platform      │    retrieve → detect_conflicts│ │
│  or priority-merge resolution.  │    → generate_answer          │ │
│  core/source_handler.py         │  core/agentic_rag.py:524      │ │
│  core/processors.py:_resolve_   │  (LangGraph workflow)         │ │
│    direct_fields_ai             │                               │ │
└─────────────────────────────────────────────────────────────────┘ │
               │                                                    │
               ▼                                                    │
┌─────────────────────────────────────────────────────────────────┐ │
│ Build the Provenance BOM (triplet shape per field)             │ │
│ core/processors.py:_build_triplet_payload (:143)                │ │
└──────────────┬──────────────────────────────────────────────────┘ │
               │                                                    │
   ┌───────────┼───────────┬─────────────────────────────────────┐  │
   │           │           │                                     │  │
   ▼           ▼           ▼                                     │  │
┌─────┐  ┌──────────┐ ┌──────────────────┐ ┌──────────────────┐  │  │
│SPDX │  │CycloneDX │ │Recursive walker  │ │Linked SPDX bundle│  │  │
│3.0.1│  │1.6 (beta)│ │(beta, depth N)   │ │(beta)            │  │  │
│JSON-│  │ML-BOM    │ │utils/recursive_  │ │utils/recursive_  │  │  │
│LD   │  │          │ │bom.py:generate_  │ │bom.py:build_     │  │  │
│     │  │          │ │recursive_boms    │ │linked_spdx_bundle│  │  │
└─────┘  └──────────┘ └──────────────────┘ └──────────────────┘  │  │
   │           │           │                       │              │  │
   └───────────┴───────────┴───────────────────────┘              │  │
                          │                                       │  │
                          ▼                                       │  │
            Validators (default + strict SHACL for SPDX;          │  │
            sbom-utility for CycloneDX)                           │  │
            utils/spdx_validator.py:validate_spdx_export          │  │
            utils/cyclonedx_validator.py:validate_cyclonedx       │  │
                                                                  │  │
                                       ◄──────────────────────────┘  │
                                                                     │
                                       ◄─────────────────────────────┘
```

The same processor object handles AI and dataset BOMs — only the source
objects, the question bank (`ai/` vs `data/`), and the SPDX class
(`ai_AIPackage` vs `dataset_DatasetPackage`) differ.

---

## 2. Entry points: how a request actually gets in

| Surface | Entrypoint | What happens |
|---|---|---|
| **Web UI** | `POST /process` in [`src/aikaboom/web/app.py:424`](../src/aikaboom/web/app.py) | Parses JSON form payload (`bom_type`, `mode`, provider/model, URLs, beta toggles). Runs the Link Fallback Agent if `GEMINI_API_KEY` is set (`web/app.py:488-535`), then calls `AIBOMProcessor.process_ai_model` or `DATABOMProcessor.process_dataset`. |
| **CLI** | `aikaboom generate …` → [`src/aikaboom/cli.py:164`](../src/aikaboom/cli.py) (`cmd_generate`) | Resolves provider, calls the same processor methods. Recursive / SPDX / CycloneDX exports are emitted inline based on flags. |
| **Python API** | `from aikaboom import AIBOMProcessor` ([`src/aikaboom/__init__.py`](../src/aikaboom/__init__.py)) | The processor classes are the canonical Python interface — both web and CLI are thin wrappers. |
| **HuggingFace Space** | Same Flask app as Web UI, packaged in [`Dockerfile`](../Dockerfile). | All web-UI semantics apply identically. See [`docs/HF_SPACES.md`](./HF_SPACES.md). |

**Recursive walking is off by default on every surface** and gated behind:
- CLI: `--recursive-bom` flag (`store_true` → off unless passed)
- Web UI: an unchecked checkbox + a JavaScript `confirm()` modal that
  warns about cost before the box can stay checked
  ([`templates/index.html:725-758, 868-880`](../src/aikaboom/web/templates/index.html))
- HF Space: identical to Web UI

---

## 3. Source ingestion: HuggingFace, GitHub, arXiv

AIkaBoOM treats every source as text that goes into one bucket per source,
plus a "structured chunk" of normalised metadata that gets prepended to
the same source's prose. Code lives in
[`src/aikaboom/utils/metadata_fetcher.py`](../src/aikaboom/utils/metadata_fetcher.py).

### 3.1 HuggingFace

| Fetched | How | Used by |
|---|---|---|
| `model_info` / `dataset_info` (cardData, tags, model-index, files, sha, lastModified) | `huggingface_hub` API at `processors.py:_fetch_source_objects` (line 447) | Direct fields (license, suppliedBy, releaseTime, etc.) + structured chunk |
| Inspected dict (`license`, `suppliedBy`, `downloadLocation`, `packageVersion`, `releaseTime`, modelLineage targets, …) | `MetadataFetcher.inspect_huggingface_BOM_Fields` | Direct-field resolver |
| README.md content | `MetadataFetcher.fetch_huggingface_readme` | RAG retriever |
| Structured-metadata chunk (YAML-style markdown of cardData + tags + model-tree) | `MetadataFetcher.huggingface_structured_chunk` | Prepended to the HF source bucket for RAG |
| Cross-platform identity aliases (`Qwen` ↔ `QwenLM`, `cais` ↔ `hendrycks`, …) | [`utils/supplier_alias.py`](../src/aikaboom/utils/supplier_alias.py) `SupplierAliasIndex` | `suppliedBy` / `originatedBy` conflict-collapse |

### 3.2 GitHub

Same shape: a PyGithub `Repository` object for the API, a `README.md`
fetch for prose, and a structured chunk (topics, license API result, repo
description, default branch). Code at the same `_fetch_source_objects`.
`GITHUB_TOKEN` raises the anonymous 60 req/hr cap to 5 000.

### 3.3 arXiv

Fetched as PDF text by `AgenticRAG.fetch_arxiv_pdf_text`
([`core/agentic_rag.py`](../src/aikaboom/core/agentic_rag.py)). Unlike HF /
GH, arXiv has no structured chunk — it enters the pipeline as PDF-extracted
prose only. Direct fields ignore arXiv entirely; arXiv contributes solely
to RAG retrieval.

### 3.4 Link Fallback (optional, web UI only)

If `GEMINI_API_KEY` is set and the user supplied fewer than 3 of the 3
sources, [`utils/link_fallback.py`](../src/aikaboom/utils/link_fallback.py)
asks Gemini to discover the missing links. Discovered links are
LLM-validated against the target model before fetching — if the link
agent returns the wrong arXiv paper, AIkaBoOM rejects it.

---

## 4. The RAG pipeline in detail

The RAG pipeline answers a structured question for each non-direct field
using **retrieval + LLM extraction + conflict audit**. It is implemented
as a 3-node [LangGraph](https://github.com/langchain-ai/langgraph) workflow
in [`src/aikaboom/core/agentic_rag.py:524`](../src/aikaboom/core/agentic_rag.py):

```
global_retrieve_top_k  →  detect_conflicts  →  generate_answer
   (file:agentic_rag.py:571 _build_workflow)
```

### 4.1 Chunking

AIkaBoOM uses a **header-aware** splitter — `HeaderAwareTextSplitter`
([`core/agentic_rag.py:248`](../src/aikaboom/core/agentic_rag.py)) — that:
- Cuts text at markdown headers so sections stay together (chunk size
  defaults to 1000 chars, overlap 200)
- Treats GitHub/HF markdown tables atomically — a metric table on a model
  card is emitted as a single chunk instead of fragmented row-by-row
  (`_emit_table_chunks`, line 466)
- Prepends the structured chunk (license / tags / model-tree) at the top
  of each source's prose before chunking

### 4.2 Embeddings

Local by default — `BAAI/bge-small-en-v1.5` via
`HuggingFaceEmbeddings` (`agentic_rag.py:556`). ~50 MB, runs on CPU,
no API key. The first run downloads the model to
`~/.cache/huggingface/hub/`. See
[`docs/LOCAL_EMBEDDINGS.md`](./LOCAL_EMBEDDINGS.md) for swapping
models or enabling GPU. OpenAI embeddings are still selectable
(`embedding_provider="openai"`) but require `OPENAI_API_KEY` and cost
money.

### 4.3 Retrieval (`_global_retrieve_top_k`, line 1217)

For every question:
1. Each source's text is indexed with both **FAISS dense vectors** and
   a **BM25 sparse index**.
2. The question's `keywords` and `hypothetical_passage` (from the
   question-bank JSON's `retrieval` block) drive a hybrid search.
3. Per-source **Reciprocal Rank Fusion** (k=60) merges the dense and
   sparse rankings.
4. The top chunks are bucketed by source. **No source is dropped at
   retrieval time** — even sources with weak overlap contribute their
   strongest chunk so the conflict detector can see what every source
   actually said.

### 4.4 Conflict detection (`_detect_conflicts`, line 591)

This is the heart of the Phase 12 conflict rebuild.

1. Retrieved chunks are bucketed into **anonymous groups** `A`, `B`,
   `C`, … one per source (`_build_groups`, line 620). The LLM only sees
   the group letter — never the source name — so it cannot bias toward
   any platform.
2. The LLM is asked to emit **one structured response** with three
   kinds of lines:
   ```
   CLAIM_A: <one-sentence summary of what group A actually says>
   CLAIM_B: <…> | "No relevant information"
   CONFLICT_WITHIN_A: No | Yes: "<stmt 1>" vs "<stmt 2>"
   CONFLICT_A_VS_B: No | Yes (confidence=0.85): A says "..." vs B says "..."
   ```
   The prompt template lives at
   [`core/prompt.py:prompt_detect_conflicts`](../src/aikaboom/core/prompt.py)
   and applies four explicit suppression rules
   (silence ≠ conflict; complementary lists ≠ conflict; different
   aspects of the same field ≠ conflict; only directly opposing
   factual assertions are conflict).
3. Output is parsed by `_parse_detector_output`
   ([`core/conflict_routing.py:176`](../src/aikaboom/core/conflict_routing.py))
   into structured fields:
   - `source_claims: Dict[source -> str | None]`
   - `internal_conflicts: Dict[source -> narrative]`
   - `external_conflicts: List[{sources, statement_a, statement_b, confidence, grounding_score}]`
4. Phase 12B adds a **grounding score** per external conflict — the
   audit re-scores `statement_a` and `statement_b` against the actual
   chunks they're claimed to come from. Scores 0.0–1.0; the BOM trace
   carries this so downstream consumers can threshold high-confidence
   vs low-confidence conflicts (`_score_grounding`, line 743).

**Short-circuit:** if only one source has chunks (0 or 1 group), the
LLM call is skipped entirely. Single-source intra-document detection
was retired in Phase 12 to cut cost and false positives.

### 4.5 Answer generation (`_generate_answer_node`)

After the conflict pass:
1. **Consensus-based source routing** (`_route_chunks`, line 812):
   sources flagged as self-contradicting in `internal_conflicts` are
   dropped; the remainder is then filtered by the field's priority
   list from
   [`config/source_priority.json`](../src/aikaboom/config/source_priority.json).
2. The surviving chunks are passed to the answer-generation LLM call
   with the question's `instruction`, `field_spec`, and
   `output_guidance` from the question-bank JSON
   ([`core/prompt.py:prompt_generate_answer`](../src/aikaboom/core/prompt.py)).
3. If `post_process` is set in the question JSON (e.g.
   `normalize_license`, `dedupe_named_entities`, `collapse_whitespace`),
   the answer runs through it before being stored.
4. The result lands in the BOM's `rag_fields` triplet:
   ```json
   "domain": {
     "value": "Natural Language Processing",
     "source": "huggingface, arxiv",
     "conflict": {
       "internal": "No",
       "external": "No",
       "trace": {
         "claims": { "huggingface": "...", "arxiv": "...", "github": "silent" },
         "selected_sources": ["huggingface", "arxiv"],
         "internal_conflicts": {},
         "external_conflicts": []
       }
     }
   }
   ```

### 4.6 The question bank — one JSON per field

Every RAG question lives as one file under
[`src/aikaboom/question_bank/<ai|data>/<field>.json`](../src/aikaboom/question_bank/).
Schema:

```json
{
  "field": "license",
  "bom_type": "ai",
  "question": "Under what license is the AI model and its code released?",
  "keywords": "license licensed under released under apache mit gpl bsd …",
  "description": "<verbatim from spdx/spdx-3-model at tag 3.0.1>",
  "post_process": "normalize_license",
  "retrieval": {
    "hypothetical_passage": "<paragraph mimicking ideal source text>",
    "bm25_terms": ["Apache-2.0", "MIT", "SPDX-License-Identifier", …]
  },
  "extraction": {
    "instruction": "Extract the license under which the model and its code are released, as an SPDX license identifier or expression.",
    "field_spec": "SPDX class `simplelicensing_LicenseExpression`. Value MUST be a valid SPDX license expression string per the SPDX License List …",
    "output_guidance": "Prefer the canonical SPDX identifier even if the source uses prose ('Apache 2.0' -> `Apache-2.0`) …"
  }
}
```

The `description` slot is sourced **verbatim** from the official
[`spdx/spdx-3-model`](https://github.com/spdx/spdx-3-model) at tag
`3.0.1` — see [`docs/SPDX_3.0.1_FIELD_REFERENCE.md`](./SPDX_3.0.1_FIELD_REFERENCE.md).
Four AIkaBoOM-internal relationship fields
(`trainedOnDatasets`, `testedOnDatasets`, `modelLineage`,
`sourceInfo`) carry `"aikaboom_internal": true` and have their own
descriptions. Tightening one field's prompt is a one-file edit — no
Python change needed.

---

## 5. Direct-field resolution

Direct fields are **read straight from the HF / GitHub API objects**
that `_fetch_source_objects` already pulled. No LLM is involved. The
code path is
[`core/processors.py:_resolve_direct_fields_ai`](../src/aikaboom/core/processors.py) (line 334)
and `_resolve_direct_fields_data` (line 702). Resolution uses
[`core/source_handler.py`](../src/aikaboom/core/source_handler.py):

- **`get_field_per_platform`** (line 204) — used for fields that
  describe *different artefacts* on different platforms
  (`releaseTime`, `downloadLocation`). HF reports the model checkpoint;
  GitHub reports the code repo. Their values diverging is structural,
  not a conflict. Picks the priority winner as the canonical value and
  exposes every non-empty value as `alternates`. No conflict is emitted.
- **`get_field_conflict_with_priority`** (line 243) — used for fields
  where every platform makes the *same* claim (`license`, `suppliedBy`,
  `packageVersion`). Majority wins; if all sources disagree, the
  priority list decides; non-winning values land in the conflict slot.

Identity collapse: `suppliedBy` and `originatedBy` run through
`SupplierAliasIndex`
([`utils/supplier_alias.py`](../src/aikaboom/utils/supplier_alias.py))
before comparison. Three tiers:
1. Curated alias seed (`Qwen` ↔ `QwenLM`, `cais` ↔ `hendrycks`, `google`
   ↔ `google-research`) plus a harvested cross-reference file
   ([`config/supplier_alias_harvest.json`](../src/aikaboom/config/supplier_alias_harvest.json)).
2. Normalised exact match (lowercased, hyphens stripped).
3. Jaro-Winkler ≥ 0.85 for typo / prefix-shared variants.

License-only intra-source check: `check_license_intra_source` compares
`cardData.license` against the same source's free-text README license
mention. Mismatches land in `direct_fields.license.intra_conflicts`.

The `packageVersion` field is special-cased through the **SAIL name
extractor**
([`utils/sail_version_extractor.py`](../src/aikaboom/utils/sail_version_extractor.py)) —
`mistralai/Mistral-7B-v0.1` resolves to the human-readable `7B-v0.1`
rather than a git-SHA prefix. The HF cardData / GitHub-tag cascade only
runs when the model name carries no size / version / variant signal.

---

## 6. Field reference table

The canonical per-field reference: **what gets extracted, from where,
with what priority, how conflicts work, and where it lands.**

### 6.1 Direct fields — AI BOM

These read from the HF / GitHub APIs only. No LLM call, no arXiv input.

| Field | Sources | Priority | What's extracted | Conflict mechanism | Lands in (Provenance / SPDX 3.0.1) |
|---|---|---|---|---|---|
| `releaseTime` | HF + GH | hf > gh | HF `lastModified`; GH default-branch head commit date | Per-platform — no conflict emitted; per-source value in `alternates` | `direct_fields.releaseTime` / `releaseTime` |
| `suppliedBy` | HF + GH | hf > gh | HF `author` / GH `owner.login` | Priority + majority with `SupplierAliasIndex` collapse (Tier 1-3) | `direct_fields.suppliedBy` / `suppliedBy` |
| `downloadLocation` | HF + GH | hf > gh | HF model page URL; GH `html_url` | Per-platform — no conflict emitted; per-source value in `alternates` | `direct_fields.downloadLocation` / `downloadLocation` |
| `packageVersion` | HF + GH + SAIL-name | huggingface_name > hf > gh | SAIL extractor (e.g. `7B-v0.1`); fallback HF git SHA / GH tag | Suppressed when SAIL provides a value (HF SHA vs GH tag isn't a real conflict) | `direct_fields.packageVersion` / `packageVersion` |
| `license` | HF cardData + GH license API | hf > gh | HF `cardData.license`; GH `license.spdx_id` | Priority + majority. License-only **intra-source** check (cardData vs README) attached as `intra_conflicts` | `direct_fields.license` / `hasDeclaredLicense` → `LicenseExpression` |

### 6.2 Direct fields — Dataset BOM

| Field | Sources | Priority | What's extracted | Conflict mechanism | Lands in (Provenance / SPDX 3.0.1) |
|---|---|---|---|---|---|
| `builtTime` | HF + GH | hf > gh | HF first-commit / GH first-commit | Date-merge `earliest`; > 7-day gap raises inter-source conflict | `direct_fields.builtTime` / `builtTime` |
| `originatedBy` | HF + GH | hf > gh | HF `author`; GH `owner.login` | Priority + majority with `SupplierAliasIndex` collapse | `direct_fields.originatedBy` / `originatedBy` |
| `releaseTime` | HF + GH | hf > gh | HF `lastModified`; GH head commit date | Per-platform | `direct_fields.releaseTime` / `releaseTime` |
| `downloadLocation` | HF + GH | hf > gh | HF dataset URL; GH `html_url` | Per-platform | `direct_fields.downloadLocation` / `downloadLocation` |
| `contentIdentifier` | HF + GH | **gh > hf** | HF `repo_info.sha`; GH default-branch HEAD SHA | Priority — hashes differ structurally | `direct_fields.contentIdentifier` / `contentIdentifier` |
| `license` | HF + GH | hf > gh | HF `cardData.license`; GH `license.spdx_id` | Priority + majority. Intra-source check attached as `intra_conflicts` | `direct_fields.license` / `hasDeclaredLicense` |

### 6.3 RAG fields — AI BOM (20 fields)

Every field below runs through the LangGraph workflow described in §4.
All three sources contribute. Priority order resolves ties **after**
conflict detection. The question-bank JSON lives at
`src/aikaboom/question_bank/ai/<field>.json` — click any field name to
open it.

| Field | Sources | Priority | What's extracted (from question JSON `instruction`) | Conflict | SPDX target |
|---|---|---|---|---|---|
| [`autonomyType`](../src/aikaboom/question_bank/ai/autonomyType.json) | hf + gh + arxiv | arxiv > hf > gh | Whether the AI system acts without human involvement | RAG (4-rule audit) | `ai_autonomyType` (enum) |
| [`domain`](../src/aikaboom/question_bank/ai/domain.json) | hf + gh + arxiv | arxiv > hf > gh | Domain / application area the model targets | RAG | `ai_domain` (list[str]) |
| [`energyConsumption`](../src/aikaboom/question_bank/ai/energyConsumption.json) | hf + gh + arxiv | hf > arxiv > gh | Reported energy figures (train + fine-tune + inference) | RAG | `ai_energyConsumption` |
| [`hyperparameter`](../src/aikaboom/question_bank/ai/hyperparameter.json) | hf + gh + arxiv | arxiv > hf > gh | Every training hyperparameter and its value | RAG | `ai_hyperparameter` (DictionaryEntry list) |
| [`informationAboutApplication`](../src/aikaboom/question_bank/ai/informationAboutApplication.json) | hf + gh + arxiv | **gh > hf > arxiv** | Software integration, pre/post-processing, deployment notes | RAG | `ai_informationAboutApplication` |
| [`informationAboutTraining`](../src/aikaboom/question_bank/ai/informationAboutTraining.json) | hf + gh + arxiv | arxiv > hf > gh | Training process — checkpoint, data, optimiser, schedule | RAG | `ai_informationAboutTraining` |
| [`license`](../src/aikaboom/question_bank/ai/license.json) | hf + gh + arxiv | hf > gh > arxiv | License as SPDX identifier (LLM-extracted; complements the direct-field license) | RAG + `normalize_license` post-process | `simplelicensing_LicenseExpression` |
| [`limitation`](../src/aikaboom/question_bank/ai/limitation.json) | hf + gh + arxiv | arxiv > hf > gh | Stated limitations, caveats, out-of-scope uses | RAG | `ai_limitation` |
| [`metric`](../src/aikaboom/question_bank/ai/metric.json) | hf + gh + arxiv | arxiv > hf > gh | Every evaluation metric: name + numeric value + dataset | RAG | `ai_metric` (DictionaryEntry list) |
| [`metricDecisionThreshold`](../src/aikaboom/question_bank/ai/metricDecisionThreshold.json) | hf + gh + arxiv | arxiv > hf > gh | Decision threshold(s) used to convert scores into predictions | RAG | `ai_metricDecisionThreshold` |
| [`modelDataPreprocessing`](../src/aikaboom/question_bank/ai/modelDataPreprocessing.json) | hf + gh + arxiv | arxiv > gh > hf | Preprocessing steps applied to training data | RAG | `ai_modelDataPreprocessing` (list[str]) |
| [`modelExplainability`](../src/aikaboom/question_bank/ai/modelExplainability.json) | hf + gh + arxiv | arxiv > gh > hf | Interpretability methods applied | RAG | `ai_modelExplainability` (list[str]) |
| [`modelLineage`](../src/aikaboom/question_bank/ai/modelLineage.json) **(internal, recursive)** | hf + gh + arxiv | hf > arxiv > gh | Base/parent checkpoint this model was derived from | RAG | SPDX `dependsOn` Relationship → `ai_AIPackage` |
| [`primaryPurpose`](../src/aikaboom/question_bank/ai/primaryPurpose.json) | hf + gh + arxiv | hf > arxiv > gh | SPDX `software_primaryPurpose` enum value | RAG | `software_primaryPurpose` (enum) |
| [`safetyRiskAssessment`](../src/aikaboom/question_bank/ai/safetyRiskAssessment.json) | hf + gh + arxiv | hf > arxiv > gh | Safety risk classification (low / medium / high / serious) | RAG | `ai_safetyRiskAssessment` (enum) |
| [`standardCompliance`](../src/aikaboom/question_bank/ai/standardCompliance.json) | hf + gh + arxiv | arxiv > gh > hf | Standards / regulations / frameworks complied with | RAG | `ai_standardCompliance` (list[str]) |
| [`testedOnDatasets`](../src/aikaboom/question_bank/ai/testedOnDatasets.json) **(internal, recursive)** | hf + gh + arxiv | arxiv > hf > gh | Evaluation / benchmark datasets only (NOT training data) | RAG | SPDX `testedOn` Relationship → `dataset_DatasetPackage` |
| [`trainedOnDatasets`](../src/aikaboom/question_bank/ai/trainedOnDatasets.json) **(internal, recursive)** | hf + gh + arxiv | hf > arxiv > gh | Training data ONLY (NOT evaluation benchmarks) | RAG | SPDX `trainedOn` Relationship → `dataset_DatasetPackage` |
| [`typeOfModel`](../src/aikaboom/question_bank/ai/typeOfModel.json) | hf + gh + arxiv | arxiv > hf > gh | Learning paradigm + architecture | RAG | `ai_typeOfModel` (list[str]) |
| [`useSensitivePersonalInformation`](../src/aikaboom/question_bank/ai/useSensitivePersonalInformation.json) | hf + gh + arxiv | hf > arxiv > gh | PII / biometric / health-data presence | RAG | `ai_useSensitivePersonalInformation` (enum) |

> The four **internal** fields (`trainedOnDatasets`, `testedOnDatasets`,
> `modelLineage`, `sourceInfo`) are **not** SPDX 3.0.1 properties on the
> package — they drive the recursive walker (§7) and are emitted as SPDX
> Relationship edges between packages at export time.

### 6.4 RAG fields — Dataset BOM (17 fields)

Same RAG mechanics as §6.3.

| Field | Sources | Priority | What's extracted | Conflict | SPDX target |
|---|---|---|---|---|---|
| [`anonymizationMethodUsed`](../src/aikaboom/question_bank/data/anonymizationMethodUsed.json) | hf + gh + arxiv | arxiv > hf > gh | Anonymization / de-identification techniques | RAG | `dataset_anonymizationMethodUsed` (list[str]) |
| [`confidentialityLevel`](../src/aikaboom/question_bank/data/confidentialityLevel.json) | hf + gh + arxiv | arxiv > hf > gh | Confidentiality classification | RAG | `dataset_confidentialityLevel` (enum: amber / clear / green / red) |
| [`dataCollectionProcess`](../src/aikaboom/question_bank/data/dataCollectionProcess.json) | hf + gh + arxiv | hf > arxiv > gh | How the dataset was collected: sources, methods, time period | RAG | `dataset_dataCollectionProcess` |
| [`dataPreprocessing`](../src/aikaboom/question_bank/data/dataPreprocessing.json) | hf + gh + arxiv | gh > hf > arxiv | Preprocessing steps applied to raw data | RAG | `dataset_dataPreprocessing` (list[str]) |
| [`datasetAvailability`](../src/aikaboom/question_bank/data/datasetAvailability.json) | hf + gh + arxiv | hf > gh > arxiv | Access mechanism → SPDX enum | RAG | `dataset_datasetAvailability` (enum) |
| [`datasetNoise`](../src/aikaboom/question_bank/data/datasetNoise.json) | hf + gh + arxiv | hf > arxiv > gh | Acknowledged noise, errors, quality issues | RAG | `dataset_datasetNoise` |
| [`datasetSize`](../src/aikaboom/question_bank/data/datasetSize.json) | hf + gh + arxiv | hf > gh > arxiv | Size in bytes (from HF file siblings) + auxiliary metrics | RAG | `dataset_datasetSize` (int bytes; omitted if unparseable) |
| [`datasetType`](../src/aikaboom/question_bank/data/datasetType.json) | hf + gh + arxiv | hf > gh > arxiv | Modality enum tokens (text / image / audio / structured / …) | RAG | `dataset_datasetType` (enum-list) |
| [`datasetUpdateMechanism`](../src/aikaboom/question_bank/data/datasetUpdateMechanism.json) | hf + gh + arxiv | gh > hf > arxiv | Update mechanism + cadence | RAG | `dataset_datasetUpdateMechanism` |
| [`description`](../src/aikaboom/question_bank/data/description.json) | hf + gh + arxiv | arxiv > hf > gh | 1-2 sentence factual description | RAG + `collapse_whitespace` | `description` |
| [`hasSensitivePersonalInformation`](../src/aikaboom/question_bank/data/hasSensitivePersonalInformation.json) | hf + gh + arxiv | hf > arxiv > gh | PII / biometric / health data presence | RAG | `dataset_hasSensitivePersonalInformation` (enum) |
| [`intendedUse`](../src/aikaboom/question_bank/data/intendedUse.json) | hf + gh + arxiv | arxiv > hf > gh | Intended + out-of-scope uses | RAG | `dataset_intendedUse` |
| [`knownBias`](../src/aikaboom/question_bank/data/knownBias.json) | hf + gh + arxiv | arxiv > hf > gh | Documented biases (demographic, language, coverage, …) | RAG | `dataset_knownBias` (list[str]) |
| [`license`](../src/aikaboom/question_bank/data/license.json) | hf + gh + arxiv | hf > gh > arxiv | SPDX license identifier (LLM-extracted) | RAG + `normalize_license` | `simplelicensing_LicenseExpression` |
| [`primaryPurpose`](../src/aikaboom/question_bank/data/primaryPurpose.json) | hf + gh + arxiv | hf > arxiv > gh | SPDX `primaryPurpose` enum (always `data` for dataset BOMs) | RAG | `software_primaryPurpose` |
| [`sensorUsed`](../src/aikaboom/question_bank/data/sensorUsed.json) | hf + gh + arxiv | arxiv > gh > hf | Sensors + calibration values | RAG | `dataset_sensorUsed` (DictionaryEntry list) |
| [`sourceInfo`](../src/aikaboom/question_bank/data/sourceInfo.json) **(internal, recursive)** | hf model-tree + READMEs + arxiv | arxiv > hf > gh | Named upstream datasets / models / papers | RAG + `dedupe_named_entities` | SPDX `dependsOn` Relationship → `dataset_DatasetPackage` |

### 6.5 How to read this table

- **Sources** lists which sources contribute. Direct fields only use
  HF + GH; RAG fields use all three.
- **Priority** is the tiebreaker — when sources disagree on a RAG
  field's value, the leftmost source's claim wins after consensus
  routing has dropped self-contradicting sources. Configurable in
  [`config/source_priority.json`](../src/aikaboom/config/source_priority.json)
  or via `AIKABOOM_SOURCE_PRIORITY=/path/to/your.json`.
- **What's extracted** is the question's `instruction` text from the
  JSON. The `field_spec` and `output_guidance` (also in the JSON)
  constrain the shape of the answer.
- **Conflict** is `RAG` (the 4-rule audit described in §4.4) for
  inferred fields, and per-platform / priority+majority for direct
  fields.
- **SPDX target** is what the SPDX 3.0.1 exporter emits. See
  [`docs/SPDX_3.0.1_FIELD_REFERENCE.md`](./SPDX_3.0.1_FIELD_REFERENCE.md)
  for the canonical Summary + Description blocks for each property.

---

## 7. Recursive walker — depth-1 and beyond

The recursive walker
([`utils/recursive_bom.py`](../src/aikaboom/utils/recursive_bom.py)) walks
relationship edges to produce **child BOMs** for every related artefact.

### 7.1 What gets walked

| Source field | Parent type | Child type | SPDX relationship |
|---|---|---|---|
| `trainedOnDatasets` | AI | data | `trainedOn` |
| `testedOnDatasets` | AI | data | `testedOn` |
| `modelLineage` | AI | AI | `dependsOn` |
| `sourceInfo` | data | data | `dependsOn` |

The mapping lives in `AI_RELATIONSHIP_FIELDS` and
`DATA_RELATIONSHIP_FIELDS` (lines 34-46). The walker is **default off**
on every surface (§2).

### 7.2 How the walker works

`generate_recursive_boms(metadata, bom_type, max_depth, …)`
(line 302) maintains a unique-target set keyed by
`(bom_type, normalised_name)` and a BFS frontier. For each parent
metadata dict:

1. **Discover targets** — `discover_recursive_targets` (line 156) reads
   the relationship fields, splits comma-separated values, and filters
   out non-walkable refs (arXiv IDs, URLs, DOIs) via
   `_is_walkable_target` (line 49).
2. **Conflict-gate** — `_conflict_of` (line 78) checks the parent
   BOM's triplet for an external or internal conflict on that
   relationship field. If a conflict is flagged, the field is added to
   `skipped_due_to_conflict` and **not** walked (this is the
   field-level gate; see the known-limitation entry in
   [README.md](../README.md#known-limitations)).
3. **Dedupe** — targets already in `visited` are added to `duplicates`
   instead of being walked again. Cycles can't loop.
4. **Enrich (optional)** — if an `enrich_fn` callback is provided,
   each target is enriched by fetching the actual HF / GH metadata and
   re-running the pipeline. Without it, children carry only seed
   metadata and the tree terminates after one level. The CLI plugs in
   a real enricher via
   [`utils/recursive_enrich.build_enrich_fn`](../src/aikaboom/utils/recursive_enrich.py).
5. **Stop** — the walk stops at `max_depth` (CLI `--recursive-depth`,
   default 1; Web UI capped at 5), when `safety_cap` is reached
   (default 50), or when the unique-target set is exhausted. The
   result reports `deepest_level_reached` and `tree_exhausted`.

### 7.3 Outputs

- `--recursive-output result.recursive.json` — a JSON bundle with one
  child BOM per generated entry, plus `visited`,
  `skipped_due_to_conflict`, `duplicates`, and the strategy / depth
  metadata.
- `--linked-bom-output result.linked.spdx.json` — a single SPDX 3.0.1
  JSON-LD document with parent + every recursive child + explicit
  `Relationship` edges between them. Built by
  `build_linked_spdx_bundle` (line 461). Stub packages are
  de-duplicated; child `CreationInfo` / `Person` / `Organization`
  references are rebound onto the parent's. Validates clean against
  both the JSON Schema and strict SHACL.

### 7.4 Worked example — CLIP (depth 1)

CLIP's parent BOM populated:
- `trainedOnDatasets` = `YFCC100M` (1 child)
- `testedOnDatasets` = `Food101, CIFAR10, …, Fairface` (34 entries)
- `modelLineage` = `openai/CLIP` (1 child)

After cross-relationship dedup (33 datasets appearing in *both*
trainedOn and testedOn are emitted only once, as testedOn), the walker
produced **35 child BOMs**: 34 dataset children with `testedOn` /
`trainedOn` relationships + 1 AI child with `dependsOn`. The linked
SPDX bundle was 124 KB and 188 graph elements; all validators pass.

---

## 8. Validation — SPDX 3.0.1 and CycloneDX 1.6

### 8.1 SPDX 3.0.1

Code at [`utils/spdx_validator.py`](../src/aikaboom/utils/spdx_validator.py).

Two validators, both real (proven by the adversarial corruption tests in
the project's comprehensive test report):

| Validator | Function | What it checks | Speed |
|---|---|---|---|
| **Default (JSON Schema)** | `validate_spdx_export(spdx, bom_type)` (line 1248) | The bundled SPDX 3.0.1 JSON Schema. Catches missing required fields, wrong types, bogus enum values. | Fast (<1s) |
| **Strict (SHACL)** | `validate_spdx_export(spdx, bom_type, strict=True)` | Adds the official SPDX SHACL shapes via `pyshacl`. Catches relationship-graph semantics the schema can't express. **Beta.** | Slow (3-10s per BOM) |

Both are on by default for emitted SPDX exports (the CLI's
`--no-validate-spdx` skips the default; `--strict-spdx-validation` adds
SHACL). The result payload carries `valid`, `strict`, `validator`, and
`errors` fields.

### 8.2 CycloneDX 1.6 (beta)

Code at
[`utils/cyclonedx_validator.py`](../src/aikaboom/utils/cyclonedx_validator.py).
External validator — uses the `sbom-utility` binary
(installed automatically; bundled JSON schemas for CycloneDX 1.2-1.6).
We emit CycloneDX 1.6 specifically because the validator's bundled
schemas stop there as of v0.18.x (see
[README known limitations](../README.md#known-limitations)).

---

## 9. Output formats and where each lives

| Format | File | Built by |
|---|---|---|
| **Provenance BOM** | `result.json` | `core/processors.py:process_ai_model` (line 495) / `process_dataset` (line 842) — the canonical AIkaBoOM JSON with triplets |
| **SPDX 3.0.1 JSON-LD** | `result.spdx.json` | [`utils/spdx_validator.validate_bom_to_spdx`](../src/aikaboom/utils/spdx_validator.py) (line 1205) |
| **CycloneDX 1.6 ML-BOM** | `result.cdx.json` | [`utils/cyclonedx_exporter.bom_to_cyclonedx`](../src/aikaboom/utils/cyclonedx_exporter.py) (line 331) |
| **Recursive child BOMs (bundle)** | `result.recursive.json` | [`utils/recursive_bom.generate_recursive_boms`](../src/aikaboom/utils/recursive_bom.py) (line 302) |
| **Linked SPDX bundle** | `result.linked.spdx.json` | [`utils/recursive_bom.build_linked_spdx_bundle`](../src/aikaboom/utils/recursive_bom.py) (line 461) |

All five outputs encode the same content. SPDX and CycloneDX are
alternate vocabularies for the same Provenance BOM data; pick whichever
your downstream tooling already speaks.

---

## 10. End-to-end worked example

Here is the full trace of what `aikaboom generate --type ai
--repo google-bert/bert-base-uncased --arxiv https://arxiv.org/abs/1810.04805
--github https://github.com/google-research/bert --output bert.json
--spdx bert.spdx.json --cyclonedx bert.cdx.json --recursive-bom
--recursive-output bert.recursive.json` does:

1. **CLI parse** ([`cli.py:cmd_generate`](../src/aikaboom/cli.py)) →
   resolve provider from `.env`, build `AIBOMProcessor`.
2. **Pre-fetch sources** ([`processors.py:_fetch_source_objects`](../src/aikaboom/core/processors.py))
   — one HF API call + one GH API call + one arXiv PDF download. Each
   returns a raw API object and an inspected dict.
3. **Build structured chunks** (`huggingface_structured_chunk`,
   `github_structured_chunk`) — normalised metadata text prepended to
   each source's README.
4. **Run RAG for 20 questions** ([`agentic_rag.py:process`](../src/aikaboom/core/agentic_rag.py)) —
   for each AI BOM field, run `retrieve → detect_conflicts →
   generate_answer`. Each question makes 2 LLM calls (one for the
   detector, one for the answer). With 3 sources and ~20 fields that's
   ~40 LLM calls; the parent BOM completes in ~3 minutes on
   `openai/gpt-4o-mini` via OpenRouter.
5. **Resolve direct fields** (`_resolve_direct_fields_ai`) — 5 fields
   resolved from the already-fetched API objects, no LLM. License
   intra-source check runs against the cached READMEs.
6. **Build the Provenance BOM** (`_build_triplet_payload`) — every
   field becomes a `{value, source, conflict}` triplet. The whole BOM
   is written to `bert.json`.
7. **Emit SPDX** (`validate_bom_to_spdx`) — converts the Provenance BOM
   to SPDX 3.0.1 JSON-LD, validates against the bundled JSON Schema,
   writes `bert.spdx.json`.
8. **Emit CycloneDX** (`bom_to_cyclonedx`) — converts to CycloneDX 1.6
   ML-BOM with the `modelCard` extension, runs `sbom-utility` validate,
   writes `bert.cdx.json`.
9. **Recursive walk** (`generate_recursive_boms` at depth 1) —
   discovers BERT's `trainedOnDatasets`, `testedOnDatasets`,
   `modelLineage` targets. Filters arXiv refs. After our Phase 12.6
   prompt fix, `trainedOn = "BookCorpus, English Wikipedia"` (SQuAD
   correctly excluded) → 2 trainedOn data BOMs + 3 testedOn data BOMs
   (GLUE, SQuAD, MRPC) + 1 dependsOn AI BOM. **8 child BOMs total.**
10. **Validate everything** — by the end, 4 artefacts have been written
    and validated against their respective standards.

For a multi-BOM rerun across the 5 AI + 5 dataset golden set, see
the live data in `test_outputs/comprehensive/REPORT.md` (gitignored —
regenerate locally with
`bash test_outputs/comprehensive/run_full.sh`).

---

## Cross-references

- **High-level usage:** [README.md](../README.md)
- **Quickstart:** [QUICKSTART.md](../QUICKSTART.md)
- **Per-field SPDX 3.0.1 spec text:** [SPDX_3.0.1_FIELD_REFERENCE.md](./SPDX_3.0.1_FIELD_REFERENCE.md)
- **Pre-Phase-12 field resolution tables (older):** [FIELD_STRATEGIES.md](./FIELD_STRATEGIES.md)
- **Local embeddings details:** [LOCAL_EMBEDDINGS.md](./LOCAL_EMBEDDINGS.md)
- **HF Spaces deployment:** [HF_SPACES.md](./HF_SPACES.md)
- **Known limitations:** [README.md#known-limitations](../README.md#known-limitations)
