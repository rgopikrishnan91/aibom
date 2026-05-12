<div align="center">
  <img src="docs/assets/aikaboom-logo.png" alt="AIkaBoOM" width="600">

  <h1>AIkaBoOM</h1>

  <p><em>Builds AI Bills of Materials by aggregating, aligning, and resolving conflicting metadata across the AI supply chain.</em></p>

  <p>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
    <a href="https://spdx.github.io/spdx-spec/v3.0.1/"><img src="https://img.shields.io/badge/SPDX-3.0.1-blue.svg" alt="SPDX 3.0.1"></a>
  </p>
</div>

---

AIkaBoOM extracts metadata from **HuggingFace**, **GitHub**, and **arXiv**, uses an LLM to populate structured BOM fields, and flags conflicts when sources disagree. The result is a JSON document with field-level provenance plus SPDX 3.0.1 JSON-LD validation. CycloneDX 1.6 ML-BOM export, recursive child BOM seed generation, and strict SPDX SHACL validation are available as beta features.

## Why?

- **Aggregate.** Pull metadata from every place it already lives: HF model cards, GitHub READMEs / LICENSE files, arXiv PDFs.
- **Align.** Normalize values across sources (license aliases, date formats, author handles).
- **Resolve.** When sources disagree, surface the conflict instead of silently picking one. Every field is a triplet: `{value, source, conflict}`.

## Quick Start

```bash
git clone https://github.com/rgopikrishnan91/aikaboom && cd aikaboom
python -m venv .venv && source .venv/bin/activate   # macOS / Linux
# Windows: python -m venv .venv && .venv\Scripts\activate
pip install -e .
cp .env.example .env   # add a provider key (OpenRouter / OpenAI / Ollama)
python run.py           # opens http://localhost:5000
```

> Requires **Python 3.9+**. On modern Debian / Ubuntu the venv step is
> mandatory — system Python blocks `pip install` outside a virtual
> environment (PEP 668). Conda works too:
> `conda create -n aikaboom python=3.11 -y && conda activate aikaboom`.

No API keys at all? Use Ollama for fully local processing:

```bash
ollama serve && ollama pull llama3:8b
# Set OLLAMA_BASE_URL=http://localhost:11434/v1/ in .env
```

## How It Works

```
  HuggingFace ──┐
                 │     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
  GitHub ────────┼────→│ RAG / Direct │────→│ LLM Engine  │────→│ Provenance   │
                 │     │  extraction  │     │ (OpenAI /   │     │ BOM (JSON)   │
  arXiv ─────────┘     └──────────────┘     │  Ollama /   │     └──────┬───────┘
                              │              │  OpenRouter)│            │
                       Local embeddings      └─────────────┘     ┌──────▼───────┐
                       (no API key)                              │ SPDX 3.0.1   │
                                                                 │ CycloneDX Beta │
                                                                 └──────────────┘
```

1. **Fetch** structured metadata via APIs and unstructured text via README scraping and PDF parsing.
2. **Extract** structured fields with an LLM, either via RAG (chunk + retrieve + generate) or direct prompting.
3. **Detect conflicts** between sources (majority voting, license similarity).
4. **Output** a JSON BOM with triplet fields, SPDX 3.0.1 JSON-LD, and optional beta CycloneDX / recursive child BOM exports.

> 📖 **Want the full code-level walkthrough?** See
> **[docs/PIPELINE_WALKTHROUGH.md](docs/PIPELINE_WALKTHROUGH.md)** — a
> single canonical doc that traces a request from "user pastes link" to
> the final SPDX export, with a per-field reference table (sources,
> priority, what's extracted, how conflicts work, SPDX target) and
> file:line refs into the code at every step.

## Usage

### Web UI

```bash
python run.py            # http://localhost:5000
# or:
aikaboom serve --port 5000
```

Pick BOM type (AI / Data), mode (RAG / Direct), and provider. For OpenRouter,
click **📋 Load model catalog** to fetch the live model list from
`/v1/models`. The **Provenance BOM** and **SPDX 3.0.1** export are generated automatically. The UI also exposes beta toggles/status for **CycloneDX 1.6 ML-BOM**, **recursive BOM generation**, and **Deep SHACL validation (beta)**. Server logs stream live in the **Logs** tab; the **Conflicts** tab shows a coloured count badge.

### CLI

```bash
# List OpenRouter models
aikaboom list-models --provider openrouter --limit 10

# Generate an AI BOM (provider auto-detected from .env)
aikaboom generate --type ai \
    --repo microsoft/DialoGPT-medium \
    --arxiv https://arxiv.org/abs/1911.00536 \
    --github https://github.com/microsoft/DialoGPT \
    --output result.json --spdx result.spdx.json --cyclonedx result.cdx.json

# Add slower semantic SHACL validation for a final SPDX check
aikaboom generate --type ai --repo org/model \
    --spdx result.spdx.json --strict-spdx-validation

# Generate beta recursive child BOM seed exports from trainedOn/testedOn/dependsOn
aikaboom generate --type ai --repo org/model \
    --output result.json \
    --recursive-bom --recursive-output result.recursive.json

# Cap the standalone SPDX's children-per-relationship list (default unbounded)
aikaboom generate --type ai --repo org/model \
    --spdx result.spdx.json --spdx-relationship-cap 10

# Generate a Dataset BOM
aikaboom generate --type data \
    --hf-url https://huggingface.co/datasets/squad \
    --github https://github.com/rajpurkar/SQuAD-explorer \
    --output result.json
```

The CLI auto-detects which LLM provider to use from the keys in your `.env`.
With multiple keys set, it asks. Pass `--provider` to override or `--yes` to
skip the prompt in scripts.

### Python API

```python
from aikaboom import AIBOMProcessor, DATABOMProcessor

processor = AIBOMProcessor(
    model="meta-llama/llama-3.3-70b-instruct",
    mode="rag",
    llm_provider="openrouter",
    use_case="complete",
)
result = processor.process_ai_model(
    repo_id="microsoft/DialoGPT-medium",
    arxiv_url="https://arxiv.org/abs/1911.00536",
    github_url="https://github.com/microsoft/DialoGPT",
)

# Convert to SPDX 3.0.1 (separate step, returns JSON-LD and validates by default)
from aikaboom.utils.spdx_validator import validate_bom_to_spdx, validate_spdx_export
from aikaboom.utils.cyclonedx_exporter import bom_to_cyclonedx
from aikaboom.utils.recursive_bom import generate_recursive_boms
spdx = validate_bom_to_spdx(result, bom_type="ai", output_path="out.spdx.json")
spdx_status = validate_spdx_export(spdx, bom_type="ai")

# Beta: run the slower SHACL semantic validator too
strict_spdx = validate_bom_to_spdx(result, bom_type="ai", strict=True)

# Beta: CycloneDX and recursive child BOM seed exports
cdx = bom_to_cyclonedx(result, bom_type="ai", output_path="out.cyclonedx.json")
recursive = generate_recursive_boms(result, bom_type="ai", max_depth=1)
```

See [`examples/`](examples/) for runnable scripts.

## Output Format

Every field is a **triplet** - the value, where it came from, and whether sources disagreed:

```json
{
  "repo_id": "microsoft/DialoGPT-medium",
  "model_id": "microsoft_DialoGPT-medium",
  "use_case": "complete",
  "direct_fields": {
    "license": {
      "value": "MIT",
      "source": "huggingface",
      "conflict": {
        "value": "Apache License 2.0",
        "source": "github",
        "type": "inter"
      }
    },
    "suppliedBy": {
      "value": "microsoft",
      "source": "huggingface",
      "conflict": null
    }
  },
  "rag_fields": {
    "domain": {
      "value": "Natural Language Processing, Dialogue Systems",
      "source": "arxiv, huggingface",
      "conflict": null
    }
  }
}
```

| Key             | Meaning |
|-----------------|---------|
| `value`         | The resolved field value. |
| `source`        | Which source(s) provided this value. |
| `conflict`      | What another source reported, if different. `null` = no conflict. |
| `conflict.type` | `"inter"` (different sources disagree) or `"intra"` (same source contradicts itself). |

A complete sample lives at [`examples/sample-output.json`](examples/sample-output.json).

#### `conflict` field — two shapes

The `conflict` slot has **different shapes** depending on whether a
field came from direct API metadata or from the RAG pipeline:

- **Direct fields** (`releaseTime`, `suppliedBy`, `downloadLocation`,
  `packageVersion`, `license`, …): `{value, source, type}` or `null`.
  The summary table above describes this shape.
- **RAG fields** (everything else — `domain`, `intendedUse`,
  `safetyRiskAssessment`, `metric`, …): a richer object with both a
  legacy summary and the Phase 4 trace block:

  ```json
  "domain": {
    "value": "Natural Language Processing",
    "source": "arxiv, huggingface",
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

  `internal` / `external` are the legacy strings (`"No"` or
  `"Yes: ..."`) for backwards compatibility; `trace` carries the
  per-source claim audit (which sources were inspected, who agreed,
  who was dropped by consensus routing).

Downstream consumers should be prepared to see either shape on
`conflict`. Unifying the two is on the future-work list; until then
the difference is intentional and documented here.

### Known limitations

- **LLM-extracted numeric metrics** (benchmark scores, parameter
  counts, training token counts) may be hallucinated. The pipeline
  passes the LLM's answer through unchanged; we do not currently
  cross-validate numeric claims against the cited chunk text. Verify
  any benchmark figure against the source paper before relying on
  it. A future phase may add cross-source agreement / re-prompt
  verification for numeric fields; today the answer is best-effort.
- **CycloneDX 1.6 emission, not 1.7.** The validator we adopted
  (`sbom-utility`) ships embedded JSON schemas for 1.2-1.6 only as
  of v0.18.x; we emit 1.6 so our own outputs can actually be
  validated end-to-end. None of CycloneDX 1.7's additions are used
  by AIkaBoOM today.
- **OpenRouter free tier is unsupported.** The previous free-model
  picker and `--free` flag were retired in Phase 10 — rate-limit
  caps (8 RPM on `:free` tiers) made the path unusable for any
  non-trivial run. OpenRouter as a provider stays; supply a paid
  model id explicitly.
- **License normalisation uses a hand-maintained alias table.** The
  intra-source license check (cardData vs README) and the SPDX-id
  canonicalisation downstream rely on `SPDX_ALIASES` in
  `aikaboom/utils/normalise.py` (~60 entries covering the most
  common forms) and a regex extractor for free-text mentions. A
  future phase may swap in a layered integration with
  [`license-expression`](https://github.com/aboutcode-org/license-expression)
  to pick up the full SPDX 3.x license registry, deprecated-form
  canonicalisation (e.g. `GPL-3.0` → `GPL-3.0-only`), and compound
  expression parsing (`Apache-2.0 OR MIT`). The two off-the-shelf
  candidates we evaluated —
  [`license-expression`](https://github.com/aboutcode-org/license-expression)
  and [`spdx-matcher`](https://pypi.org/project/spdx-matcher/) —
  both target adjacent problems (structured-expression parsing and
  full-text template matching, respectively); neither resolves
  colloquial cardData strings like `"Apache 2.0"` or `"GPLv3"` on
  its own, so a real swap would have to layer them on top of the
  current alias table rather than replace it.
- **Recursive walker conflict-gate is field-level, not claim-level.**
  When `--recursive-bom` walks `trainedOnDatasets` /
  `testedOnDatasets` / `modelLineage` / `sourceInfo`, it skips a
  field if the conflict detector flagged *any* contradiction in the
  retrieved chunks — even when the conflict is about an unrelated
  fact and the field's own value is unanimous across sources. Real
  case from the 2026-05-12 golden-set run: qwen's `modelLineage`
  was unanimously `Qwen/Qwen2.5-7B`, but the walker skipped it
  because the detector noted a side-disagreement about context
  length (`128K tokens` vs `131,072 tokens` — actually the same
  number) in the same chunks. Net effect: fewer recursive children
  than the data supports, especially on densely-described models.
  A claim-level fix would require the conflict detector to populate
  `statement_a`/`statement_b` reliably (the Phase 12B trace block
  is partially wired today) plus a similarity check in
  `recursive_bom._conflict_of` so only conflicts that actually
  disagree about the candidate target value gate the walk.

## Conflict Detection and Value Selection

A field-by-field reference of how each AI and Dataset BOM property is
resolved (sources, priority, normalisation, conflict criterion,
SPDX/CycloneDX export shape) lives in
[`docs/PIPELINE_WALKTHROUGH.md`](docs/PIPELINE_WALKTHROUGH.md#6-field-reference-table) —
the canonical code-level walkthrough also covers ingestion, the RAG
workflow, recursive walking, and validation in one place.

### Editing RAG question prompts

The RAG question bank lives as one JSON per field under
[`src/aikaboom/question_bank/`](src/aikaboom/question_bank/), split into
`ai/` and `data/` folders. Each file carries the question prompt, the
keywords used for retrieval, the SPDX-citing description, and the
`post_process` callable name (if any). Edit a file to tune one field's
prompt without touching Python; the loader picks the change up at the
next module-load. Source priority is **not** in these files — it lives
in [`src/aikaboom/config/source_priority.json`](src/aikaboom/config/source_priority.json)
so prompt edits and ranking edits don't collide in PR review.

Question-bank descriptions are sourced **verbatim** from the official
[`spdx/spdx-3-model`](https://github.com/spdx/spdx-3-model) at tag
`3.0.1`, harvested into
[`docs/SPDX_3.0.1_FIELD_REFERENCE.md`](docs/SPDX_3.0.1_FIELD_REFERENCE.md).
Each mapped field's `description` slot carries the spec's full Summary
and Description blocks; AIkaBoOM-internal fields
(`trainedOnDatasets`, `testedOnDatasets`, `modelLineage`, `sourceInfo`)
are flagged `"aikaboom_internal": true` and exempted. The regression
test [`tests/test_question_bank_descriptions.py`](tests/test_question_bank_descriptions.py)
fails CI if any entry drifts. To refresh after a SPDX rev:

```bash
python tools/harvest_spdx_3_0_1.py --version 3.1.0
python tools/sync_question_bank_descriptions.py --apply
```

AIkaBoOM detects two kinds of conflicts and surfaces both in the `conflict` field of every triplet.

**Inter-source.** Different sources report different values for the *same artefact-kind question*:
> HuggingFace says `license: MIT` but the GitHub LICENSE file says `Apache-2.0`.

**Intra-source.** A single source contradicts itself:
> HuggingFace metadata says `MIT` but the README text says "licensed under Apache 2.0."

Even when a conflict is flagged, AIkaBoOM still has to pick one value to put in the `value` slot of the triplet (and into the SPDX/CycloneDX exports). The resolution rules differ by field type:

**Artefact-platform direct fields.** AI BOM: `releaseTime`, `downloadLocation`, `packageVersion`. Dataset BOM: `builtTime`, `releaseTime`, `downloadLocation`, `contentIdentifier`. These fields describe *different artefacts* on HuggingFace (the model checkpoint or dataset) and GitHub (the supporting code repository) — their values diverge for structural reasons, not because the sources disagree about anything. `SourceHandler.get_field_per_platform` picks the priority winner as the canonical value and exposes every non-empty per-source value under `alternates`; no cross-platform "conflict" is emitted. `packageVersion` additionally runs through a SAIL name-extractor (`extract_version_from_repo_id`) so `mistralai/Mistral-7B-v0.1` resolves to the human-readable `7B-v0.1` rather than a git SHA prefix.

**Identity direct fields.** `suppliedBy` (AI) / `originatedBy` (Dataset) resolve through `SupplierAliasIndex` ([`src/aikaboom/utils/supplier_alias.py`](src/aikaboom/utils/supplier_alias.py)), which collapses known cross-platform identities (`Qwen` ↔ `QwenLM`, `cais` ↔ `hendrycks`, `google` ↔ `google-research`) before conflict comparison. Tiered resolution: (1) curated alias seed + harvested cross-references, (2) normalised-exact match, (3) Jaro-Winkler ≥ 0.85 for typo / prefix-shared variants.

**Artefact-kind direct fields.** `license` is the canonical case — both platforms claim to own this fact. Resolved by `SourceHandler.get_field_conflict_with_priority`: majority value wins; if all three disagree, the priority list from [`src/aikaboom/config/source_priority.json`](src/aikaboom/config/source_priority.json) selects the canonical value; the rest are preserved in the conflict string.

**RAG-extracted fields.** Everything else, including `license`, `primaryPurpose`, `datasetAvailability`, `description`, `sourceInfo`, plus the AI-package fields (`typeOfModel`, `domain`, `limitation`, `metric`, …) and the relationship targets `trainedOnDatasets` / `testedOnDatasets` / `modelLineage`. Each question's prompt, keywords, description, and post-processor live as one JSON file under [`src/aikaboom/question_bank/<bom_type>/<field>.json`](src/aikaboom/question_bank/); the per-field priority lives in `source_priority.json` and is overlaid at module load. The pipeline (LangGraph `retrieve → detect_conflicts → generate`) runs a single LLM call per field that both *extracts* one structured claim per source and *judges* every pairwise comparison against four explicit rules: (1) absence-of-information is silence, not conflict; (2) complementary lists (one source mentions A and B, the other B and C) are not conflict; (3) claims about different aspects of the same field are not conflict; (4) only directly opposing factual assertions are conflict. Every flagged conflict carries a calibrated `contradiction_score` (0.0–1.0) emitted by the auditor so downstream consumers can prioritise high-confidence cases. The Provenance BOM keeps the raw human-readable answer; the SPDX / CycloneDX emitters apply enum / list / DictionaryEntry coercion at export time.

Discovered links are LLM-validated against the target model: if the link agent returns an arXiv paper for the wrong model version, AIkaBoOM rejects it before fetching.

### Customising the source ranking

Source ranking lives in a single JSON config that ships with the package:
[`src/aikaboom/config/source_priority.json`](src/aikaboom/config/source_priority.json).
It has three sections — `direct_fields`, `rag_fields_ai`, `rag_fields_data` —
each mapping a field name to an ordered list of source names. Every section
also has a `default` entry that applies to fields with no explicit
priority. Edit the file in place, or override without forking by pointing
the `AIKABOOM_SOURCE_PRIORITY` environment variable at your own copy:

```bash
export AIKABOOM_SOURCE_PRIORITY=/path/to/my-source-priority.json
aikaboom generate --type ai --repo org/model --output result.json
```

A user config does **not** need to be exhaustive — every section merges
field-by-field over the bundled defaults, so listing just the entries you
want to change is enough. Example: prefer GitHub over HuggingFace for the
direct `suppliedBy` field, and put arXiv ahead of HuggingFace for the
RAG `license` answer:

```json
{
  "direct_fields": { "suppliedBy": ["github", "huggingface"] },
  "rag_fields_ai": { "license": ["arxiv", "huggingface", "github"] }
}
```

Programmatic access (and a hook for tests) is available via the package
API:

```python
from aikaboom import (
    load_source_priority,
    get_direct_priority,
    get_rag_priority,
    set_source_priority_path,
)

set_source_priority_path("/path/to/my-source-priority.json")  # or None to clear
print(get_direct_priority("suppliedBy"))      # -> ["github", "huggingface"]
print(get_rag_priority("license", "ai"))      # -> ["arxiv", "huggingface", "github"]
print(get_rag_priority("trainedOnDatasets"))  # -> ["huggingface", "arxiv", "github"]
```

A malformed user config logs a warning and falls back to the bundled
defaults, so a typo can't break BOM generation.

## Export Formats

AIkaBoOM always produces the native Provenance BOM and can emit standards-focused exports for downstream consumers. SPDX and CycloneDX are both AI BOMs and cover the same use cases — regulatory compliance (EU AI Act, NIST AI RMF), supply-chain transparency, DevSecOps, vulnerability management — they just express the same content in different vocabularies. Pick whichever your downstream tooling already speaks.

| Format | Standard | Notes |
|--------|----------|-------|
| **Provenance BOM** | AIkaBoOM JSON | Native format. Field-level source attribution + structured conflict triplets. |
| **SPDX 3.0.1** | [SPDX AI Profile](https://spdx.github.io/spdx-spec/v3.0.1/) JSON-LD | The same content as the Provenance BOM, expressed using the SPDX 3.0.1 AI Profile (`ai_AIPackage`, `dataset_DatasetPackage`, `trainedOn`/`testedOn`/`dependsOn`). |
| **CycloneDX 1.6 ML-BOM (beta)** | [CycloneDX ML-BOM](https://cyclonedx.org/) JSON | The same content as the Provenance BOM, expressed using the CycloneDX 1.6 ML-BOM (`modelCard`, `pedigree.ancestors`, `quantitativeAnalysis`). |
| **Recursive BOMs (beta)** | AIkaBoOM JSON bundle | Per-child BOMs for every `trainedOn` / `testedOn` / `dependsOn` target discovered in the dependency tree. |
| **Linked SPDX bundle (beta)** | SPDX 3.0.1 JSON-LD | Single `@graph` merging the parent and every recursive child with explicit Relationship edges; passes both lightweight and strict SPDX validation. |

```bash
aikaboom generate --type ai --repo org/model \
    --output result.json \
    --spdx result.spdx.json \
    --cyclonedx result.cyclonedx.json \
    --recursive-bom --recursive-depth 2 \
    --recursive-output result.recursive.json \
    --linked-bom-output result.linked.spdx.json
```

Both SPDX and CycloneDX exports preserve the same fields, the same provenance information, and the same conflict triplets — SPDX through `ai_AIPackage` / `dataset_DatasetPackage` and the SPDX `trainedOn` / `testedOn` / `dependsOn` relationships; CycloneDX through the `modelCard` extension (`modelParameters.task`, `modelParameters.architectureFamily`, `modelParameters.datasets`, `quantitativeAnalysis.performanceMetrics`, `considerations.technicalLimitations`) and `pedigree.ancestors` for lineage. Conflict triplets are carried in SPDX core fields and as `aikaboom:conflict:*` properties in CycloneDX. SPDX exports run through the bundled SPDX 3.0.1 JSON Schema by default; `--strict-spdx-validation` adds the official SHACL pass (beta). The same validators apply to the linked SPDX bundle.

**Recursive BOM generation is beta.** It walks the dependency tree of an AI BOM:

- `trainedOnDatasets` / `testedOnDatasets` produce *data* BOM children; `modelLineage` produces *AI* BOM children that may themselves have dependencies.
- Recursion is **conflict-gated**: any field whose RAG triplet has an internal or external conflict is skipped and surfaced in `skipped_due_to_conflict`. This guarantees we never recurse on contested data.
- A unique-target set deduplicates by `(bom_type, name)` so each artefact is fetched once and cycles can't loop. Duplicates are reported in `duplicates`.
- The walk stops at `--recursive-depth` (default 1, raisable; the web UI caps it at 5) or when the unique-target set is exhausted, whichever comes first. The result reports `deepest_level_reached` and `tree_exhausted`.
- Without an enrich callback the walker uses seed metadata only and the tree usually terminates after one level. Pass an `enrich_fn` to `generate_recursive_boms(...)` to plug in real fetching (e.g. wrap `AIBOMProcessor.process_ai_model`).

**Linked SPDX bundle (beta).** `--linked-bom-output` (or `build_linked_spdx_bundle(...)`) merges the parent SPDX export and every recursive child into one spec-clean SPDX 3.0.1 JSON-LD document with a single `@graph` and explicit `Relationship` elements wiring the dependency tree. Stub packages auto-emitted by the per-child SPDX export are de-duplicated against the recursive children, child `CreationInfo`/`Person`/`Organization` references are rebound onto the parent's, and the result passes both the lightweight (JSON Schema) and strict (SHACL) validators.

**Python API:**
```python
from aikaboom.utils.spdx_validator import validate_bom_to_spdx, validate_spdx_export
from aikaboom.utils.cyclonedx_exporter import bom_to_cyclonedx
from aikaboom.utils.recursive_bom import (
    generate_recursive_boms,
    build_linked_spdx_bundle,
    linked_bundle_summary,
)

spdx = validate_bom_to_spdx(result, bom_type='ai', output_path='out.spdx.json')
cdx = bom_to_cyclonedx(result, bom_type='ai', output_path='out.cyclonedx.json')

# Walk the dependency tree two levels deep with conflict gating
recursive = generate_recursive_boms(result, bom_type='ai', max_depth=2)

# One linked SPDX bundle with parent + every child + relationship edges
linked = build_linked_spdx_bundle(result, recursive, bom_type='ai')
summary = linked_bundle_summary(linked, recursive)
report = validate_spdx_export(linked, strict=True, bom_type='ai')
```

Validation is enabled by default for SPDX exports. Pass `validate=False` to `validate_bom_to_spdx(...)` to skip it, or `strict=True` to run the beta SHACL pass after JSON Schema. The CLI equivalents are `--no-validate-spdx` and `--strict-spdx-validation`. Use `--recursive-bom --recursive-output result.recursive.json` for the per-child bundle and `--linked-bom-output result.linked.spdx.json` for the single linked document.

## Use-Case Presets

Focus the BOM on a specific compliance need:

| Preset      | Focus                                                       |
|-------------|-------------------------------------------------------------|
| `complete`  | All fields (default).                                        |
| `safety`    | Risk assessment, bias, limitations, standards compliance.   |
| `security`  | Sensitive data, autonomy, security posture.                 |
| `lineage`   | Training data, preprocessing, hyperparameters.              |
| `license`   | License and standards compliance only.                      |

```bash
aikaboom generate --type ai --repo org/model --use-case safety
```

## LLM Providers

AIkaBoOM works with any OpenAI-compatible chat API. Pick the one that fits your environment.

| Provider     | When to use it                                                                                         |
|--------------|--------------------------------------------------------------------------------------------------------|
| OpenRouter   | Single key for many hosted models (Llama, Qwen, GPT, Claude, Mistral, …). Paid; no free-tier support.  |
| OpenAI       | If you already have credits or want the highest-quality reasoning.                                     |
| Ollama       | Fully local / offline. Pulls a model to your machine; no key required.                                 |

### Picking a model on HuggingFace Spaces

When deployed to HuggingFace Spaces (see below), the container itself does **not** host a large model. The Space calls out to whichever provider you configure:

- **OpenRouter** is the easiest path. Set `OPENROUTER_API_KEY` in the Space secrets, hit **Load model catalog**, pick a paid model id (e.g. `openai/gpt-4o-mini`, `meta-llama/llama-3.3-70b-instruct`).
- **OpenAI / Anthropic / Mistral hosted APIs** also work via OpenRouter.
- **Ollama-in-Spaces** is technically possible but constrained: the free Spaces tier has 16 GB RAM and 50 GB ephemeral disk, so only smaller models (~`llama3:8b`) fit, and cold-start times are long. Most users keep Ollama on their own hardware and only use Spaces for the web UI.
- The local embedding model (`BAAI/bge-small-en-v1.5`, ~50 MB) runs inside the Space without any configuration.

## Configuration

```bash
cp .env.example .env
```

```env
# Pick ONE LLM provider:
OPENAI_API_KEY=sk-...                       # Option 1: OpenAI
OPENROUTER_API_KEY=sk-or-...                # Option 2: OpenRouter (paid)
OLLAMA_BASE_URL=http://localhost:11434/v1/  # Option 3: Ollama (local, no key)

# Source API tokens (optional, increases rate limits):
GITHUB_TOKEN=ghp_...
HUGGINGFACE_TOKEN=hf_...

# Optional: enables the Link Fallback Agent (auto-discovers missing links)
GEMINI_API_KEY=AI...
```

RAG mode uses local HuggingFace embeddings by default - no OpenAI key needed for embeddings.

See [`.env.example`](.env.example) for every supported variable.

## Installation

```bash
git clone https://github.com/rgopikrishnan91/aikaboom && cd aikaboom
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Or with conda:

```bash
conda create -n aikaboom python=3.11 -y && conda activate aikaboom
pip install -e .
```

Requires Python 3.9+ (uses `importlib.resources.files()`). Tested on Linux, macOS, and Windows.

## Deploy to HuggingFace Spaces

Host the AIkaBoOM web UI on a free HuggingFace Space (Docker SDK) so anyone
can use it from a public URL. Full walkthrough in
[`docs/HF_SPACES.md`](docs/HF_SPACES.md). Short version:

```bash
git remote add hf https://huggingface.co/spaces/<you>/aikaboom
bash scripts/deploy_to_hf_spaces.sh
```

The repo ships with a `Dockerfile` and `README_HF.md` (HF-Spaces YAML
frontmatter). Set `OPENROUTER_API_KEY` in the Space's secrets and you are
done.

## Testing

```bash
pytest tests/                              # 720+ tests, ~3 min
pytest tests/ --cov=aikaboom --cov-report=html
```

A fresh clone with no API keys runs the suite green
(`pytest` will report 720+ passed, 2 skipped). The 2 skipped tests are
real-LLM integration tests gated behind `AIKABOOM_RUN_LLM_TESTS=1` (and
an `OPENROUTER_API_KEY`) because their answer-content assertions are
inherently nondeterministic.

## Troubleshooting

**Ollama connection issues.** Ensure Ollama is running (`ollama serve`) and reachable at `http://localhost:11434/api/tags`.

**Rate limits.** Set `GITHUB_TOKEN` and `HUGGINGFACE_TOKEN` in `.env` to lift API rate limits.

**Link Fallback Agent inactive.** That UI message means `GEMINI_API_KEY` is unset. Get a free key from <https://aistudio.google.com/app/apikey>; the rest of the tool still works without it.

**arXiv PDF parsing.** Complex PDFs may yield imperfect text. Try adding GitHub/HF sources, switching to Direct mode, or reviewing the retrieved evidence chunks in the UI.

## License

MIT License. See [LICENSE](LICENSE).
