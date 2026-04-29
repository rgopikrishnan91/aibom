<div align="center">
  <img src="docs/assets/aikaboom-logo.png" alt="AIkaBoOM" width="600">

  <h1>AIkaBoOM</h1>

  <p><em>Builds AI Bills of Materials by aggregating, aligning, and resolving conflicting metadata across the AI supply chain.</em></p>

  <p>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
    <a href="https://spdx.github.io/spdx-spec/v3.0.1/"><img src="https://img.shields.io/badge/SPDX-3.0.1-blue.svg" alt="SPDX 3.0.1"></a>
  </p>
</div>

---

AIkaBoOM extracts metadata from **HuggingFace**, **GitHub**, and **arXiv**, uses an LLM to populate structured BOM fields, and flags conflicts when sources disagree. The result is a JSON document with field-level provenance plus SPDX 3.0.1 JSON-LD validation. CycloneDX 1.7 export, recursive child BOM seed generation, and strict SPDX SHACL validation are available as beta features.

## Why?

- **Aggregate.** Pull metadata from every place it already lives: HF model cards, GitHub READMEs / LICENSE files, arXiv PDFs.
- **Align.** Normalize values across sources (license aliases, date formats, author handles).
- **Resolve.** When sources disagree, surface the conflict instead of silently picking one. Every field is a triplet: `{value, source, conflict}`.

## Quick Start

```bash
git clone https://github.com/rgopikrishnan91/aikaboom && cd aikaboom
pip install -e .
cp .env.example .env   # add a provider key (OpenRouter free tier works)
python run.py           # opens http://localhost:5000
```

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

## Usage

### Web UI

```bash
python run.py            # http://localhost:5000
# or:
aikaboom serve --port 5000
```

Pick BOM type (AI / Data), mode (RAG / Direct), and provider. For OpenRouter,
click **🎯 Pick a free model** to load free models directly from
`/v1/models`. The **Provenance BOM** and **SPDX 3.0.1** export are generated automatically. The UI also exposes beta toggles/status for **CycloneDX 1.7**, **recursive BOM generation**, and **Deep SHACL validation (beta)**. Server logs stream live in the **Logs** tab; the **Conflicts** tab shows a coloured count badge.

### CLI

```bash
# List free OpenRouter models
aikaboom list-models --free --limit 10

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

# Auto-pick a free OpenRouter model
aikaboom generate --type ai --repo org/model --pick-free-model

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
from aikaboom import (
    AIBOMProcessor, DATABOMProcessor,
    pick_free_openrouter_model, list_free_openrouter_models,
)

# Optional: pick a free model dynamically
model = pick_free_openrouter_model()

processor = AIBOMProcessor(
    model=model,
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

## Conflict Detection

AIkaBoOM detects two kinds of conflicts and surfaces both in the `conflict` field of every triplet.

**Inter-source.** Different sources report different values:
> HuggingFace says `license: MIT` but the GitHub LICENSE file says `Apache-2.0`.

Resolution: majority voting when 3+ sources agree, priority ordering otherwise.

**Intra-source.** A single source contradicts itself:
> HuggingFace metadata says `MIT` but the README text says "licensed under Apache 2.0."

Resolution: similarity scoring (difflib) between structured metadata and extracted free text. Flagged when similarity drops below 80%.

Discovered links are also LLM-validated against the target model: if the link agent returns an arXiv paper for the wrong model version, AIkaBoOM rejects it before fetching.

## Export Formats

AIkaBoOM always produces the native Provenance BOM and can emit standards-focused exports for downstream consumers. Each captures the same underlying data in a different shape.

| Format | Standard | Use case |
|--------|----------|----------|
| **Provenance BOM** | AIkaBoOM JSON | Field-level source attribution + conflict detection (our native format) |
| **SPDX 3.0.1** | [SPDX AI Profile](https://spdx.github.io/spdx-spec/v3.0.1/) JSON-LD | Regulatory compliance (EU AI Act, NIST), supply chain transparency |
| **CycloneDX 1.7 (beta)** | [CycloneDX ML-BOM](https://cyclonedx.org/) JSON | DevSecOps integration, vulnerability management workflows |
| **Recursive BOMs (beta)** | AIkaBoOM JSON bundle | Per-child BOMs for every `trainedOn` / `testedOn` / `dependsOn` target discovered in the dependency tree |
| **Linked SPDX bundle (beta)** | SPDX 3.0.1 JSON-LD | Single `@graph` merging the parent and every recursive child with explicit Relationship edges; passes both lightweight and strict SPDX validation |

```bash
aikaboom generate --type ai --repo org/model \
    --output result.json \
    --spdx result.spdx.json \
    --cyclonedx result.cyclonedx.json \
    --recursive-bom --recursive-depth 2 \
    --recursive-output result.recursive.json \
    --linked-bom-output result.linked.spdx.json
```

**SPDX 3.0.1** output includes `ai_AIPackage` elements, `trainedOn`/`testedOn`/`dependsOn` relationships to dataset/model stubs, license relationships, and JSON-LD validated by the bundled SPDX 3.0.1 JSON Schema. With `--strict-spdx-validation` the same document also runs through the official SPDX SHACL shapes (beta). The CLI and web UI report pass/fail without blocking export. The same validators apply to the linked SPDX bundle.

**CycloneDX 1.7 is beta.** It uses the `modelCard` extension: `modelParameters.task`, `modelParameters.architectureFamily`, `modelParameters.datasets`, `quantitativeAnalysis.performanceMetrics`, `considerations.technicalLimitations`, and `pedigree.ancestors` for model lineage. Conflict data is preserved as `aikaboom:conflict:*` properties.

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
| OpenRouter   | **Recommended for free / hobby use.** Free models available, click "Pick a free model" in the UI.    |
| OpenAI       | If you already have credits or want the highest-quality reasoning.                                     |
| Ollama       | Fully local / offline. Pulls a model to your machine; no key required.                                 |

### Picking a model on HuggingFace Spaces

When deployed to HuggingFace Spaces (see below), the container itself does **not** host a large model. The Space calls out to whichever provider you configure:

- **OpenRouter `:free` models** are the easiest path. Set `OPENROUTER_API_KEY` in the Space secrets and use the in-app **Pick a free model** button. Works on the free CPU tier.
- **OpenAI / Anthropic / Mistral hosted APIs** also work via OpenRouter.
- **Ollama-in-Spaces** is technically possible but constrained: the free tier has 16 GB RAM and 50 GB ephemeral disk, so only smaller models (~`llama3:8b`) fit, and cold-start times are long. Most users keep Ollama on their own hardware and only use Spaces for the web UI.
- The local embedding model (`BAAI/bge-small-en-v1.5`, ~50 MB) runs inside the Space without any configuration.

## Configuration

```bash
cp .env.example .env
```

```env
# Pick ONE LLM provider:
OPENAI_API_KEY=sk-...                       # Option 1: OpenAI
OPENROUTER_API_KEY=sk-or-...                # Option 2: OpenRouter (free models available)
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

Requires Python 3.8+. Tested on Linux, macOS, and Windows.

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
pytest                                    # 234+ tests
pytest --cov=aikaboom --cov-report=html
```

## Troubleshooting

**Ollama connection issues.** Ensure Ollama is running (`ollama serve`) and reachable at `http://localhost:11434/api/tags`.

**Rate limits.** Set `GITHUB_TOKEN` and `HUGGINGFACE_TOKEN` in `.env` to lift API rate limits.

**Link Fallback Agent inactive.** That UI message means `GEMINI_API_KEY` is unset. Get a free key from <https://aistudio.google.com/app/apikey>; the rest of the tool still works without it.

**arXiv PDF parsing.** Complex PDFs may yield imperfect text. Try adding GitHub/HF sources, switching to Direct mode, or reviewing the retrieved evidence chunks in the UI.

## License

MIT License. See [LICENSE](LICENSE).
