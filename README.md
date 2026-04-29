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

AIkaBoOM extracts metadata from **HuggingFace**, **GitHub**, and **arXiv**, uses an LLM to populate structured BOM fields, and flags conflicts when sources disagree. The result is a JSON document with field-level provenance plus an SPDX 3.0.1 export, suitable for AI governance, supply-chain transparency, and EU AI Act / NIST AI RMF compliance work.

## Why?

- **Aggregate.** Pull metadata from every place it already lives: HF model cards, GitHub READMEs / LICENSE files, arXiv PDFs.
- **Align.** Normalize values across sources (license aliases, date formats, author handles).
- **Resolve.** When sources disagree, surface the conflict instead of silently picking one. Every field is a triplet: `{value, source, conflict}`.

## Quick Start

```bash
git clone https://github.com/rgopikrishnan91/aibom && cd aibom
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
                                                                 └──────────────┘
```

1. **Fetch** structured metadata via APIs and unstructured text via README scraping and PDF parsing.
2. **Extract** structured fields with an LLM, either via RAG (chunk + retrieve + generate) or direct prompting.
3. **Detect conflicts** between sources (majority voting, license similarity).
4. **Output** a JSON BOM with triplet fields and an SPDX 3.0.1 JSON-LD export.

## Usage

### Web UI

```bash
python run.py            # http://localhost:5000
# or:
aikaboom serve --port 5000
```

Pick BOM type (AI / Data), mode (RAG / Direct), and provider. For OpenRouter,
click **🎯 Pick a free model** to load free models directly from
`/v1/models`. Both the **Provenance BOM** (with conflict triplets) and the
**SPDX 3.0.1** export are generated automatically. Server logs stream live
in the **Logs** tab; the **Conflicts** tab shows a coloured count badge.

### CLI

```bash
# List free OpenRouter models
aikaboom list-models --free --limit 10

# Generate an AI BOM (provider auto-detected from .env)
aikaboom generate --type ai \
    --repo microsoft/DialoGPT-medium \
    --arxiv https://arxiv.org/abs/1911.00536 \
    --github https://github.com/microsoft/DialoGPT \
    --output result.json --spdx result.spdx.json

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
skip the prompt in scripts. The legacy `bom-tools` command remains available
as an alias.

### Python API

```python
from bom_tools import (
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

# Convert to SPDX 3.0.1 (separate step, returns JSON-LD)
from bom_tools.utils.spdx_validator import validate_bom_to_spdx
spdx = validate_bom_to_spdx(result, bom_type="ai", output_path="out.spdx.json")
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

## SPDX 3.0.1 Conversion

Convert any BOM to [SPDX 3.0.1](https://spdx.github.io/spdx-spec/v3.0.1/) JSON-LD - the standard for software supply-chain transparency.

- **CLI:** `aikaboom generate ... --spdx output.spdx.json`
- **Web UI:** generated automatically alongside the Provenance BOM; click **Download SPDX 3.0.1**.
- **Python API:** `validate_bom_to_spdx(result, bom_type='ai', output_path='out.spdx.json')`

The SPDX output contains:
- `AI_AIPackage` (or `dataset_DatasetPackage`) elements with mapped fields
- `CreationInfo` with timestamp and tool attribution
- License relationships (`hasConcludedLicense`, `hasDeclaredLicense`)
- SPDX 3.0.1 JSON-LD structure with `@context` and `@graph`

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
git clone https://github.com/rgopikrishnan91/aibom && cd aibom
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
pytest --cov=bom_tools --cov-report=html
```

## Troubleshooting

**Ollama connection issues.** Ensure Ollama is running (`ollama serve`) and reachable at `http://localhost:11434/api/tags`.

**Rate limits.** Set `GITHUB_TOKEN` and `HUGGINGFACE_TOKEN` in `.env` to lift API rate limits.

**Link Fallback Agent inactive.** That UI message means `GEMINI_API_KEY` is unset. Get a free key from <https://aistudio.google.com/app/apikey>; the rest of the tool still works without it.

**arXiv PDF parsing.** Complex PDFs may yield imperfect text. Try adding GitHub/HF sources, switching to Direct mode, or reviewing the retrieved evidence chunks in the UI.

## License

MIT License. See [LICENSE](LICENSE).
