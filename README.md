# BOM Tools

Generate Software Bills of Materials for AI models and datasets - with source-level conflict detection and SPDX 3.0.1 support.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![SPDX 3.0.1](https://img.shields.io/badge/SPDX-3.0.1-blue.svg)](https://spdx.github.io/spdx-spec/v3.0.1/)

BOM Tools extracts metadata from **HuggingFace**, **GitHub**, and **arXiv**, uses an LLM (via RAG or direct extraction) to populate structured BOM fields, and flags conflicts when sources disagree. Output is a JSON document with field-level provenance - each field records its value, which source it came from, and whether other sources reported something different. Results can be converted to [SPDX 3.0.1](https://spdx.github.io/spdx-spec/v3.0.1/) format.

## Why?

- **Transparency** - Know what's inside your AI models and datasets: training data, licenses, limitations, safety risks
- **Compliance** - Generate SPDX 3.0.1-compliant AI and Dataset BOMs for regulatory needs (EU AI Act, ISO/IEC standards)
- **Conflict detection** - Automatically flag when GitHub says "MIT" but HuggingFace says "Apache-2.0"

## How It Works

```
  HuggingFace ──┐
                 │     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
  GitHub ────────┼────→│ RAG / Direct │────→│ LLM Engine  │────→│ BOM JSON │
                 │     │  Extraction  │     │ (OpenAI /   │     │ (triplet │
  arXiv ─────────┘     └──────────────┘     │  Ollama /   │     │  fields) │
                              │             │  OpenRouter) │     └────┬─────┘
                       Local embeddings     └─────────────┘          │
                       (no API key needed)                     ┌─────▼──────┐
                                                               │ SPDX 3.0.1 │
                                                               └────────────┘
```

1. **Fetch** metadata from source APIs (HuggingFace model cards, GitHub repos, arXiv PDFs)
2. **Extract** structured fields using an LLM - either via RAG (embed + retrieve + generate) or direct prompting
3. **Detect conflicts** between sources using majority voting and license similarity checks
4. **Output** a JSON BOM with triplet fields, optionally converting to SPDX 3.0.1

## Quick Start

```bash
git clone https://github.com/rgopikrishnan91/aibom && cd aibom
pip install -e .
cp .env.example .env   # add your LLM provider key
python run.py           # open http://localhost:5000
```

**No API keys?** Use [Ollama](https://ollama.com/) for fully local processing:

```bash
ollama serve && ollama pull llama3:8b
# Set OLLAMA_BASE_URL=http://localhost:11434/v1/ in .env
# Select "Ollama" as provider in the web UI
```

## Usage

### Web UI

```bash
python run.py
# Open http://localhost:5000
```

Select BOM type (AI / Data), processing mode (RAG / Direct), and LLM provider. Both the Provenance BOM (with conflict triplets) and SPDX 3.0.1 export are generated automatically.

### CLI

```bash
# Generate an AI model BOM (provider auto-detected from .env)
bom-tools generate --type ai \
    --repo microsoft/DialoGPT-medium \
    --arxiv https://arxiv.org/abs/1911.00536 \
    --github https://github.com/microsoft/DialoGPT \
    --output result.json --spdx result.spdx.json

# Generate a dataset BOM
bom-tools generate --type data \
    --hf-url https://huggingface.co/datasets/squad \
    --github https://github.com/rajpurkar/SQuAD-explorer \
    --output result.json

# Start web UI
bom-tools serve --port 5000
```

The CLI auto-detects which LLM provider to use based on the API keys in your
`.env` file. If only `OPENROUTER_API_KEY` is set, it uses OpenRouter; if multiple
keys are set, it asks which one you want. Pass `--provider openai|openrouter|ollama`
to override, or `--yes` to skip the confirmation prompt in scripts.

### Python API

**AI Model BOM:**

```python
from bom_tools.core.processors import AIBOMProcessor

processor = AIBOMProcessor(
    model="gpt-4o",
    mode="rag",
    llm_provider="openai",
    use_case="complete"
)

result = processor.process_ai_model(
    repo_id="microsoft/DialoGPT-medium",
    arxiv_url="https://arxiv.org/abs/1911.00536",
    github_url="https://github.com/microsoft/DialoGPT"
)
```

**Dataset BOM:**

```python
from bom_tools.core.processors import DATABOMProcessor

processor = DATABOMProcessor(
    model="gpt-4o",
    mode="rag",
    llm_provider="openai",
    use_case="complete"
)

result = processor.process_dataset(
    arxiv_url="https://arxiv.org/abs/1606.05250",
    github_url="https://github.com/rajpurkar/SQuAD-explorer",
    hf_url="https://huggingface.co/datasets/squad"
)
```

See [examples/](examples/) for complete runnable scripts.

## Output Format

Every field is a **triplet** - the value, where it came from, and whether sources disagreed:

```json
{
  "repo_id": "microsoft/DialoGPT-medium",
  "model_id": "microsoft_DialoGPT-medium",
  "use_case": "complete",
  "direct_fields": {
    "suppliedBy": {
      "value": "microsoft",
      "source": "huggingface",
      "conflict": null
    },
    "license": {
      "value": "MIT",
      "source": "huggingface",
      "conflict": {
        "value": "Apache License 2.0",
        "source": "github",
        "type": "inter"
      }
    }
  },
  "rag_fields": {
    "domain": {
      "value": "Natural Language Processing, Dialogue Systems",
      "source": "arxiv, huggingface",
      "conflict": null
    },
    "typeOfModel": {
      "value": "GPT-2 based autoregressive language model",
      "source": "arxiv",
      "conflict": null
    }
  }
}
```

| Key | Meaning |
|-----|---------|
| `value` | The resolved field value |
| `source` | Which source(s) provided this value |
| `conflict` | What another source reported, if different (`null` = no conflict) |
| `conflict.type` | `"inter"` = different sources disagree; `"intra"` = same source is internally inconsistent |

See [examples/sample-output.json](examples/sample-output.json) for a complete example.

## Conflict Detection

BOM Tools automatically detects when metadata sources disagree.

**Inter-source conflicts** - different sources report different values:
> HuggingFace model card says `license: MIT` but the GitHub repo's LICENSE file says `Apache-2.0`.

Resolution: majority voting when 3+ sources available; priority ordering otherwise.

**Intra-source conflicts** - the same source contradicts itself:
> HuggingFace API metadata says `MIT` but the README text says "licensed under the Apache License 2.0."

Resolution: similarity scoring (difflib) between structured metadata and extracted text. Flagged when similarity drops below 80%.

Both conflict types appear in the `conflict` field of each triplet, with `type: "inter"` or `type: "intra"`.

## SPDX 3.0.1 Conversion

Convert any BOM output to [SPDX 3.0.1](https://spdx.github.io/spdx-spec/v3.0.1/) format - the standard for software supply chain transparency.

**Three ways to generate SPDX:**

1. **CLI**: `bom-tools generate --type ai --repo org/model --spdx output.spdx.json`
2. **Web UI**: Always generated - see the "SPDX 3.0.1" tab and "Download SPDX 3.0.1" button after processing
3. **Python API**:

```python
from bom_tools.utils.spdx_validator import validate_bom_to_spdx

spdx = validate_bom_to_spdx(result, bom_type='ai', output_path='output.spdx.json')
```

The SPDX output contains:
- `AI_AIPackage` or `dataset_DatasetPackage` elements with mapped fields
- `CreationInfo` with generation timestamp
- License relationships (`hasConcludedLicense`, `hasDeclaredLicense`)
- SPDX 3.0.1 JSON-LD structure with `@context` and `@graph`

## Use-Case Presets

Focus BOM generation on specific compliance needs:

| Preset | Focus |
|--------|-------|
| `complete` | All fields (default) |
| `safety` | Safety risks, bias, limitations, compliance |
| `security` | Security posture, sensitive data, autonomy |
| `lineage` | Training data, preprocessing, hyperparameters |
| `license` | License and standards compliance only |

```bash
bom-tools generate --type ai --repo org/model --use-case safety
```

## Configuration

Copy `.env.example` and fill in the keys you need:

```bash
cp .env.example .env
```

```env
# Pick ONE LLM provider:
OPENAI_API_KEY=sk-...          # Option 1: OpenAI
OPENROUTER_API_KEY=sk-or-...   # Option 2: OpenRouter (free models available)
OLLAMA_BASE_URL=http://localhost:11434/v1/  # Option 3: Ollama (local, no key)

# Source API tokens (optional, increases rate limits):
GITHUB_TOKEN=ghp_...
HUGGINGFACE_TOKEN=hf_...

# Optional - enables automatic link discovery:
GEMINI_API_KEY=AI...
```

RAG mode uses local HuggingFace embeddings by default - no OpenAI key needed for embeddings.

See [.env.example](.env.example) for all available variables.

## Installation

```bash
# Clone and install
git clone https://github.com/rgopikrishnan91/aibom && cd aibom
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env  # edit with your keys
```

Or with conda:

```bash
conda create -n bom-tools python=3.11 -y && conda activate bom-tools
pip install -e .
```

Requires Python 3.8+. Tested on Linux, macOS, and Windows.

## Testing

```bash
pytest
pytest --cov=bom_tools --cov-report=html
```

## Troubleshooting

**Ollama connection issues** - Ensure Ollama is running (`ollama serve`) and reachable at `http://localhost:11434/api/tags`.

**Rate limits** - Set `GITHUB_TOKEN` and `HUGGINGFACE_TOKEN` in `.env` to increase API rate limits.

**arXiv PDF parsing** - Complex PDFs may yield imperfect text. Try adding GitHub/HF sources, using Direct mode, or reviewing the retrieved evidence chunks.

## License

MIT License. See [LICENSE](LICENSE).
