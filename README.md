# BOM Tools (BOM_Tools)

Generate AI BOMs (for models) and Data BOMs (for datasets) from evidence sources such as Hugging Face, GitHub, and arXiv.

This repository currently supports both RAG and Direct extraction flows, multiple LLM providers, conflict-aware output fields, and a Flask web UI.

## Features (Current)

- AI BOM and Data BOM generation
- RAG mode and Direct mode
- LLM providers: OpenAI, Ollama, OpenRouter
- Local embeddings by default in RAG mode (no OpenAI key required for embeddings)
- Configurable use-case presets: complete, safety, security, lineage, license
- Field-level output triplets: value, source, conflict
- Optional Gemini-based link fallback when identifiers/links are missing
- SPDX validation/conversion utilities

## Changes Reflected In This README

Added/updated:
- OpenRouter provider support
- Local embedding-first workflow for RAG
- Use-case preset support in the web/API flow
- Dependency notes now include the currently used runtime stack (including chromadb, numpy, pandas, langchain-core, and langchain-text-splitters)

Removed/deprecated references:
- CSV batch scripts and helper scripts that are not present in this repository
- Hugging Face relationship extraction files that are not present in this repository

## Project Structure

```text
BOM_Tools/
├── src/bom_tools/
│   ├── core/
│   │   ├── agentic_rag.py
│   │   ├── processors.py
│   │   ├── source_handler.py
│   │   └── internal_conflict.py
│   ├── utils/
│   │   ├── metadata_fetcher.py
│   │   ├── link_fallback.py
│   │   └── spdx_validator.py
│   └── web/
│       ├── app.py
│       ├── templates/
│       └── static/
├── examples/
│   ├── example_ai_bom.py
│   ├── example_data_bom.py
│   └── sample-ouput.json
├── tests/
│   ├── test_processors.py
│   └── test_link_fallback.py
├── docs/
│   ├── README.md
│   ├── migration.md
│   └── LOCAL_EMBEDDINGS.md
├── requirements.txt
├── pyproject.toml
├── setup.py
├── run.py
└── README.md
```

## Requirements

- Python 3.8+
- Linux/macOS/Windows (Linux is most tested)

Optional credentials depending on provider/features:
- OpenAI: OPENAI_API_KEY
- OpenRouter: OPENROUTER_API_KEY
- Gemini link fallback: GEMINI_API_KEY
- GitHub API access (recommended): GITHUB_TOKEN
- Hugging Face access (recommended): hug_token or HUGGINGFACE_TOKEN

## Installation

### Virtual environment

```bash
cd BOM_Tools
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Conda environment

```bash
cd BOM_Tools
conda create -n bom-tools python=3.11 -y
conda activate bom-tools
pip install -r requirements.txt
pip install -e .
```

## Configuration

Create a .env file in project root if you need cloud providers or fallback services.

```env
OPENAI_API_KEY=
OPENROUTER_API_KEY=
GEMINI_API_KEY=
GITHUB_TOKEN=
hug_token=
HUGGINGFACE_TOKEN=
OLLAMA_BASE_URL=http://localhost:11434/v1/
```

Notes:
- RAG uses local Hugging Face embeddings by default.
- You can still switch to OpenAI embeddings by setting embedding_provider="openai" in processor initialization.

## Quick Start

For the fastest setup path, see [QUICKSTART.md](QUICKSTART.md).

## Run The Web App

```bash
python run.py
```

Then open http://localhost:5000.

Web UI supports:
- BOM type: AI or Data
- Mode: rag or direct
- Provider: openai, ollama, openrouter
- Optional link completion endpoint via Gemini fallback

## Python API Example

```python
from bom_tools.core.processors import AIBOMProcessor

processor = AIBOMProcessor(
    model="gpt-4o",
    mode="rag",
    llm_provider="openai",
    use_case="complete",
    embedding_provider="local",
    embedding_model="BAAI/bge-small-en-v1.5"
)

metadata = processor.process_ai_model(
    repo_id="microsoft/DialoGPT-medium",
    arxiv_url="https://arxiv.org/abs/1911.00536",
    github_url="https://github.com/microsoft/DialoGPT"
)

print(metadata.get("model_id"))
```

More runnable examples:
- [examples/example_ai_bom.py](examples/example_ai_bom.py)
- [examples/example_data_bom.py](examples/example_data_bom.py)

## Testing

```bash
pytest
pytest --cov=bom_tools --cov-report=html
```

## Output Shape

BOM fields use triplets:

```json
"license": {
  "value": "odc-by",
  "source": "huggingface",
  "conflict": {
    "value": "Apache License 2.0",
    "source": "github",
    "type": "inter"
  }
}
```

## Docs

- [docs/README.md](docs/README.md)
- [docs/LOCAL_EMBEDDINGS.md](docs/LOCAL_EMBEDDINGS.md)
- [docs/migration.md](docs/migration.md)

## Troubleshooting

If package installation fails due to permissions, use a virtual environment or conda environment rather than system Python.

### Ollama connection issues

Ensure Ollama is running and reachable:

```bash
ollama serve
curl http://localhost:11434/api/tags
```

Then set:
`OLLAMA_BASE_URL=http://localhost:11434/v1/`

### GitHub / Hugging Face rate limits

Set tokens in `.env`:
- `GITHUB_TOKEN`
- `hug_token` (or `HUGGINGFACE_TOKEN`)

### arXiv PDF parsing issues

arXiv PDFs are parsed via PyMuPDF. Some papers with complex formatting may yield imperfect text. If a field is consistently “Not found”, try:
- adding GitHub/HF sources
- using Direct mode
- manually reviewing evidence chunks retrieved in RAG mode

## License

MIT License. See `LICENSE`.
