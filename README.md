# BOM Tools (BOM_Tools)

Generate **AI BOMs** (for models) and **Data BOMs** (for datasets) from multiple evidence sources (Hugging Face, GitHub, arXiv) using either:
- **Direct** LLM extraction, or
- **RAG** (retrieval augmented generation) over fetched evidence.

The project also includes:
- **Conflict detection** and source-attributed field resolution
- Optional **Gemini-powered link fallback** (discover missing GitHub/arXiv/HF links)
- Optional **Gemini-powered Hugging Face relationship extraction** (model↔dataset lineage)
- **SPDX 3.0.1 JSON-LD conversion** utilities

This README is written for someone setting up the project on a new machine.

## Key capabilities

- **Multi-source evidence collection**: GitHub README, HF README/model card, arXiv PDF text
- **Two processing modes**:
  - **RAG**: chunk + embed + retrieve evidence, then ask the LLM
  - **Direct**: send full evidence to the LLM (with truncation safeguards)
- **Provider options**:
  - OpenAI (cloud)
  - Ollama (local, OpenAI-compatible API endpoint)
- **Field-level conflict handling**: store `value`, `source`, and `conflict` per field
- **Batch processing**: CSV-driven generation + checkpoint/resume

## Project structure

```
BOM_Tools/
├── src/bom_tools/           # Source code package
│   ├── core/                # Core processing logic
│   │   ├── agentic_rag.py   # RAG and Direct LLM engines
│   │   ├── processors.py    # AI/Data BOM processors
│   │   └── source_handler.py # Conflict resolution
│   ├── utils/               # Utility modules
│   │   ├── metadata_fetcher.py # API clients
│   │   ├── link_fallback.py    # Link discovery
│   │   └── spdx_validator.py   # SPDX conversion
│   └── web/                 # Web application
│       ├── app.py           # Flask application
│       ├── templates/       # HTML templates
│       └── static/          # CSS, JS, images
├── tests/                   # CLI scripts + unit tests
├── docs/                    # Documentation
├── examples/                # Example outputs (UI-style JSON)
├── scripts/                 # Setup & utility scripts
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── pyproject.toml          # Modern Python config
├── LICENSE                  # MIT License
└── README.md               # This file
```

## Requirements

### System
- Linux/macOS/Windows supported (Linux is the most tested)
- **Python**: recommended **3.10–3.12**
  - Python **3.13** can work, but some environments restrict installing packages globally (see “Troubleshooting: permission denied”).

### Accounts / API keys (depending on what you run)
- **GitHub**: `GITHUB_TOKEN` (recommended to avoid low rate limits)
- **Hugging Face**: `hug_token` or `HUGGINGFACE_TOKEN` (recommended)
- **OpenAI** (if using OpenAI LLMs, and also for embeddings in RAG): `OPENAI_API_KEY`
- **Gemini** (optional):
  - `GEMINI_API_KEY` for **link fallback** (`src/bom_tools/utils/link_fallback.py`)
  - `GEMINI_API_KEY` for **HF relationship extraction** (`src/bom_tools/core/HF_relations.py`)
- **Ollama** (optional): a running Ollama server + `OLLAMA_BASE_URL`

## Installation

### Option A (recommended): virtualenv in the project

```bash
cd BOM_Tools
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Option B: conda environment

```bash
cd BOM_Tools
conda create -n bom-tools python=3.11 -y
conda activate bom-tools
pip install -r requirements.txt
pip install -e .
```

## Configuration

Create a `.env` file in the project root (you already have one in this repo). Example:

```env
# Required for GitHub metadata (recommended)
GITHUB_TOKEN=...

# Required for Hugging Face metadata (recommended)
hug_token=...
# or
HUGGINGFACE_TOKEN=...

# OpenAI (required if using OpenAI LLMs; optional for embeddings - see below)
OPENAI_API_KEY=...

# Gemini (optional)
GEMINI_API_KEY=...
DEBUG_FALLBACK=false
DEBUG_HF_RELATIONS=false

# Ollama (optional; must be OpenAI-compatible base URL)
OLLAMA_BASE_URL=http://localhost:11434/v1/
```

Notes:
- **RAG mode now supports LOCAL embeddings** (default) using HuggingFace models - no OpenAI API key needed!
- **Local embeddings** use `sentence-transformers/all-MiniLM-L6-v2` by default (fast, free, runs on CPU)
- To use OpenAI embeddings instead, set `embedding_provider="openai"` when creating processors
- If you want "Ollama-only" runs with no OpenAI dependency, use **RAG mode with local embeddings** (default) or **Direct** mode

## Quick start

### 1) Web UI

```bash
python run.py
```

Then open `http://localhost:5000`.

The UI supports:
- AI BOM vs Data BOM
- RAG vs Direct mode
- OpenAI vs Ollama provider
- Optional link fallback (Gemini) when links are missing

### 2) Batch processing from CSV (recommended for experiments)

The repo includes a CSV-driven batch script that:
- reads `data/sources_real_200_fallback.csv` by default
- checks `type` column (`model` vs `dataset`)
- calls the correct processor (`AIBOMProcessor` or `DATABOMProcessor`)
- writes **UI-style BOM JSON** files into `results_200nodes/json/`
- saves a checkpoint after each row and **auto-resumes** on restart

```bash
python tests/test_200_nodes_processing.py --mode rag --provider openai
```

Useful flags:

```bash
# Process only first N rows (testing)
python tests/test_200_nodes_processing.py --limit 5

# Start from a specific row (1-based row number in the CSV)
python tests/test_200_nodes_processing.py --start-row 100

# Force restart from the beginning (ignore checkpoint)
python tests/test_200_nodes_processing.py --force-restart

# Use a different CSV file in data/
python tests/test_200_nodes_processing.py --input-csv sources_real_200_fallback.csv
```

Checkpoint file:
- `results_200nodes/json/intermediate_results.json`

UI-format individual output files:
- `results_200nodes/json/<id>_<provider>-<mode>_<use_case>_<model>_databom.json`
- `results_200nodes/json/<id>_<provider>-<mode>_<use_case>_<model>_aibom.json`

### 3) Link completion (only if your CSV is missing both links)

If you want to fill in missing links in-place for rows where **both** `github link` and `arxiv link` are empty:

```bash
python tests/run_link_fallback.py
```

This script:
- reads `data/sources_real_200_fallback.csv`
- calls Gemini only for rows with both links empty
- writes updates back into that CSV
- creates a timestamped backup first

### 4) HF relationship extraction (Gemini)

If you want to infer relationships like model→dataset (trained_on/finetuned_on) or dataset→model, see:
- `src/bom_tools/core/HF_relations.py`
- `tests/test_hf_relations.py`
- `docs/HF_RELATIONSHIP_FORMAT.md`

This is separate from BOM generation and is only used when you explicitly run it.

## Python API usage (library)

```python
from bom_tools.core.processors import AIBOMProcessor

# Initialize processor
processor = AIBOMProcessor(
    model="gpt-4o",
    mode="rag",
    llm_provider="openai",
    use_case="complete"
)

# Process an AI model
metadata = processor.process_ai_model(
    repo_id="microsoft/DialoGPT-medium",
    arxiv_url="https://arxiv.org/abs/1911.00536",
    github_url="https://github.com/microsoft/DialoGPT"
)

print(f"Generated BOM for: {metadata['model_id']}")
```

See `examples/` for example UI-style JSON outputs.

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=bom_tools --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Output formats

### UI-style BOM JSON (what the batch script writes)

Example file in this repo:
- `examples/boun-tabi_squad_tr_ollama-rag_complete_llama3_databom.json`

Fields are stored as triplets:

```json
"license": {
  "value": "odc-by",
  "conflict": "github: Apache License 2.0",
  "source": "huggingface"
}
```

### SPDX 3.0.1 JSON-LD

The SPDX conversion utility is implemented in:
- `src/bom_tools/utils/spdx_validator.py` (`SPDXValidator`)

If you plan to ship SPDX JSON-LD as the final artifact, integrate this validator into your export path (web or CLI) and run schema/structure checks as part of your pipeline.

## Documentation

- [Migration Guide](docs/migration.md) - Important information about changes and migration
- [HF Relationship Format](docs/HF_RELATIONSHIP_FORMAT.md) - Gemini-based relationship extraction output format

## Troubleshooting

### Permission denied when installing packages

If you see errors like:
`Permission denied: ... site-packages/...`

Use one of these options:
- Use a **virtualenv** in the project (`python -m venv .venv`)
- Use **conda** (`conda create -n bom-tools python=3.11`)
- As a last resort: `pip install --user -r requirements.txt`

### Embedding Options for RAG Mode

**Local Embeddings (Default - FREE, No API Key Required)**
- Uses HuggingFace models that run locally on your machine
- Default model: `sentence-transformers/all-MiniLM-L6-v2` (fast, accurate, CPU-friendly)
- No internet required after first download
- Completely free forever

**Alternative Local Models:**
- `sentence-transformers/all-mpnet-base-v2` - Higher quality, slightly slower
- `BAAI/bge-small-en-v1.5` - Good balance of speed and quality
- `BAAI/bge-base-en-v1.5` - Better quality, more resource intensive

**OpenAI Embeddings (Optional)**
To use OpenAI embeddings instead:
- Set `embedding_provider="openai"` when creating processors
- Requires `OPENAI_API_KEY` in your `.env` file
- Costs money per API call

**Example:**
```python
from bom_tools.core.processors import AIBOMProcessor

# Local embeddings (default, free)
processor = AIBOMProcessor(mode="rag", embedding_provider="local")

# Or use a different local model
processor = AIBOMProcessor(
    mode="rag", 
    embedding_provider="local",
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)

# Or use OpenAI embeddings
processor = AIBOMProcessor(mode="rag", embedding_provider="openai")
```

### RAG mode requires OpenAI embeddings

DEPRECATED: RAG mode now uses local embeddings by default (no OpenAI key needed).
If you want OpenAI embeddings, set `embedding_provider="openai"`.

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

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the BOM_Tools directory:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
GITHUB_TOKEN=your_github_token_here
hug_token=your_huggingface_token_here

# Optional API Keys
GEMINI_API_KEY=your_gemini_api_key_here  # For link fallback feature

# Optional Debug Flags
DEBUG_FALLBACK=false  # Set to 'true' for detailed link fallback logs
```

## Usage

### Starting the Application

```bash
# Start the web application using run.py
python run.py

# Or use the Python module command
python -m bom_tools.web.app
```

The web interface will be available at `http://localhost:5000`

### Processing an AI Model BOM

1. Select "AI Model" as the BOM Type
2. Enter the Model Repository ID (e.g., `google/gemma-2b`)
3. Optionally provide ArXiv paper URL and GitHub repository URL
4. Select processing mode (RAG or Direct)
5. Choose LLM provider and model
6. Select use case (Complete, Minimal, or Custom)
7. Click "Process BOM"

### Processing a Dataset BOM

1. Select "Dataset" as the BOM Type
2. Enter the Hugging Face Dataset URL
3. Optionally provide GitHub repository URL
4. Select processing mode (RAG or Direct)
5. Choose LLM provider and model
6. Select use case (Complete, Minimal, or Custom)
7. Click "Process BOM"

## Architecture

### AIBOMProcessor

Processes AI model metadata from:
- GitHub repositories (README, license, metadata)
- Hugging Face model cards (config, README)
- ArXiv papers (PDF text extraction)

Extracts fields like:
- Model name, type, version
- Training information and hyperparameters
- Performance metrics and limitations
- Safety assessments and standard compliance

### DATABOMProcessor

Processes dataset metadata from:
- GitHub repositories (README, license, metadata)
- Hugging Face datasets (config, README, card data)

Extracts fields like:
- Dataset name, size, type
- Data collection and preprocessing
- Known biases and intended use
- PII handling and confidentiality

### Unified RAG Core

- **AgenticRAG**: Uses LangGraph workflow with retrieval and generation nodes
- **DirectLLM**: Sends full content directly to LLM without chunking
- **Dynamic Questions**: Different question sets for AI models vs datasets
- **Source Aggregation**: Combines metadata from multiple sources

### SPDX Validator

Converts extracted BOM metadata to SPDX 3.0.1 format:
- **AI BOMs**: Uses AI Package profile
- **Dataset BOMs**: Uses Dataset Package profile
- Generates compliant JSON-LD with proper relationships

## API Endpoints

### POST /process

Process a BOM request.

**Request Body:**
```json
{
  "bom_type": "ai",  // or "data"
  "repo_id": "google/gemma-2b",  // for AI models
  "hf_url": "https://huggingface.co/datasets/...",  // for datasets
  "arxiv_url": "https://arxiv.org/abs/...",  // optional
  "github_url": "https://github.com/...",  // optional
  "mode": "rag",  // or "direct"
  "llm_provider": "openai",  // or "ollama"
  "model": "gpt-4o",
  "use_case": "complete"  // or "minimal"
}
```

**Response:**
```json
{
  "success": true,
  "repo_id": "google/gemma-2b",
  "direct_fields": { ... },
  "rag_fields": { ... },
  "spdx_output": { ... }
}
```

## Troubleshooting

### Missing Dependencies

If you see import errors, the application will continue to work with degraded functionality:
- **httpx not installed**: Link fallback will be disabled
- **google-genai not installed**: Link fallback will be disabled

### Network Issues

If you're behind a proxy or firewall:
1. The link fallback feature may not work (Gemini API connectivity issues)
2. The main BOM processing will continue normally
3. Set `DEBUG_FALLBACK=true` in `.env` for detailed error messages

### No Retrievers Created Error

This indicates that the sources (GitHub README, HuggingFace card, ArXiv paper) have no content:
- Verify the URLs are correct
- Check that the repositories/papers are publicly accessible
- Try using Direct LLM mode instead of RAG mode

## Development

### Adding New Questions

Edit the question dictionaries in `src/bom_tools/core/agentic_rag.py`:
- `FIXED_QUESTIONS_AI`: Questions for AI models
- `FIXED_QUESTIONS_DATA`: Questions for datasets

### Adding New Metadata Sources

1. Add fetching logic to `src/bom_tools/utils/metadata_fetcher.py`
2. Update `AIBOMProcessor` or `DATABOMProcessor` in `src/bom_tools/core/processors.py` to call the new fetcher
3. Update `src/bom_tools/core/source_handler.py` to handle conflicts

### Customizing SPDX Output

Edit `src/bom_tools/utils/spdx_validator.py`:
- Modify field mappings in `AI_FIELD_MAPPING` or `DATASET_FIELD_MAPPING`
- Adjust the SPDX structure in `_convert_ai_bom()` or `_convert_dataset_bom()`

## Known Limitations

1. **Kaggle Support Removed**: The DataBOM tool no longer supports Kaggle datasets
2. **Network Dependency**: Requires internet access for API calls and metadata fetching
3. **Link Fallback**: May fail in restricted network environments
4. **PDF Parsing**: ArXiv PDF extraction may miss complex formatting

## License

[Your License Here]

## Contributors

BOM Tools Development Team

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions and support, please open an issue on GitHub.

---

Made with ❤️ by the BOM Tools team
