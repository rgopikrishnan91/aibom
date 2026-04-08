# Quick Start (Lowest Effort)

This guide is the fastest path to run BOM Tools with minimal setup and minimal configuration.

## 1) Open the project

```bash
cd BOM_Tools
```

## 2) Use your conda environment

If your environment already exists:

```bash
conda activate VENV
```

If not, create it once:

```bash
conda create -n VENV python=3.11 -y
conda activate VENV
```

## 3) Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## 4) Start the app

```bash
python run.py
```

Open:
- http://localhost:5000

## 5) Minimal no-key usage (recommended)

In the UI, choose:
- Mode: rag
- Provider: ollama

Why this is lowest effort:
- RAG uses local embeddings by default
- No OpenAI key is required for embeddings
- You can run fully local if Ollama is available

If needed, start/pull Ollama model:

```bash
ollama serve
ollama pull llama3:8b
```

Use this base URL in UI/settings if prompted:

```text
http://localhost:11434/v1/
```

## Optional .env (only if you use cloud/fallback features)

Create a .env file in repo root when needed:

```env
OPENAI_API_KEY=
OPENROUTER_API_KEY=
GEMINI_API_KEY=
GITHUB_TOKEN=
HUGGINGFACE_TOKEN=
OLLAMA_BASE_URL=http://localhost:11434/v1/
```

## Quick API run (without web UI)

```bash
python examples/example_ai_bom.py
python examples/example_data_bom.py
```

## Common issues

- Import errors after install:
  - Re-run `pip install -r requirements.txt`
  - Confirm `conda activate VENV`
- Port busy on 5000:
  - Stop the process using that port, then rerun `python run.py`
- Ollama not reachable:
  - Make sure `ollama serve` is running
  - Check URL is `http://localhost:11434/v1/`
