# Migration Guide: Switching to Local Embeddings

## Summary
Your BOM Tools project now uses **FREE local embeddings** by default instead of requiring OpenAI API credentials!

## What Changed
- **✅ Default now:** Local HuggingFace embeddings (completely free, no API key needed)
- **❌ Old default:** OpenAI embeddings (required OPENAI_API_KEY, costs money)

## Benefits of Local Embeddings
- 🆓 **100% Free** - No API costs ever
- 🔒 **Privacy** - All processing happens on your machine
- 🚀 **Fast** - No network calls for embeddings
- 💻 **Works Offline** - After first model download

## Installation
Install the new dependencies:
```bash
pip install -r requirements.txt
# or
pip install sentence-transformers langchain-huggingface
```

## Usage

### Default (Local Embeddings - Recommended)
```python
from aikaboom.core.processors import AIBOMProcessor, DATABOMProcessor

# Uses local embeddings automatically - no API key needed!
processor = AIBOMProcessor(mode="rag")

# Or explicitly specify
processor = AIBOMProcessor(
    mode="rag",
    embedding_provider="local",  # default
    embedding_model="BAAI/bge-small-en-v1.5"  # default
)
```

### Alternative Local Models
Choose different models based on your needs:

**Fast and Efficient (Default)**
```python
embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # ~80MB, very fast
```

**Higher Quality**
```python
embedding_model="sentence-transformers/all-mpnet-base-v2"  # ~420MB, more accurate
```

**Good Balance**
```python
embedding_model="BAAI/bge-small-en-v1.5"  # ~130MB, good quality
embedding_model="BAAI/bge-base-en-v1.5"   # ~440MB, better quality
```

### OpenAI Embeddings (Optional)
Only use this if you specifically need OpenAI embeddings:
```python
processor = AIBOMProcessor(
    mode="rag",
    embedding_provider="openai"  # Requires OPENAI_API_KEY in .env
)
```

## Web Interface
The web interface (`python run.py`) automatically uses local embeddings - no configuration needed!

## Environment Variables
You can now remove `OPENAI_API_KEY` from your `.env` file if you're only using local embeddings and not OpenAI LLMs.

**Still Required:**
- `GITHUB_TOKEN` - For GitHub API access
- `hug_token` or `HUGGINGFACE_TOKEN` - For Hugging Face metadata

**Optional:**
- `OPENAI_API_KEY` - Only if using OpenAI LLMs or OpenAI embeddings
- `GEMINI_API_KEY` - Only if using link fallback feature
- `OLLAMA_BASE_URL` - Only if using Ollama

## First-Time Model Download
The first time you run with local embeddings, the model will be downloaded (80-440MB depending on model choice). This is a one-time download:

```
✓ Using LOCAL embeddings: sentence-transformers/all-MiniLM-L6-v2
Downloading model files... (this happens once)
```

Models are cached in: `~/.cache/huggingface/hub/`

## Performance
- **Local embeddings:** 100-1000 tokens/sec (CPU), 5000+ tokens/sec (GPU)
- **OpenAI embeddings:** Limited by API rate limits + network latency

## GPU Support (Optional)
To use GPU for faster embeddings (if you have CUDA):
```python
from aikaboom.core.agentic_rag import AgenticRAG

# Edit agentic_rag.py line ~478 to change 'cpu' to 'cuda':
self.embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs={'device': 'cuda'},  # Change from 'cpu' to 'cuda'
    encode_kwargs={'normalize_embeddings': True}
)
```

## Troubleshooting

### Import Error
If you see: `ImportError: cannot import name 'HuggingFaceEmbeddings'`
```bash
pip install --upgrade langchain-huggingface sentence-transformers
```

### Out of Memory
If you get OOM errors, use a smaller model:
```python
embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # Smallest, fastest
```

### Slow First Run
First run downloads the model (~80-440MB). Subsequent runs are instant.

## Questions?
- Check [README.md](README.md) section "Embedding Options for RAG Mode"
- All code defaults to local embeddings - just run it!
