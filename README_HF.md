---
title: BOM Tools
emoji: 📦
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Generate SPDX 3.0.1 BOMs for AI models and datasets.
---

# BOM Tools on HuggingFace Spaces

Generate Software Bills of Materials (SBOMs) for AI models and datasets, with
source-level conflict detection and SPDX 3.0.1 export.

## How to use this Space

1. Pick a BOM type: **AI** (model) or **Data** (dataset)
2. Paste any combination of HuggingFace, GitHub, and arXiv links
3. Choose an LLM provider (OpenAI / OpenRouter / Ollama). For OpenRouter,
   click **🎯 Pick a free model** to use one of the free OpenRouter models
4. Hit **Generate** and watch the live logs stream in the Logs tab
5. Download the **Provenance BOM** (JSON with conflict triplets) and the
   **SPDX 3.0.1** standards-compliant export

If a link is missing the **Link Fallback Agent** (Gemini) will try to find
it. The agent is disabled if `GEMINI_API_KEY` is not set in this Space's
secrets.

## Required configuration (Space secrets)

Set at least one LLM provider key in **Settings → Variables and secrets**:

- `OPENAI_API_KEY` — OpenAI
- `OPENROUTER_API_KEY` — OpenRouter (free models available, recommended)
- `OLLAMA_BASE_URL` — for a remote Ollama server

Optional secrets:

- `GITHUB_TOKEN` — increases GitHub API rate limits
- `HUGGINGFACE_TOKEN` — needed for gated/private models
- `GEMINI_API_KEY` — enables the Link Fallback Agent

## Source code & docs

- GitHub: <https://github.com/rgopikrishnan91/aibom>
- Full docs and CLI usage: see the README in the GitHub repo.

## License

MIT
