---
title: AIkaBoOM
emoji: 💥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Build AI BOMs by aggregating, aligning, and resolving conflicting metadata.
---

# AIkaBoOM

Builds AI Bills of Materials by aggregating, aligning, and resolving
conflicting metadata across the AI supply chain.

## How to use this Space

1. Pick a BOM type: **AI** (model) or **Data** (dataset).
2. Paste any combination of HuggingFace, GitHub, and arXiv links.
3. Choose an LLM provider. For OpenRouter, click **🎯 Pick a free model**
   to use one of the free models from `/v1/models`.
4. Hit **Generate** and watch live logs in the **Logs** tab.
5. Inspect the **Conflicts** tab (red badge if any disagreement was found),
   then download the **Provenance BOM** and the **SPDX 3.0.1** export.

If a link is missing, the **Link Fallback Agent** (Gemini) tries to find it.
Disabled when `GEMINI_API_KEY` is not set in this Space's secrets.

## Required configuration (Space secrets)

Set at least one LLM provider key in **Settings → Variables and secrets**:

| Secret                | When you need it                                  |
|-----------------------|---------------------------------------------------|
| `OPENROUTER_API_KEY`  | Recommended. Free models available.               |
| `OPENAI_API_KEY`      | If you want to use OpenAI directly.               |
| `OLLAMA_BASE_URL`     | If you point at a remote Ollama server.           |
| `GITHUB_TOKEN`        | Optional. Higher GitHub API rate limit.           |
| `HUGGINGFACE_TOKEN`   | Optional. Required for gated/private HF models.   |
| `GEMINI_API_KEY`      | Optional. Enables the Link Fallback Agent.        |

These are exposed as environment variables inside the container at runtime.

## What runs inside this Space?

The Space itself does **not** host a large LLM. It runs:

- The Flask web UI
- A small local embedding model (`BAAI/bge-small-en-v1.5`, ~50 MB)
- HTTP clients that call out to whichever LLM provider you configured

This keeps the Space well within the free-tier 16 GB RAM / 8 GB image
limits and avoids cold-start costs of downloading a multi-billion-parameter
model.

## Source code & docs

- GitHub: <https://github.com/rgopikrishnan91/aibom>
- Full docs and CLI usage: see the README in the GitHub repo.

## License

MIT
