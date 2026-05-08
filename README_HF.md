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
3. Choose an LLM provider. For OpenRouter, click **📋 Load model
   catalog** to fetch the live list from `/v1/models` and pick a paid
   model id (e.g. `openai/gpt-4o-mini`, `meta-llama/llama-3.3-70b-instruct`).
4. Optionally cap the number of trainedOn / testedOn / dependsOn child
   packages emitted in the standalone SPDX via the **SPDX cap** input;
   leave empty for unbounded.
5. Hit **Generate** and watch live logs in the **Logs** tab.
6. Inspect the **Conflicts** tab (red badge if any disagreement was found),
   then download the **Provenance BOM**, **SPDX 3.0.1**, and
   **CycloneDX 1.6 beta** exports.

If a link is missing, the **Link Fallback Agent** (Gemini) tries to find it.
Disabled when `GEMINI_API_KEY` is not set in this Space's secrets.

SPDX exports are validated by default against the official bundled SPDX 3.0.1
JSON Schema. The SPDX tab shows pass/fail status and concise errors while still
letting you inspect/download the generated JSON-LD. Enable **Deep SHACL
validation (beta)** only for final checks; it uses the official SPDX SHACL
shapes and is slower on free CPU Spaces.

CycloneDX 1.6 ML-BOM export, recursive BOM generation, and strict SHACL
validation are beta in the Space UI. Recursive BOM generation walks the
dependency tree — each `trainedOn` / `testedOn` / `dependsOn` target
produces another BOM, the walk stops at the configured depth or when the
unique-target set is exhausted, and any field with a detected conflict is
skipped. With recursion on, the UI also surfaces a single
**Linked SPDX Beta** download that merges the parent and every child into
one SPDX 3.0.1 JSON-LD `@graph` (validated by both the JSON Schema and
SHACL passes). AIkaBoOM also auto-extracts model lineage hints from
HuggingFace's `cardData.base_model`, `cardData.datasets`, `model-index`,
and repository tags (e.g. `dataset:squad`, `base_model:...`) so they
participate in cross-source conflict detection.

## Required configuration (Space secrets)

Set at least one LLM provider key in **Settings → Variables and secrets**:

| Secret                | When you need it                                  |
|-----------------------|---------------------------------------------------|
| `OPENROUTER_API_KEY`  | Recommended. Single key for many hosted models.   |
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
- Bundled SPDX 3.0.1 JSON Schema and SHACL validation artifacts
- Beta CycloneDX 1.6 ML-BOM and recursive BOM export helpers

This keeps the Space well within the free-tier 16 GB RAM / 8 GB image
limits and avoids cold-start costs of downloading a multi-billion-parameter
model.

## Choosing a model — how it works on this Space

1. Click **📋 Load model catalog** in the OpenRouter section.
2. The browser hits `/models?provider=openrouter` on this Space.
3. The Space's backend fetches `https://openrouter.ai/api/v1/models`
   (public, unauthenticated) and returns the full list, sorted by
   context window.
4. The dropdown populates. Pick one, click **Generate**.

**Important:** running a model requires `OPENROUTER_API_KEY` set in the
Space's secrets and a credited account — Phase 10 retired the free-tier
path because rate-limit caps made it unusable for any non-trivial run.

If the key is missing, the **Generate** call will surface a `401` error
in the **Logs** tab.

The model list is cached for 1 hour in memory.

## Source code & docs

- GitHub: <https://github.com/rgopikrishnan91/aikaboom>
- Full docs and CLI usage: see the README in the GitHub repo.

## License

MIT
