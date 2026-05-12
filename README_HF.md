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
hf_oauth: true
hf_oauth_scopes:
  - openid
  - profile
  - inference-api
---

# AIkaBoOM

Builds AI Bills of Materials by aggregating, aligning, and resolving
conflicting metadata across the AI supply chain.

## How to use this Space

1. Click **Sign in with Hugging Face** at the top of the page. This Space
   uses HF OAuth so every LLM call is billed against *your* HF account
   and *your* free [inference credits](https://huggingface.co/docs/hub/rate-limits#billing-dashboard)
   — the Space owner does not provide an API key.
2. Pick a BOM type: **AI** (model) or **Data** (dataset).
3. Paste any combination of HuggingFace, GitHub, and arXiv links.
4. The provider defaults to **🤗 Hugging Face**. Click **📋 Load model
   catalog** to pull the live list of models served by HF Inference
   Providers (Together, Fireworks, Cerebras, Novita, …) and pick one,
   or paste any served model id.
5. Hit **Generate** and watch live logs in the **Logs** tab.
6. Inspect the **Conflicts** tab (red badge if any disagreement was found),
   then download the **Provenance BOM**, **SPDX 3.0.1**, and
   **CycloneDX 1.6 beta** exports.

On this public Space, recursive BOM walks are disabled and the SPDX
relationship list is capped at 10 children — both fan out a lot of
inference calls and would chew through a visitor's free credits very
quickly. For unbounded runs, clone the repo and self-host (see below).

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

## Required configuration

**Nothing.** The HF OAuth path means visitors bring their own token —
the Space needs no LLM secrets to demo. The Space owner can optionally
set the following in **Settings → Variables and secrets** to enable
extras:

| Secret                | When you need it                                              |
|-----------------------|---------------------------------------------------------------|
| `GITHUB_TOKEN`        | Optional. Higher GitHub API rate limit for source fetches.    |
| `GEMINI_API_KEY`      | Optional. Enables the Link Fallback Agent.                    |
| `OPENROUTER_API_KEY`  | Optional. Exposes the OpenRouter provider as a fallback.      |
| `OPENAI_API_KEY`      | Optional. Exposes the OpenAI provider as a fallback.          |

OAuth itself needs no manual setup: declaring `hf_oauth: true` in this
README's frontmatter makes HF inject `OAUTH_CLIENT_ID`,
`OAUTH_CLIENT_SECRET`, `OAUTH_SCOPES`, and `OPENID_PROVIDER_URL` into
the container at runtime; the Flask blueprint in
`src/aikaboom/web/hf_oauth.py` reads them.

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

1. After signing in with Hugging Face, click **📋 Load model catalog**
   in the Hugging Face section.
2. The browser hits `/models?provider=huggingface` on this Space.
3. The Space's backend fetches `https://huggingface.co/api/models`
   filtered to `inference_provider=all&pipeline_tag=text-generation` —
   i.e. chat-capable models served right now by at least one HF
   Inference Provider (Together, Fireworks, Cerebras, Novita, etc.).
4. The dropdown populates, sorted by downloads. Pick one, click **Generate**.
5. The Space forwards the chat call to
   `https://router.huggingface.co/v1/chat/completions` using your
   OAuth access token — which means usage is metered against your HF
   account.

If you haven't signed in yet, the **Generate** call returns a `401` and
the UI tells you to click **Sign in with Hugging Face**.

The model list is cached for 1 hour in memory.

## Source code & docs

- GitHub: <https://github.com/rgopikrishnan91/aikaboom>
- Full docs and CLI usage: see the README in the GitHub repo.

## License

MIT
