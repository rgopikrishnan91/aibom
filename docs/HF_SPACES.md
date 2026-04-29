# Deploy to HuggingFace Spaces

This guide walks through deploying the AIkaBoOM web app to a free
HuggingFace Space using the Docker SDK. End users will then be able to
generate BOMs from a public URL without installing anything.

## Prerequisites

- A HuggingFace account
- `git` and a HuggingFace access token with **write** permission
  (Settings -> Access Tokens on huggingface.co)
- Optionally: an OpenRouter API key (free tier works)

## 1. Create the Space

1. Go to <https://huggingface.co/new-space>
2. Owner: your account or org
3. Space name: e.g. `aikaboom`
4. License: MIT
5. Space SDK: **Docker** -> **Blank**
6. Hardware: `cpu-basic` (free) is enough for the embedding model and Flask app
7. Click **Create Space**

You will end up with a URL like
`https://huggingface.co/spaces/<you>/aikaboom`.

## 2. Configure secrets

In the Space's **Settings -> Variables and secrets**, add at least one of:

| Secret name             | When you need it                                 |
|-------------------------|--------------------------------------------------|
| `OPENROUTER_API_KEY`    | Recommended. Free models available.              |
| `OPENAI_API_KEY`        | If you want to use OpenAI                        |
| `OLLAMA_BASE_URL`       | If you want to point at a remote Ollama server   |
| `GITHUB_TOKEN`          | Optional. Higher GitHub API rate limit.          |
| `HUGGINGFACE_TOKEN`     | Optional. Required for gated/private models.     |
| `GEMINI_API_KEY`        | Optional. Enables the Link Fallback Agent.       |

These show up as environment variables inside the container.

## 3. Push the code

From a clone of this GitHub repo:

```bash
# Add the HF Space as a git remote
git remote add hf https://huggingface.co/spaces/<you>/aikaboom

# Use the HF-flavored README (which has the YAML frontmatter Spaces needs)
bash scripts/deploy_to_hf_spaces.sh
```

The script:
1. Verifies the `hf` remote exists.
2. Stages a copy of `README_HF.md` as `README.md` in a temporary commit
   (so the GitHub README stays unchanged).
3. Pushes to `hf main`.
4. Resets the working copy back to its original state.

The first build takes ~5 minutes (downloading torch, sentence-transformers,
and faiss). Subsequent builds are faster thanks to Docker layer caching.

## 4. Use the Space

Open the Space URL. You should see the AIkaBoOM web UI. Try generating
a BOM for `microsoft/DialoGPT-medium` to confirm everything works.

SPDX 3.0.1 exports are validated automatically with the official bundled
JSON Schema. The UI shows the validation status in the SPDX tab and keeps the
export downloadable even if validation reports errors. The **Deep SHACL
validation (beta)** checkbox runs the official SPDX SHACL shapes after JSON
Schema; leave it off for normal free-tier use and enable it for slower final
checks. CycloneDX 1.7 export and recursive BOM generation are also beta in the
Space UI. Recursive BOM generation walks the dependency tree of an AI BOM:
each `trainedOn` / `testedOn` / `dependsOn` target produces another BOM, the
walk stops at the configured depth (capped at 5 in the Space UI) or when the
unique-target set is exhausted, and any field flagged with a conflict is
skipped. When recursion is on, the UI offers a **Linked SPDX Beta** download
that merges the parent and every recursive child into a single SPDX 3.0.1
JSON-LD document with explicit Relationship edges, validated by both the
JSON Schema and (when **Deep SHACL validation** is on) the SHACL shapes.

## 5. Choosing a model on the Space

The free-models picker works exactly the same on a Space as it does
locally. Concretely:

1. The user opens `https://huggingface.co/spaces/<you>/aikaboom`.
2. They pick **OpenRouter** as the provider.
3. They click **🎯 Pick a free model**.
4. The browser calls `/models?provider=openrouter&free_only=true` on the
   Space.
5. The Space's Flask backend fetches
   `https://openrouter.ai/api/v1/models` (public endpoint, no auth needed
   for listing) and returns the filtered free list.
6. The dropdown populates with free models sorted by context window. The
   user picks one.
7. They click **Generate**; the BOM is built using their selected model.

### Important nuance

Listing free models is unauthenticated. **Actually running** any of them
still requires a valid `OPENROUTER_API_KEY` set in the Space's
**Settings -> Variables and secrets**. OpenRouter charges $0 for `:free`
models but enforces account-level rate limits (~50 requests/day without
credits, ~1000/day after purchasing $10+ in credits).

If the user has not set the key yet:

- The picker will still list models (it calls a public endpoint).
- The actual Generate call will fail with a clear `401` / "no API key"
  error from OpenRouter, surfaced in the **Logs** tab of the UI.

### Caching

The Space caches the model list for 1 hour in memory. The first user to
click "Pick a free model" triggers a live fetch (~200 ms); everyone after
that gets it instantly until the cache expires or the Space restarts.

### What if OpenRouter is unreachable from the Space?

A curated fallback of 5 known-free models is returned, and the UI hint
shows "Loaded N free models". The picker degrades gracefully and never
shows an empty dropdown.

So nothing about this flow changes when you deploy. The only
Space-specific config is setting `OPENROUTER_API_KEY` in secrets.

## Known limitations on the free tier

- **Ephemeral storage.** The container's filesystem is wiped on restart and
  on the periodic 2-day idle sleep. Generated BOMs in `results/` are not
  persisted. Download anything you want to keep.
- **16 GB RAM.** Plenty for the small embedding model
  (`BAAI/bge-small-en-v1.5`) and Flask, but not for running an Ollama LLM
  inside the same container.
- **Cold starts.** When the Space wakes from sleep, the embedding model
  re-downloads (~50 MB) before serving the first request.
- **8 GB image cap.** Our Dockerfile uses `python:3.11-slim` plus
  `--user` pip installs, which keeps the image well under the cap.
- **Public visibility.** Anyone with the URL can use the Space and consume
  your API quotas. Keep it private if that is a concern.

## Upgrading to persistent storage (paid)

If you want generated BOMs to survive restarts, enable persistent storage in
the Space settings (starts at $5/month for 20 GB). The mount point is
`/data` inside the container. You can then:

- Set `BOM_RESULTS_DIR=/data/results` (not currently a documented env var,
  but you can edit `app.py` to honor it), OR
- Bind-mount `/data/results` to the existing results directory.

## Troubleshooting

**Build fails with "no space left on device"**
The 8 GB image cap was exceeded. Look at the Dockerfile and remove any
unnecessary system packages, or switch from `python:3.11-slim` to a smaller
base. Most often this means a non-`--user` pip install duplicating wheels;
double-check `--user` is set on the `pip install` lines.

**App starts but UI is unreachable**
The container must bind to `0.0.0.0:7860`. Our Dockerfile sets `BOM_HOST`
and `BOM_PORT` env vars; do not override them in Space settings to anything
other than `7860`.

**"Link Fallback Agent inactive"**
You did not set `GEMINI_API_KEY` in Space secrets. This is optional --
the app still works, you just have to fill in arXiv/GitHub links manually.

**OpenRouter says "401 Unauthorized" on free models**
Even free models require a valid `OPENROUTER_API_KEY` (the free key has
quotas, just no charges). Sign up at <https://openrouter.ai>.
