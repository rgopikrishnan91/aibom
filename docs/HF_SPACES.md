# Deploy to HuggingFace Spaces

This guide walks through deploying the BOM Tools web app to a free
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
3. Space name: e.g. `aibom`
4. License: MIT
5. Space SDK: **Docker** -> **Blank**
6. Hardware: `cpu-basic` (free) is enough for the embedding model and Flask app
7. Click **Create Space**

You will end up with a URL like
`https://huggingface.co/spaces/<you>/aibom`.

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
git remote add hf https://huggingface.co/spaces/<you>/aibom

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

Open the Space URL. You should see the BOM Tools web UI. Try generating
a BOM for `microsoft/DialoGPT-medium` to confirm everything works.

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
