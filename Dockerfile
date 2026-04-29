# Dockerfile for BOM Tools — designed for HuggingFace Spaces (Docker SDK)
# but also runs as a standalone container (docker build/run).
#
# HF Spaces conventions honored:
#   - Container runs as UID 1000 (non-root)
#   - App listens on 0.0.0.0:7860
#   - HF model cache lives under HF_HOME (writable by user)

FROM python:3.11-slim

# Build-time tools needed for some Python packages (faiss, torch wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — HF Spaces requires UID 1000
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/user/.cache/huggingface \
    BOM_HOST=0.0.0.0 \
    BOM_PORT=7860

WORKDIR $HOME/app

# Install dependencies first for better layer caching
COPY --chown=user:user pyproject.toml setup.py requirements.txt ./
COPY --chown=user:user src/ ./src/
RUN pip install --no-cache-dir --user -e .

# App code (run.py + templates + static)
COPY --chown=user:user run.py ./
COPY --chown=user:user .env.example ./

# Pre-download the small embedding model so first request is fast.
# This is best-effort — if it fails (offline build), the model will be
# downloaded at runtime instead.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')" || true

EXPOSE 7860

CMD ["python", "run.py"]
