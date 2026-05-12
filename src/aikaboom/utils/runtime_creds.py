"""Per-request credential overrides.

The HF Spaces deploy uses OAuth: each visitor signs in with their own
Hugging Face account and the resulting token is scoped to *their*
inference-providers quota (and billed against *their* HF account,
never the Space owner's). The token therefore changes per request,
so it can't live in ``os.environ`` — Flask handler threads share the
process environment and we'd race.

A ``contextvars.ContextVar`` is the right primitive: it's
per-thread/per-task, propagates into child threads/awaits inside the
same request, and falls back cleanly to the env var when nothing has
been set (CLI, self-hosted, OpenRouter path, etc.).
"""
from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Optional


_hf_token_var: ContextVar[Optional[str]] = ContextVar("hf_token", default=None)


def set_hf_token(token: Optional[str]) -> None:
    """Stash an HF token for the current request/thread context."""
    _hf_token_var.set(token or None)


def get_hf_token() -> Optional[str]:
    """Return the per-request HF token, falling back to ``HF_TOKEN`` env.

    Also honours the legacy ``HUGGINGFACE_TOKEN`` name used elsewhere
    in the codebase so a self-hosted user with that env var set still
    works without renaming.
    """
    token = _hf_token_var.get()
    if token:
        return token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


__all__ = ["set_hf_token", "get_hf_token"]
