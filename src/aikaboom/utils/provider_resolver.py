"""Provider auto-detection shared by CLI + web.

The CLI's ``_resolve_provider_and_model`` is coupled to ``argparse``
namespaces; the web ``/config`` route needs the same priority logic
without going through CLI args. This module exposes a stateless
``detect_default_provider()`` that both call (Phase 9 / Finding #11).
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple


# Priority order — first provider with a configured env var wins. Mirrors the
# behaviour of ``aikaboom.cli._resolve_provider_and_model``: prefer
# OpenRouter (free tier available) and Ollama (local) over OpenAI when more
# than one is configured.
PROVIDER_PRIORITY: List[Tuple[str, Optional[str], str]] = [
    ("openrouter", "OPENROUTER_API_KEY", "qwen/qwen-2.5-72b-instruct:free"),
    ("ollama",     "OLLAMA_BASE_URL",    "llama3:8b"),
    ("openai",     "OPENAI_API_KEY",     "gpt-4o"),
]


def detect_default_provider() -> Tuple[Optional[str], Optional[str]]:
    """Pick the (provider, default_model) the user is most likely set up for.

    Returns ``(None, None)`` when no provider env var is set so the caller
    can fall back to a sensible UI default rather than silently picking
    one. The web UI hits this on every ``/config`` request so the form
    starts on the user's actually-configured provider.
    """
    for provider, env_var, default_model in PROVIDER_PRIORITY:
        if env_var and os.getenv(env_var):
            return provider, default_model
    return None, None


def available_providers() -> List[str]:
    """List of provider names whose env var is currently set."""
    return [
        provider
        for provider, env_var, _default in PROVIDER_PRIORITY
        if env_var and os.getenv(env_var)
    ]


__all__ = ["detect_default_provider", "available_providers", "PROVIDER_PRIORITY"]
