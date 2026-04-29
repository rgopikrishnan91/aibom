"""
OpenRouter model catalog helpers.

Fetches the public OpenRouter model list (https://openrouter.ai/api/v1/models),
caches it for 1 hour, and exposes filtering helpers for free models.

Public API:
    list_openrouter_models(force_refresh=False) -> list[dict]
    list_free_openrouter_models(force_refresh=False) -> list[dict]
    pick_free_openrouter_model() -> str
    CURATED_FREE_FALLBACK -> list[str]

Each returned model dict has at least: id, name, context_length, pricing.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_CACHE_TTL_SECONDS = 3600

# Module-level cache: {key: (timestamp, value)}
_cache: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}

# Hand-curated list of free models, used as a fallback when the live API is
# unreachable.  These have been stable for a while; if any disappear the
# function still returns the rest plus whatever the catalog had.
CURATED_FREE_FALLBACK: List[str] = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-r1:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]


def _curated_fallback_models() -> List[Dict[str, Any]]:
    """Return CURATED_FREE_FALLBACK shaped like normal model entries."""
    return [
        {
            "id": mid,
            "name": mid.split("/", 1)[-1].replace(":free", "") + " (free)",
            "context_length": None,
            "pricing": {"prompt": "0", "completion": "0"},
        }
        for mid in CURATED_FREE_FALLBACK
    ]


def _slim(model: Dict[str, Any]) -> Dict[str, Any]:
    """Pick only the fields we need for the UI/CLI."""
    return {
        "id": model.get("id"),
        "name": model.get("name") or model.get("id"),
        "context_length": model.get("context_length"),
        "pricing": model.get("pricing", {}),
    }


def _is_free(model: Dict[str, Any]) -> bool:
    """A model is free if it has a :free id suffix or zero pricing."""
    mid = model.get("id") or ""
    if mid.endswith(":free"):
        return True
    pricing = model.get("pricing") or {}
    # OpenRouter returns prices as decimal strings, e.g. "0" or "0.0000005".
    return pricing.get("prompt") == "0" and pricing.get("completion") == "0"


def list_openrouter_models(force_refresh: bool = False, *, timeout: int = 10) -> List[Dict[str, Any]]:
    """Fetch the full OpenRouter model catalog. Cached for 1 hour.

    Returns a list of slim dicts: {id, name, context_length, pricing}.
    On network error returns the curated fallback list (still slim-shaped).
    """
    cache_key = "all"
    now = time.monotonic()
    cached = _cache.get(cache_key)
    if cached and not force_refresh and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        resp = requests.get(OPENROUTER_MODELS_URL, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or []
        models = [_slim(m) for m in data if m.get("id")]
        _cache[cache_key] = (now, models)
        return models
    except Exception as exc:  # network, timeout, JSON, etc.
        print(f"  ⚠️ OpenRouter /models fetch failed ({exc}); using curated fallback")
        fallback = _curated_fallback_models()
        # Cache the fallback too, but with a short-ish TTL via timestamp.
        _cache[cache_key] = (now, fallback)
        return fallback


def list_free_openrouter_models(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Return only free models, sorted by context_length descending."""
    all_models = list_openrouter_models(force_refresh=force_refresh)
    free = [m for m in all_models if _is_free(m)]

    def _ctx_key(m: Dict[str, Any]) -> int:
        ctx = m.get("context_length")
        return ctx if isinstance(ctx, int) else 0

    free.sort(key=_ctx_key, reverse=True)
    return free


def pick_free_openrouter_model() -> str:
    """Return one free model id. Picks the highest-context model available.

    Used by ``bom-tools generate --pick-free-model`` and the equivalent
    Python API path.  Always returns a string (uses CURATED_FREE_FALLBACK[0]
    as the absolute last resort).
    """
    free = list_free_openrouter_models()
    if free and free[0].get("id"):
        return free[0]["id"]
    return CURATED_FREE_FALLBACK[0]
