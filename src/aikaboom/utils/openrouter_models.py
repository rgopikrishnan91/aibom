"""
OpenRouter model catalog helpers.

Fetches the public OpenRouter model list (https://openrouter.ai/api/v1/models),
caches it for 1 hour, and exposes a single helper for the CLI/web to
present the catalog.

Public API:
    list_openrouter_models(force_refresh=False) -> list[dict]

Each returned model dict has at least: id, name, context_length, pricing.

Phase 10 retired the free-model surface (picker, modality filter,
curated fallback list). Real-user testing showed the free-tier path
was a usability trap — rate limits made it unusable for non-trivial
runs, and the picker mislabeled non-chat models. OpenRouter as a
provider stays; users supply their own paid model id explicitly.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import requests


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_CACHE_TTL_SECONDS = 3600

# Module-level cache: {key: (timestamp, value)}
_cache: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}


def _slim(model: Dict[str, Any]) -> Dict[str, Any]:
    """Pick only the fields we need for the UI/CLI."""
    return {
        "id": model.get("id"),
        "name": model.get("name") or model.get("id"),
        "context_length": model.get("context_length"),
        "pricing": model.get("pricing", {}),
    }


def list_openrouter_models(force_refresh: bool = False, *, timeout: int = 10) -> List[Dict[str, Any]]:
    """Fetch the full OpenRouter model catalog. Cached for 1 hour.

    Returns a list of slim dicts: {id, name, context_length, pricing}.
    On network / parse failure returns an empty list (caller is expected
    to handle that — typically by surfacing a "couldn't fetch catalog"
    message and asking the user to provide a model id explicitly).
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
        print(f"  ⚠️ OpenRouter /models fetch failed ({exc})")
        return []
