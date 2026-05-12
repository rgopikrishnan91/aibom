"""
HuggingFace Inference Providers model catalog.

The HF "Inference Providers" router exposes an OpenAI-compatible
``/v1/chat/completions`` endpoint at ``https://router.huggingface.co/v1``,
fronting Together / Fireworks / Cerebras / Novita / Nebius / SambaNova
/ Hyperbolic / etc. behind a single HF token. The model catalog comes
from the Hub model index, sorted by downloads. Not every text-generation
model is currently warm on a provider — picking a cold one surfaces a
clear error from the router when Generate is clicked, which is
preferable to over-filtering here (the inference-provider filter on
/api/models has moved a few times).

Public API:
    list_huggingface_models(force_refresh=False) -> list[dict]
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import requests


HF_MODELS_URL = "https://huggingface.co/api/models"
_CACHE_TTL_SECONDS = 3600

_cache: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}


def _slim(model: Dict[str, Any]) -> Dict[str, Any]:
    """Pick only the fields we need for the UI."""
    return {
        "id": model.get("id") or model.get("modelId"),
        "name": model.get("id") or model.get("modelId"),
        # ``inference`` is "warm"/"cold"/None; warm = served right now.
        "inference": model.get("inference"),
        "downloads": model.get("downloads"),
        "likes": model.get("likes"),
        "pipeline_tag": model.get("pipeline_tag"),
    }


def list_huggingface_models(
    force_refresh: bool = False, *, timeout: int = 10, limit: int = 200,
) -> List[Dict[str, Any]]:
    """Fetch HF models served by an inference provider. Cached for 1h.

    Returns a list of slim dicts sorted by downloads desc. On network
    failure returns ``[]`` (caller surfaces a "couldn't fetch" message).
    """
    cache_key = "all"
    now = time.monotonic()
    cached = _cache.get(cache_key)
    if cached and not force_refresh and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    params = {
        # Only text-generation chat-capable models — others won't fit the
        # OpenAI chat-completions shape the rest of the app expects.
        "pipeline_tag": "text-generation",
        "sort": "downloads",
        "direction": -1,
        "limit": limit,
    }
    try:
        resp = requests.get(HF_MODELS_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json() or []
        models = [_slim(m) for m in data if (m.get("id") or m.get("modelId"))]
        _cache[cache_key] = (now, models)
        return models
    except Exception as exc:
        print(f"  ⚠️ HuggingFace /models fetch failed ({exc})")
        return []


__all__ = ["list_huggingface_models"]
