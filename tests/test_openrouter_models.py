"""
Tests for the OpenRouter model catalog helpers.

The helpers must:
- parse the /v1/models response shape
- correctly identify "free" models by either :free suffix or zero pricing
- cache for 1 hour and respect force_refresh
- fall back to a curated list on network failure
- never return None from pick_free_openrouter_model()
"""
import time
from unittest.mock import patch, MagicMock

import pytest

from aikaboom.utils import openrouter_models as orm


SAMPLE_RESPONSE = {
    "data": [
        {
            "id": "meta-llama/llama-3.3-70b-instruct:free",
            "name": "Llama 3.3 70B (free)",
            "context_length": 131072,
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "id": "meta-llama/llama-3.3-70b-instruct",
            "name": "Llama 3.3 70B",
            "context_length": 131072,
            "pricing": {"prompt": "0.0000005", "completion": "0.0000007"},
        },
        {
            "id": "openai/gpt-4o",
            "name": "GPT-4o",
            "context_length": 128000,
            "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
        },
        {
            "id": "google/gemini-2.0-flash-exp:free",
            "name": "Gemini 2.0 Flash Exp (free)",
            "context_length": 1000000,
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "id": "experimental/no-suffix-but-zero",
            "name": "Free without :free suffix",
            "context_length": 4096,
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ]
}


@pytest.fixture(autouse=True)
def clear_cache():
    """Reset module-level cache between tests."""
    orm._cache.clear()
    yield
    orm._cache.clear()


def _mock_response(status=200, json_data=None):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = json_data
    if status >= 400:
        from requests.exceptions import HTTPError
        m.raise_for_status.side_effect = HTTPError(f"{status}")
    else:
        m.raise_for_status.return_value = None
    return m


class TestListOpenrouterModels:

    def test_returns_slim_dicts(self):
        with patch("requests.get", return_value=_mock_response(json_data=SAMPLE_RESPONSE)):
            models = orm.list_openrouter_models()
        assert len(models) == 5
        first = models[0]
        # Slim shape: id, name, context_length, pricing
        assert set(first.keys()) >= {"id", "name", "context_length", "pricing"}

    def test_caches_within_ttl(self):
        with patch("requests.get", return_value=_mock_response(json_data=SAMPLE_RESPONSE)) as mock_get:
            orm.list_openrouter_models()
            orm.list_openrouter_models()
            assert mock_get.call_count == 1

    def test_force_refresh_re_fetches(self):
        with patch("requests.get", return_value=_mock_response(json_data=SAMPLE_RESPONSE)) as mock_get:
            orm.list_openrouter_models()
            orm.list_openrouter_models(force_refresh=True)
            assert mock_get.call_count == 2

    def test_network_failure_falls_back_to_curated(self):
        with patch("requests.get", side_effect=Exception("boom")):
            models = orm.list_openrouter_models()
        ids = {m["id"] for m in models}
        assert ids == set(orm.CURATED_FREE_FALLBACK)

    def test_403_falls_back_to_curated(self):
        with patch("requests.get", return_value=_mock_response(status=403)):
            models = orm.list_openrouter_models()
        # Curated fallback is used on any error including 403
        assert all(m["id"].endswith(":free") for m in models)


class TestListFreeOpenrouterModels:

    def test_filters_by_free_suffix(self):
        with patch("requests.get", return_value=_mock_response(json_data=SAMPLE_RESPONSE)):
            free = orm.list_free_openrouter_models()
        ids = {m["id"] for m in free}
        # Both :free models AND the "all-zero pricing" model should be returned
        assert "meta-llama/llama-3.3-70b-instruct:free" in ids
        assert "google/gemini-2.0-flash-exp:free" in ids
        assert "experimental/no-suffix-but-zero" in ids
        # Paid models excluded
        assert "openai/gpt-4o" not in ids
        assert "meta-llama/llama-3.3-70b-instruct" not in ids

    def test_sorted_by_context_length_desc(self):
        with patch("requests.get", return_value=_mock_response(json_data=SAMPLE_RESPONSE)):
            free = orm.list_free_openrouter_models()
        ctxs = [m["context_length"] for m in free]
        # Gemini 2.0 has 1M context, Llama 3.3 has 131K, experimental has 4K
        assert ctxs == sorted(ctxs, reverse=True)


class TestPickFreeOpenrouterModel:

    def test_returns_string(self):
        with patch("requests.get", return_value=_mock_response(json_data=SAMPLE_RESPONSE)):
            picked = orm.pick_free_openrouter_model()
        assert isinstance(picked, str)
        assert picked  # non-empty

    def test_picks_largest_context(self):
        with patch("requests.get", return_value=_mock_response(json_data=SAMPLE_RESPONSE)):
            picked = orm.pick_free_openrouter_model()
        # Gemini 2.0 has the largest context window in the sample
        assert picked == "google/gemini-2.0-flash-exp:free"

    def test_falls_back_when_offline(self):
        with patch("requests.get", side_effect=Exception("offline")):
            picked = orm.pick_free_openrouter_model()
        assert picked in orm.CURATED_FREE_FALLBACK


class TestPublicAPIReexports:

    def test_can_import_from_top_level(self):
        from aikaboom import (
            list_openrouter_models,
            list_free_openrouter_models,
            pick_free_openrouter_model,
        )
        # All three should be callable
        assert callable(list_openrouter_models)
        assert callable(list_free_openrouter_models)
        assert callable(pick_free_openrouter_model)
