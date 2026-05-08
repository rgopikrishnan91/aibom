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

    def test_network_failure_returns_empty(self):
        """Phase 10 retired the curated free-model fallback (Findings #6,
        #12). The catalog now returns ``[]`` on network failure; callers
        decide how to surface the missing data."""
        with patch("requests.get", side_effect=Exception("boom")):
            models = orm.list_openrouter_models()
        assert models == []

    def test_403_returns_empty(self):
        with patch("requests.get", return_value=_mock_response(status=403)):
            models = orm.list_openrouter_models()
        assert models == []


# Phase 10 retired the free-model surface entirely
# (TestListFreeOpenrouterModels / TestPickFreeOpenrouterModel /
# CURATED_FREE_FALLBACK). The free-tier path was a usability trap;
# tests/test_phase10_fixes.py now guards the removal with importability
# regression tests.


class TestPublicAPIReexports:

    def test_can_import_list_from_top_level(self):
        from aikaboom import list_openrouter_models
        assert callable(list_openrouter_models)

    def test_free_helpers_are_not_re_exported(self):
        """Regression for Phase 10: re-exporting the deleted helpers
        from the top-level package would silently break downstream
        consumers that rely on the new shape."""
        import aikaboom
        assert not hasattr(aikaboom, "list_free_openrouter_models")
        assert not hasattr(aikaboom, "pick_free_openrouter_model")
