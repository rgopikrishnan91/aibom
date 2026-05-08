"""Tests for the /models REST endpoint.

Phase 10 retired the ``free_only`` query parameter and the curated
free-model fallback; the endpoint now always returns the full
catalogue (or ``[]`` on network failure).
"""
import pytest
from unittest.mock import patch

from aikaboom.utils import openrouter_models as orm


_SAMPLE = {
    "data": [
        {
            "id": "openai/gpt-4o-mini",
            "name": "GPT-4o mini",
            "context_length": 128000,
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
        },
        {
            "id": "meta-llama/llama-3.3-70b-instruct",
            "name": "Llama 3.3 70B",
            "context_length": 131072,
            "pricing": {"prompt": "0.0000002", "completion": "0.0000002"},
        },
    ]
}


@pytest.fixture
def client():
    from aikaboom.web.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def clear_cache():
    orm._cache.clear()
    yield
    orm._cache.clear()


def _ok_response():
    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return _SAMPLE
    return R()


class TestModelsEndpoint:

    def test_default_returns_full_catalog(self, client):
        with patch("requests.get", return_value=_ok_response()):
            resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["provider"] == "openrouter"
        assert isinstance(data["models"], list)
        assert len(data["models"]) == 2

    def test_network_failure_returns_empty_list(self, client):
        """Phase 10: no curated fallback; the endpoint surfaces
        ``models: []`` so the UI can show a "couldn't fetch" hint."""
        with patch("requests.get", side_effect=Exception("offline")):
            resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["models"] == []

    def test_unknown_provider_returns_empty(self, client):
        resp = client.get("/models?provider=openai")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["models"] == []

    def test_response_shape(self, client):
        with patch("requests.get", return_value=_ok_response()):
            resp = client.get("/models")
        data = resp.get_json()
        assert "provider" in data
        assert "models" in data
        for m in data["models"]:
            # Slim shape: id, name, context_length, pricing
            assert "id" in m
            assert "name" in m
            assert "pricing" in m

    def test_free_only_param_is_ignored_in_phase10(self, client):
        """The ``free_only`` query param was retired in Phase 10. Sending
        it must not error — the endpoint just returns the full catalog
        like any other call."""
        with patch("requests.get", return_value=_ok_response()):
            resp = client.get("/models?free_only=true")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["models"]) == 2  # full list, not filtered
