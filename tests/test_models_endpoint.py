"""Tests for the /models REST endpoint."""
import pytest
from unittest.mock import patch

from aikaboom.utils import openrouter_models as orm


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


class TestModelsEndpoint:

    def test_default_returns_openrouter_all(self, client):
        # Force the helper to return the curated fallback (no real network)
        with patch("requests.get", side_effect=Exception("offline")):
            resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["provider"] == "openrouter"
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0

    def test_free_only_filters(self, client):
        with patch("requests.get", side_effect=Exception("offline")):
            resp = client.get("/models?free_only=true")
        data = resp.get_json()
        # All curated fallback entries are :free, so any returned ids must be :free
        for m in data["models"]:
            assert m["id"].endswith(":free")

    def test_unknown_provider_returns_empty(self, client):
        resp = client.get("/models?provider=openai")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["models"] == []

    def test_response_shape(self, client):
        with patch("requests.get", side_effect=Exception("offline")):
            resp = client.get("/models?free_only=true")
        data = resp.get_json()
        assert "provider" in data
        assert "models" in data
        for m in data["models"]:
            # Slim shape: id, name, context_length, pricing
            assert "id" in m
            assert "name" in m
            assert "pricing" in m
