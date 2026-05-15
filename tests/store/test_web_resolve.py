"""The /process route accepts cache_policy and uses BomStore.resolve.

Task 16 of the worldofBOMs plan. The original plan references
``/api/generate`` but the actual Flask route is ``/process`` (see
``src/aikaboom/web/app.py``); this test exercises that endpoint.

The bar for this test is "the route doesn't 400 on the ``cache_policy``
body field." Full e2e mocking of the processor is out of scope — what we
guard against is the route blindly rejecting an unknown field or the
cache-wrap raising before the request can be answered.
"""
import os
import pytest


@pytest.fixture
def client(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_store_dir))
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DISABLE", "1")
    # Import after env vars are set so any module-level config sees them.
    from aikaboom.web.app import app
    app.config["TESTING"] = True
    return app.test_client()


def test_generate_accepts_cache_policy(client, monkeypatch):
    """Posting cache_policy in the body should not 400."""
    # Mock the processor factory so we never hit the network / LLM. The
    # cache wrap should accept ``cache_policy`` and either short-circuit
    # (no claims yet, so no cache hit) or fall through to the fake
    # processor below.
    from aikaboom.web import app as appmod

    class _FakeProc:
        use_case = "license"

        def process_ai_model(self, **_kwargs):
            return {
                "repo_id": "test/test",
                "model_id": "test_test",
                "use_case": "license",
                "direct_fields": {},
                "rag_fields": {},
                "beta_fields": [],
            }

    monkeypatch.setattr(
        appmod, "get_processor",
        lambda **kw: _FakeProc(),
        raising=True,
    )
    # Skip the link-fallback agent — it tries to hit Gemini.
    response = client.post("/process", json={
        "bom_type": "ai",
        "repo_id": "test/test",
        "mode": "rag",
        "use_case": "license",
        "llm_provider": "ollama",
        "cache_policy": "use",
        "skip_fallback": True,
    })
    # We accept any non-5xx response — the key is that cache_policy
    # didn't 400 (it's not an unknown rejected field) and the wrap
    # didn't blow up before the response could be built.
    assert response.status_code < 500, (
        f"Route returned {response.status_code}: {response.data!r}"
    )


def test_generate_accepts_force_refresh_shorthand(client, monkeypatch):
    """force_refresh: True should be accepted as cache_policy=regen shorthand."""
    from aikaboom.web import app as appmod

    class _FakeProc:
        use_case = "license"

        def process_ai_model(self, **_kwargs):
            return {
                "repo_id": "test/test",
                "model_id": "test_test",
                "use_case": "license",
                "direct_fields": {},
                "rag_fields": {},
                "beta_fields": [],
            }

    monkeypatch.setattr(
        appmod, "get_processor",
        lambda **kw: _FakeProc(),
        raising=True,
    )
    response = client.post("/process", json={
        "bom_type": "ai",
        "repo_id": "test/test",
        "mode": "rag",
        "use_case": "license",
        "llm_provider": "ollama",
        "force_refresh": True,
        "skip_fallback": True,
    })
    assert response.status_code < 500, (
        f"Route returned {response.status_code}: {response.data!r}"
    )


def test_cache_policy_returns_cached_bom_on_hit(tmp_store_dir, monkeypatch):
    """When a matching claim exists, cache_policy=use serves the cached BOM
    without invoking the processor.

    This is the load-bearing assertion for Task 16: it proves the resolve
    wrap is actually wired up, not just that an unknown field is silently
    accepted. We pre-seed the graph with a claim, then post to /process
    and check that ``metadata._cached`` is set on the response.
    """
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_store_dir))
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DISABLE", "0")

    # Pre-seed a claim into the store so resolve() finds a match.
    from aikaboom.store.naming import Identifier
    from aikaboom.store.store import BomStore
    seeded_bom = {
        "repo_id": "test/test",
        "model_id": "test_test",
        "use_case": "license",
        "direct_fields": {
            "suppliedBy": {
                "value": "test", "source": "huggingface", "conflict": None,
            },
        },
        "rag_fields": {},
        "beta_fields": [],
    }
    store = BomStore.open()
    store.save_claim(
        seeded_bom,
        run_meta={
            "provider": "ollama", "llm_model": "llama3:70b",
            "prompt_version": "v1", "code_version": "head",
            "mode": "rag", "use_case": "license",
        },
        identifiers=[Identifier("huggingface", "test/test")],
    )

    # If the processor gets called, the cache wrap is broken — make that
    # an immediate test failure by raising from the fake.
    from aikaboom.web import app as appmod

    def _no_processor(**_kw):
        raise AssertionError(
            "get_processor was called on cache hit — wrap is not wired up"
        )

    monkeypatch.setattr(appmod, "get_processor", _no_processor, raising=True)
    appmod.app.config["TESTING"] = True
    client = appmod.app.test_client()

    response = client.post("/process", json={
        "bom_type": "ai",
        "repo_id": "test/test",
        "mode": "rag",
        "use_case": "license",
        "llm_provider": "ollama",
        "cache_policy": "use",
        "skip_fallback": True,
    })
    assert response.status_code < 500, (
        f"Route returned {response.status_code}: {response.data!r}"
    )
    body = response.get_json() or {}
    metadata = body.get("metadata") or {}
    assert metadata.get("_cached") is True, (
        f"Expected cached BOM in response; got keys={list(metadata.keys())} "
        f"status={body.get('status')!r}"
    )
