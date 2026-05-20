from aikaboom.store.cache_resolver import (
    CachePolicy,
    decide,
    render_prompt,
)
from aikaboom.store.store import ResolveResult


def make_result(claims):
    return ResolveResult(
        existing_artifact="bom:artifact/abc",
        artifact_label="mistralai/Mistral-7B-v0.1",
        matching_claims=claims,
    )


class TestDecide:
    def test_no_claims_means_generate(self):
        result = ResolveResult(existing_artifact=None, artifact_label=None)
        assert (
            decide(result, CachePolicy.PROMPT, interactive=True, input_fn=lambda _: "u")
            == "generate"
        )

    def test_auto_uses_most_recent(self):
        result = make_result([{"iri": "bom:claim/x", "llm_model": "x", "created_at": "2026-01-01"}])
        assert decide(result, CachePolicy.USE, interactive=False) == "use"

    def test_regen_policy_skips_prompt(self):
        result = make_result([{"iri": "bom:claim/x", "llm_model": "x", "created_at": "2026-01-01"}])
        assert decide(result, CachePolicy.REGEN, interactive=False) == "generate"

    def test_prompt_with_use_response(self):
        result = make_result(
            [{"iri": "bom:claim/x", "llm_model": "claude-3-haiku", "created_at": "2025-11-04"}]
        )
        assert decide(result, CachePolicy.PROMPT, interactive=True, input_fn=lambda _: "u") == "use"

    def test_prompt_with_regen_response(self):
        result = make_result(
            [{"iri": "bom:claim/x", "llm_model": "claude-3-haiku", "created_at": "2025-11-04"}]
        )
        assert (
            decide(result, CachePolicy.PROMPT, interactive=True, input_fn=lambda _: "r")
            == "generate"
        )

    def test_non_interactive_with_prompt_policy_defaults_to_use(self):
        """When TTY is unavailable, prompt policy degrades to use."""
        result = make_result([{"iri": "bom:claim/x", "llm_model": "x", "created_at": "2026-01-01"}])
        assert decide(result, CachePolicy.PROMPT, interactive=False) == "use"


class TestRenderPrompt:
    def test_lists_existing_claims(self):
        result = make_result(
            [
                {
                    "iri": "bom:claim/a",
                    "llm_model": "claude-3-haiku",
                    "created_at": "2025-11-04T10:00:00Z",
                },
                {
                    "iri": "bom:claim/b",
                    "llm_model": "gpt-4o-mini",
                    "created_at": "2025-12-19T10:00:00Z",
                },
            ]
        )
        text = render_prompt(result, planned_llm="claude-opus-4-7")
        assert "claude-3-haiku" in text
        assert "gpt-4o-mini" in text
        assert "claude-opus-4-7" in text
        assert "[u]" in text
        assert "[r]" in text
