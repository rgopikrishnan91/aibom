import pytest
from aikaboom.store.naming import Identifier
from aikaboom.store.iris import (
    artifact_iri,
    version_iri,
    claim_iri,
    run_iri,
    vote_iri,
    agent_iri,
    source_iri,
)


class TestArtifactIri:
    def test_deterministic(self):
        primary = Identifier("huggingface", "mistralai/mistral-7b")
        assert artifact_iri(primary) == artifact_iri(primary)

    def test_url_safe(self):
        primary = Identifier("huggingface", "mistralai/mistral-7b")
        iri = artifact_iri(primary)
        # IRI hash is hex — no slashes, colons, unicode.
        prefix = "bom:artifact/"
        assert iri.startswith(prefix)
        suffix = iri[len(prefix):]
        assert all(c in "0123456789abcdef" for c in suffix)
        assert len(suffix) == 64  # sha256 hex digest length


class TestVersionIri:
    def test_includes_version(self):
        primary = Identifier("huggingface", "mistralai/mistral-7b")
        v = version_iri(artifact_iri(primary), "27d67f1b")
        assert v.startswith("bom:version/")
        assert v.endswith("/27d67f1b")


class TestRunIri:
    def test_hash_of_run_params_is_stable(self):
        params = {
            "provider": "openrouter",
            "llm_model": "anthropic/claude-3-haiku",
            "prompt_version": "v12",
            "code_version": "abc1234",
            "mode": "rag",
            "use_case": "license",
        }
        assert run_iri(params) == run_iri(params)

    def test_different_params_different_iri(self):
        a = run_iri({"provider": "openrouter", "llm_model": "claude-3-haiku", "prompt_version": "v12", "code_version": "abc1234", "mode": "rag", "use_case": "license"})
        b = run_iri({"provider": "openrouter", "llm_model": "gpt-4o-mini", "prompt_version": "v12", "code_version": "abc1234", "mode": "rag", "use_case": "license"})
        assert a != b


class TestClaimAndVote:
    def test_claim_iri_is_uuid_form(self):
        iri = claim_iri()
        assert iri.startswith("bom:claim/")
        suffix = iri[len("bom:claim/"):]
        # uuid4 hex form: 32 chars, no dashes (we normalize).
        assert len(suffix) >= 32

    def test_two_claims_distinct(self):
        assert claim_iri() != claim_iri()

    def test_vote_iri_format(self):
        assert vote_iri().startswith("bom:vote/")


class TestAgentIri:
    def test_deterministic_from_string(self):
        assert agent_iri("gopi@laptop") == agent_iri("gopi@laptop")

    def test_different_strings_different_iri(self):
        assert agent_iri("gopi@laptop") != agent_iri("alice@desktop")


class TestSourceIri:
    def test_huggingface(self):
        assert source_iri("huggingface") == "aibom:source/huggingface"

    def test_github(self):
        assert source_iri("github") == "aibom:source/github"
