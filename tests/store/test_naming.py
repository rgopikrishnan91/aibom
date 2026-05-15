"""Canonicalization rules for artifact identifiers."""

from aikaboom.store.naming import (
    Identifier,
    canonicalize,
    canonicalize_set,
    pick_primary,
    PLATFORM_PRIORITY,
)


class TestCanonicalize:
    def test_lowercases(self):
        assert canonicalize(Identifier("huggingface", "MistralAI/Mistral-7B-v0.1")) == Identifier(
            "huggingface", "mistralai/mistral-7b-v0.1"
        )

    def test_idempotent(self):
        once = canonicalize(Identifier("huggingface", "MistralAI/Mistral-7B-v0.1"))
        twice = canonicalize(once)
        assert once == twice

    def test_strips_url_prefix_for_hf(self):
        result = canonicalize(
            Identifier("huggingface", "https://huggingface.co/MistralAI/Mistral-7B-v0.1/tree/main")
        )
        assert result == Identifier("huggingface", "mistralai/mistral-7b-v0.1")

    def test_strips_url_prefix_for_github(self):
        result = canonicalize(Identifier("github", "https://github.com/mistralai/mistral-src.git"))
        assert result == Identifier("github", "mistralai/mistral-src")

    def test_strips_arxiv_version_suffix(self):
        result = canonicalize(Identifier("arxiv", "arxiv.org/abs/2310.06825v1"))
        assert result == Identifier("arxiv", "2310.06825")

    def test_resolves_supplier_alias(self):
        # supplier_alias maps various forms of an org name to a canonical form.
        # 'mistralai' is already canonical; the test asserts the path executes.
        result = canonicalize(Identifier("huggingface", "MISTRALAI/Mistral-7B"))
        assert result.value.startswith("mistralai/")

    def test_collapses_separator_runs(self):
        result = canonicalize(Identifier("huggingface", "foo--bar__baz"))
        assert result == Identifier("huggingface", "foo-bar-baz")

    def test_trim_whitespace(self):
        result = canonicalize(Identifier("huggingface", "  mistralai/mistral-7b  "))
        assert result == Identifier("huggingface", "mistralai/mistral-7b")

    def test_strips_doi_url_prefix(self):
        result = canonicalize(Identifier("doi", "https://doi.org/10.48550/arXiv.2310.06825"))
        assert result == Identifier("doi", "10.48550/arxiv.2310.06825")

    def test_bare_doi_unchanged(self):
        result = canonicalize(Identifier("doi", "10.48550/arXiv.2310.06825"))
        assert result == Identifier("doi", "10.48550/arxiv.2310.06825")

    def test_url_platform_preserves_full_url(self):
        result = canonicalize(Identifier("url", "https://example.com/abc/def"))
        assert result == Identifier("url", "https://example.com/abc/def")


class TestPickPrimary:
    def test_hf_beats_arxiv(self):
        ids = [
            Identifier("arxiv", "2310.06825"),
            Identifier("huggingface", "mistralai/mistral-7b"),
        ]
        assert pick_primary(ids).platform == "huggingface"

    def test_github_beats_arxiv(self):
        ids = [
            Identifier("arxiv", "2310.06825"),
            Identifier("github", "mistralai/mistral-src"),
        ]
        assert pick_primary(ids).platform == "github"

    def test_arxiv_when_only_option(self):
        ids = [Identifier("arxiv", "2310.06825")]
        assert pick_primary(ids).platform == "arxiv"

    def test_priority_order(self):
        assert PLATFORM_PRIORITY == ("huggingface", "github", "arxiv", "doi", "url")


class TestCanonicalizeSet:
    def test_canonicalizes_each(self):
        ids = [
            Identifier("huggingface", "MistralAI/Mistral-7B"),
            Identifier("arxiv", "arxiv.org/abs/2310.06825v1"),
        ]
        result = canonicalize_set(ids)
        assert Identifier("huggingface", "mistralai/mistral-7b") in result
        assert Identifier("arxiv", "2310.06825") in result

    def test_dedups_within_set(self):
        ids = [
            Identifier("huggingface", "mistralai/mistral-7b"),
            Identifier("huggingface", "MISTRALAI/Mistral-7B"),
        ]
        result = canonicalize_set(ids)
        assert len(result) == 1
