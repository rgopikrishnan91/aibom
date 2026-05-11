"""Tests for the SAIL-derived parsing utilities.

Covers ``sail_link_extractor`` (GitHub-URL extraction from HF model
cards) and ``sail_version_extractor`` (size / variant / explicit
version components). All tests run offline.
"""
import pytest

from aikaboom.utils.sail_link_extractor import (
    extract_github_urls_from_card,
    extract_org_from_github_url,
)
from aikaboom.utils.sail_version_extractor import (
    extract_explicit_version,
    extract_size_component,
    extract_variant_tag,
    extract_version_from_repo_id,
)


# ---------------------------------------------------------------------------
# sail_link_extractor
# ---------------------------------------------------------------------------

class TestExtractGithubUrlsFromCard:
    def test_returns_matching_url(self):
        card = "Source code: https://github.com/mistralai/mistral-src for inference."
        urls = extract_github_urls_from_card(card, "mistralai/Mistral-7B-v0.1")
        assert urls == ["https://github.com/mistralai/mistral-src"]

    def test_filters_unrelated_repo(self):
        # owner segment doesn't match the model name segments at all
        card = "We use https://github.com/random/unrelated-tool for evaluation."
        urls = extract_github_urls_from_card(card, "mistralai/Mistral-7B-v0.1")
        assert urls == []

    def test_strips_trailing_markdown_punctuation(self):
        card = "Repo: (https://github.com/QwenLM/Qwen2.5)."
        urls = extract_github_urls_from_card(card, "Qwen/Qwen2.5-7B-Instruct")
        assert urls == ["https://github.com/QwenLM/Qwen2.5"]

    def test_returns_empty_for_empty_card(self):
        assert extract_github_urls_from_card("", "mistralai/Mistral-7B-v0.1") == []
        assert extract_github_urls_from_card(None, "mistralai/Mistral-7B-v0.1") == []

    def test_dedup_and_sort(self):
        card = (
            "https://github.com/mistralai/mistral-src "
            "and again https://github.com/mistralai/mistral-src "
            "plus https://github.com/mistralai/mistral-finetune"
        )
        urls = extract_github_urls_from_card(card, "mistralai/Mistral-7B-v0.1")
        assert urls == [
            "https://github.com/mistralai/mistral-finetune",
            "https://github.com/mistralai/mistral-src",
        ]


class TestExtractOrgFromGithubUrl:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://github.com/QwenLM/Qwen2.5", "QwenLM"),
            ("https://github.com/mistralai/mistral-src/blob/main/README.md", "mistralai"),
            ("https://github.com/", None),
            ("https://huggingface.co/mistralai", None),
            ("", None),
            (None, None),
        ],
    )
    def test_pulls_owner_segment(self, url, expected):
        assert extract_org_from_github_url(url) == expected


# ---------------------------------------------------------------------------
# sail_version_extractor
# ---------------------------------------------------------------------------

class TestExtractSizeComponent:
    @pytest.mark.parametrize(
        "repo_id,expected",
        [
            ("mistralai/Mistral-7B-v0.1", "7B"),
            ("Qwen/Qwen2.5-7B-Instruct", "7B"),
            ("TheBloke/Llama-2-7B-Chat-GPTQ", "7B"),
            ("EleutherAI/pythia-1B-deduped", "1B"),
            ("microsoft/phi-3-mini-4k-instruct", None),  # no plain digit+unit token
            ("google-bert/bert-base-uncased", None),
            ("foo/something-350M", "350M"),
            ("", None),
            (None, None),
        ],
    )
    def test_size_token_extraction(self, repo_id, expected):
        assert extract_size_component(repo_id) == expected


class TestExtractVariantTag:
    @pytest.mark.parametrize(
        "repo_id,expected",
        [
            ("TheBloke/Llama-2-7B-Chat-GPTQ", "quantized"),
            ("TheBloke/Mistral-7B-v0.1-AWQ", "quantized"),
            ("Qwen/Qwen2.5-7B-Instruct", "instruction-tuned"),
            ("distilbert/distilbert-base-uncased", "compressed"),
            ("EleutherAI/pythia-1B-deduped", "deduplicated"),
            ("google-bert/bert-base-uncased", None),
            ("", None),
            (None, None),
        ],
    )
    def test_variant_classification(self, repo_id, expected):
        assert extract_variant_tag(repo_id) == expected


class TestExtractExplicitVersion:
    @pytest.mark.parametrize(
        "repo_id,expected",
        [
            ("mistralai/Mistral-7B-v0.1", "v0.1"),
            ("mistralai/Mistral-7B-Instruct-v0.2", "v0.2"),
            ("foo/Bar-v1.2.3", "v1.2.3"),
            ("Qwen/Qwen2.5-7B-Instruct", None),
            ("google-bert/bert-base-uncased", None),
            ("", None),
            (None, None),
        ],
    )
    def test_version_marker_extraction(self, repo_id, expected):
        assert extract_explicit_version(repo_id) == expected


class TestExtractVersionFromRepoId:
    @pytest.mark.parametrize(
        "repo_id,expected",
        [
            ("mistralai/Mistral-7B-v0.1", "7B-v0.1"),
            ("Qwen/Qwen2.5-7B-Instruct", "7B-instruction-tuned"),
            ("TheBloke/Llama-2-7B-Chat-GPTQ", "7B-quantized"),
            ("EleutherAI/pythia-1B-deduped", "1B-deduplicated"),
            ("google-bert/bert-base-uncased", None),
        ],
    )
    def test_combined_extraction(self, repo_id, expected):
        assert extract_version_from_repo_id(repo_id) == expected
