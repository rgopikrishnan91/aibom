"""
Baseline regression tests for MetadataFetcher URL parsing functions.
Tests only pure logic functions — no network calls.
"""
import pytest
from bom_tools.utils.metadata_fetcher import MetadataFetcher


class TestExtractHfRepoId:
    """Tests for MetadataFetcher.extract_hf_repo_id."""

    def test_standard_model_url(self):
        result = MetadataFetcher.extract_hf_repo_id("https://huggingface.co/microsoft/DialoGPT-medium")
        assert result == "microsoft/DialoGPT-medium"

    def test_dataset_url(self):
        result = MetadataFetcher.extract_hf_repo_id("https://huggingface.co/datasets/squad")
        # "datasets" is removed, so for single-part path it may return None
        # For multi-part: "datasets/namespace/repo" → "namespace/repo"
        # "squad" alone after removing "datasets" has only 1 part
        # Behavior: parts = ["squad"] → len < 2 → None
        assert result is None

    def test_dataset_url_with_namespace(self):
        result = MetadataFetcher.extract_hf_repo_id("https://huggingface.co/datasets/rajpurkar/squad")
        assert result == "rajpurkar/squad"

    def test_url_with_trailing_slash(self):
        result = MetadataFetcher.extract_hf_repo_id("https://huggingface.co/microsoft/DialoGPT-medium/")
        assert result == "microsoft/DialoGPT-medium"

    def test_short_url(self):
        result = MetadataFetcher.extract_hf_repo_id("https://huggingface.co/")
        assert result is None


class TestExtractGithubUserRepo:
    """Tests for MetadataFetcher.extract_github_user_repo."""

    def test_standard_url(self):
        user, repo = MetadataFetcher.extract_github_user_repo("https://github.com/microsoft/DialoGPT")
        assert user == "microsoft"
        assert repo == "DialoGPT"

    def test_url_with_subpath(self):
        user, repo = MetadataFetcher.extract_github_user_repo("https://github.com/org/repo/tree/main/src")
        assert user == "org"
        assert repo == "repo"

    def test_short_url(self):
        user, repo = MetadataFetcher.extract_github_user_repo("https://github.com/")
        assert user is None
        assert repo is None


class TestExtractRepoPath:
    """Tests for MetadataFetcher.extract_repo_path."""

    def test_standard_url(self):
        result = MetadataFetcher.extract_repo_path("https://github.com/microsoft/DialoGPT")
        assert result == "microsoft/DialoGPT"

    def test_url_with_branches(self):
        result = MetadataFetcher.extract_repo_path("https://github.com/org/repo/tree/main/subfolder")
        assert result == "org/repo"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Invalid GitHub repository URL"):
            MetadataFetcher.extract_repo_path("https://github.com/")


class TestExtractRepoIdFromHfUrl:
    """Tests for MetadataFetcher.extract_repo_id_from_hf_url."""

    def test_dataset_url(self):
        result = MetadataFetcher.extract_repo_id_from_hf_url(
            "https://huggingface.co/datasets/rajpurkar/squad"
        )
        assert result == "rajpurkar/squad"

    def test_dataset_url_with_blob(self):
        result = MetadataFetcher.extract_repo_id_from_hf_url(
            "https://huggingface.co/datasets/org/data/blob/main/file.csv"
        )
        assert result == "org/data"

    def test_model_url(self):
        result = MetadataFetcher.extract_repo_id_from_hf_url(
            "https://huggingface.co/microsoft/DialoGPT-medium"
        )
        assert result == "microsoft/DialoGPT-medium"

    def test_none_input(self):
        assert MetadataFetcher.extract_repo_id_from_hf_url(None) is None

    def test_empty_string(self):
        assert MetadataFetcher.extract_repo_id_from_hf_url("") is None

    def test_non_string_input(self):
        assert MetadataFetcher.extract_repo_id_from_hf_url(123) is None
