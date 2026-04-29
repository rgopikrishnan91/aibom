"""
Baseline regression tests for MetadataFetcher URL parsing functions.
Tests only pure logic functions — no network calls.
"""
import pytest
from aikaboom.utils.metadata_fetcher import MetadataFetcher


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


class _FakeHFRepoInfo:
    """Minimal stand-in for huggingface_hub ModelInfo."""

    def __init__(self, cardData=None, tags=None):
        self.cardData = cardData or {}
        self.tags = tags or []


class TestExtractHuggingfaceModelTree:
    """Tests for MetadataFetcher.extract_huggingface_model_tree."""

    def test_card_data_datasets_and_base_model(self):
        info = _FakeHFRepoInfo(
            cardData={
                "datasets": ["squad", "Common Crawl"],
                "base_model": "meta-llama/Llama-3",
            }
        )
        tree = MetadataFetcher.extract_huggingface_model_tree(info)
        assert tree["trainedOnDatasets"] == ["squad", "Common Crawl"]
        assert tree["modelLineage"] == ["meta-llama/Llama-3"]
        assert tree["testedOnDatasets"] == []

    def test_model_index_results_become_tested_on(self):
        info = _FakeHFRepoInfo(
            cardData={
                "model-index": [
                    {"results": [
                        {"dataset": {"name": "MMLU"}},
                        {"dataset": {"name": "HellaSwag"}},
                    ]}
                ],
            }
        )
        tree = MetadataFetcher.extract_huggingface_model_tree(info)
        assert tree["testedOnDatasets"] == ["MMLU", "HellaSwag"]

    def test_tags_supplement_card_data(self):
        info = _FakeHFRepoInfo(
            cardData={"datasets": ["squad"]},
            tags=["dataset:squad", "dataset:wikitext", "base_model:google-bert/bert-base-uncased"],
        )
        tree = MetadataFetcher.extract_huggingface_model_tree(info)
        assert tree["trainedOnDatasets"] == ["squad", "wikitext"]
        assert tree["modelLineage"] == ["google-bert/bert-base-uncased"]

    def test_none_input_returns_empty_buckets(self):
        tree = MetadataFetcher.extract_huggingface_model_tree(None)
        assert tree == {"trainedOnDatasets": [], "testedOnDatasets": [], "modelLineage": []}
