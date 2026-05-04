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


class _FakeHFSibling:
    def __init__(self, size=None):
        self.size = size


class _FakeHFRepoInfo:
    """Minimal stand-in for huggingface_hub ModelInfo."""

    def __init__(self, cardData=None, tags=None, siblings=None, repo_id="test/repo"):
        self.id = repo_id
        self.siblings = siblings
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


class _FakeGHRepo:
    """Minimal stand-in for a PyGithub Repository."""

    def __init__(
        self,
        full_name="org/repo",
        description=None,
        topics=None,
        license_name=None,
        language=None,
        default_branch=None,
        size=None,
    ):
        self.full_name = full_name
        self.description = description
        self._topics = topics or []
        self._license_name = license_name
        self.language = language
        self.default_branch = default_branch
        self.size = size

    def get_topics(self):
        return self._topics

    def get_license(self):
        if self._license_name is None:
            return None
        class _Wrapper:
            class license:
                pass
        w = _Wrapper()
        w.license.name = self._license_name
        return w


class TestStructuredChunks:
    """Synthetic structured chunks pack HF/GH metadata into prose so the
    RAG retriever can compare against README / arXiv text. The dataset
    size signal in particular feeds the new HF > GH > arXiv priority for
    `datasetSize`.
    """

    def test_hf_dataset_chunk_includes_estimated_total_size_bytes(self):
        info = _FakeHFRepoInfo(
            siblings=[_FakeHFSibling(size=1_000_000_000), _FakeHFSibling(size=200_000_000)],
            cardData={"license": "cc-by-4.0"},
        )
        chunk = MetadataFetcher.huggingface_structured_chunk(info, bom_type="data")
        assert "estimated_total_size_bytes: 1200000000" in chunk

    def test_hf_ai_chunk_omits_size(self):
        # Even with siblings, AI BOM chunks don't include a size aggregate.
        info = _FakeHFRepoInfo(
            siblings=[_FakeHFSibling(size=999)],
            cardData={"license": "mit"},
        )
        chunk = MetadataFetcher.huggingface_structured_chunk(info, bom_type="ai")
        assert "estimated_total_size_bytes" not in chunk

    def test_hf_chunk_skips_size_when_no_siblings(self):
        info = _FakeHFRepoInfo(cardData={"license": "mit"})
        chunk = MetadataFetcher.huggingface_structured_chunk(info, bom_type="data")
        assert "estimated_total_size_bytes" not in chunk

    def test_gh_chunk_includes_repository_size_bytes(self):
        repo = _FakeGHRepo(size=4096, description="A test repo")
        chunk = MetadataFetcher.github_structured_chunk(repo, bom_type="data")
        # GitHub returns size in KB; we render bytes.
        assert "repository_size_bytes: 4194304" in chunk

    def test_gh_chunk_skips_size_when_zero(self):
        repo = _FakeGHRepo(size=0, description="empty repo")
        chunk = MetadataFetcher.github_structured_chunk(repo, bom_type="data")
        assert "repository_size_bytes" not in chunk

    def test_gh_chunk_with_no_metadata_returns_empty(self):
        repo = _FakeGHRepo()
        chunk = MetadataFetcher.github_structured_chunk(repo, bom_type="data")
        assert chunk == ""
