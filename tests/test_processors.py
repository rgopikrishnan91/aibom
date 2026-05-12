"""
Unit tests for AIBOMProcessor and DATABOMProcessor.

``AIBOMProcessor.__init__`` and ``DATABOMProcessor.__init__`` build a real
``AgenticRAG`` which instantiates a ``ChatOpenAI`` client; the OpenAI SDK
requires ``OPENAI_API_KEY`` at construction time. These tests focus on the
processor's own attribute wiring and ID-generation helpers, not the LLM
client itself, so we mock ``create_llm`` and ``HuggingFaceEmbeddings`` to
keep the tests hermetic (mirrors the pattern in
``test_regression.TestProcessorInitWithoutAPIKey``).
"""
import pytest
from unittest.mock import patch

from aikaboom.core.processors import AIBOMProcessor, DATABOMProcessor


@pytest.fixture
def _mock_llm():
    with patch("aikaboom.core.agentic_rag.create_llm") as llm, \
         patch("aikaboom.core.agentic_rag.HuggingFaceEmbeddings") as emb:
        yield llm, emb


class TestAIBOMProcessor:
    """Tests for AIBOMProcessor"""

    def test_initialization(self, _mock_llm):
        """Test processor initialization"""
        processor = AIBOMProcessor(
            model="gpt-4o",
            mode="rag",
            use_case="complete"
        )
        assert processor.mode == "rag"
        assert processor.model == "gpt-4o"
        assert processor.use_case == "complete"

    def test_generate_model_id(self, _mock_llm):
        """Test model ID generation"""
        processor = AIBOMProcessor()

        # Test with repo_id
        model_id = processor.generate_model_id(
            repo_id="microsoft/DialoGPT-medium",
            github_url=None
        )
        assert model_id == "microsoft_DialoGPT-medium"

        # Test with github_url
        model_id = processor.generate_model_id(
            repo_id=None,
            github_url="https://github.com/microsoft/DialoGPT"
        )
        assert model_id == "microsoft_DialoGPT"


class TestDATABOMProcessor:
    """Tests for DATABOMProcessor"""

    def test_initialization(self, _mock_llm):
        """Test processor initialization"""
        processor = DATABOMProcessor(
            model="gpt-4o",
            mode="direct",
            use_case="safety"
        )
        assert processor.mode == "direct"
        assert processor.model == "gpt-4o"
        assert processor.use_case == "safety"

    def test_generate_dataset_id(self, _mock_llm):
        """Test dataset ID generation"""
        processor = DATABOMProcessor()

        # Test with HF URL
        dataset_id = processor.generate_dataset_id(
            arxiv_url=None,
            github_url=None,
            hf_url="https://huggingface.co/datasets/squad"
        )
        assert dataset_id is not None
        assert "squad" in dataset_id.lower()
