"""
Unit tests for AIBOMProcessor and DATABOMProcessor
"""
import pytest
from bom_tools.core.processors import AIBOMProcessor, DATABOMProcessor


class TestAIBOMProcessor:
    """Tests for AIBOMProcessor"""
    
    def test_initialization(self):
        """Test processor initialization"""
        processor = AIBOMProcessor(
            model="gpt-4o",
            mode="rag",
            use_case="complete"
        )
        assert processor.mode == "rag"
        assert processor.model == "gpt-4o"
        assert processor.use_case == "complete"
    
    def test_generate_model_id(self):
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
    
    def test_initialization(self):
        """Test processor initialization"""
        processor = DATABOMProcessor(
            model="gpt-4o",
            mode="direct",
            use_case="safety"
        )
        assert processor.mode == "direct"
        assert processor.model == "gpt-4o"
        assert processor.use_case == "safety"
    
    def test_generate_dataset_id(self):
        """Test dataset ID generation"""
        processor = DATABOMProcessor()
        
        # Test with HF URL
        dataset_id = processor.generate_dataset_id(
            arxiv_url=None,
            github_url=None,
            hf_url="https://huggingface.co/datasets/squad"
        )
        assert "squad" in dataset_id or "datasets_squad" in dataset_id
