"""
Unit tests for LinkFallbackFinder
"""
import pytest
from aikaboom.utils.link_fallback import LinkFallbackFinder


class TestLinkFallbackFinder:
    """Tests for LinkFallbackFinder"""
    
    def test_initialization_without_api_key(self):
        """Test that initialization without API key creates disabled client"""
        finder = LinkFallbackFinder(api_key=None)
        assert finder.client is None
    
    def test_is_valid_url(self):
        """Test URL validation"""
        finder = LinkFallbackFinder(api_key="dummy_key_for_testing")
        
        # Test HuggingFace URLs
        assert finder._is_valid_url("https://huggingface.co/microsoft/model", "huggingface")
        assert not finder._is_valid_url("https://github.com/user/repo", "huggingface")
        
        # Test ArXiv URLs
        assert finder._is_valid_url("https://arxiv.org/abs/2301.12345", "arxiv")
        assert not finder._is_valid_url("https://github.com/user/repo", "arxiv")
        
        # Test GitHub URLs
        assert finder._is_valid_url("https://github.com/user/repo", "github")
        assert not finder._is_valid_url("https://arxiv.org/abs/123", "github")
