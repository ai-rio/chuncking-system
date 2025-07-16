import pytest
from unittest.mock import Mock, patch, MagicMock
from src.chunkers.hybrid_chunker import HybridMarkdownChunker
from src.llm.factory import LLMFactory
from src.llm.providers import OpenAIProvider, JinaProvider, AnthropicProvider
from src.exceptions import TokenizationError, ConfigurationError


class TestLLMIntegrationWithHybridChunker:
    """Integration tests for LLM providers with HybridMarkdownChunker"""
    
    @pytest.fixture
    def sample_markdown_content(self):
        """Sample markdown content for testing"""
        return """# Chapter 1: Introduction

This is the introduction to our comprehensive guide on advanced machine learning techniques.

## Section 1.1: Overview

Machine learning has revolutionized how we approach complex problems in various domains including natural language processing, computer vision, and data analysis.

### Subsection 1.1.1: Core Concepts

The fundamental concepts include supervised learning, unsupervised learning, and reinforcement learning approaches.

## Section 1.2: Applications

Real-world applications span across industries from healthcare to finance, enabling automated decision-making and pattern recognition.

# Chapter 2: Technical Implementation

This chapter covers the technical aspects of implementing machine learning solutions.

## Section 2.1: Data Processing

Proper data preprocessing is crucial for model performance and includes cleaning, normalization, and feature engineering steps.
"""
    
    @pytest.fixture
    def mock_openai_provider(self):
        """Mock OpenAI provider for testing"""
        mock_provider = Mock(spec=OpenAIProvider)
        mock_provider.count_tokens.return_value = 50
        mock_provider.provider_name = "openai"
        mock_provider.model = "gpt-3.5-turbo"
        return mock_provider
    
    @pytest.fixture
    def mock_jina_provider(self):
        """Mock Jina provider for testing"""
        mock_provider = Mock(spec=JinaProvider)
        mock_provider.count_tokens.return_value = 45
        mock_provider.provider_name = "jina"
        mock_provider.model = "jina-embeddings-v2-base-en"
        return mock_provider
    
    @patch('src.llm.factory.LLMFactory')
    def test_hybrid_chunker_with_openai_provider(self, mock_factory, mock_openai_provider, sample_markdown_content):
        """Test HybridMarkdownChunker with OpenAI provider"""
        mock_factory.create_provider.return_value = mock_openai_provider
        
        chunker = HybridMarkdownChunker(chunk_size=200, chunk_overlap=50)
        
        # Verify LLM provider is used
        assert chunker.llm_provider == mock_openai_provider
        assert chunker.llm_provider is not None
        
        # Test token counting uses LLM provider
        token_count = chunker._token_length("Test content")
        assert token_count == 50
        mock_openai_provider.count_tokens.assert_called_with("Test content")
        
        # Test chunking with LLM provider
        chunks = chunker.chunk_document(sample_markdown_content)
        
        assert len(chunks) > 0
        assert all(chunk.page_content for chunk in chunks)
        assert all(chunk.metadata.get('chunk_tokens') for chunk in chunks)
    
    @patch('src.llm.factory.LLMFactory')
    def test_hybrid_chunker_with_jina_provider(self, mock_factory, mock_jina_provider, sample_markdown_content):
        """Test HybridMarkdownChunker with Jina provider"""
        mock_factory.create_provider.return_value = mock_jina_provider
        
        chunker = HybridMarkdownChunker(chunk_size=300, chunk_overlap=75)
        
        # Verify Jina provider is used
        assert chunker.llm_provider == mock_jina_provider
        
        # Test token counting uses Jina provider
        token_count = chunker._token_length("Test content for Jina")
        assert token_count == 45
        mock_jina_provider.count_tokens.assert_called_with("Test content for Jina")
        
        # Test chunking produces valid results
        chunks = chunker.chunk_document(sample_markdown_content)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.page_content.strip()
            assert chunk.metadata.get('chunk_tokens') == 45  # Mocked value
            assert chunk.metadata.get('llm_provider') == 'jina'
    
    @patch('src.llm.factory.LLMFactory')
    @patch('src.chunkers.hybrid_chunker.tiktoken')
    def test_hybrid_chunker_fallback_to_tiktoken(self, mock_tiktoken, mock_factory, sample_markdown_content):
        """Test HybridChunker fallback to tiktoken when LLM provider fails"""
        # Mock LLM factory to raise exception
        mock_factory.create_provider.side_effect = Exception("LLM provider failed")
        
        # Mock tiktoken for fallback
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        chunker = HybridMarkdownChunker(chunk_size=200, chunk_overlap=50)
        
        # Verify fallback occurred
        assert chunker.llm_provider is None
        assert chunker.tokenizer is not None
        
        # Test token counting uses tiktoken fallback
        token_count = chunker._token_length("Test content")
        assert token_count == 5
        mock_tokenizer.encode.assert_called_with("Test content")
        
        # Test chunking still works with fallback
        chunks = chunker.chunk_document(sample_markdown_content)
        assert len(chunks) > 0
    
    @patch('src.llm.factory.LLMFactory')
    def test_hybrid_chunker_with_provider_token_error(self, mock_factory, mock_openai_provider):
        """Test HybridChunker handling LLM provider token counting errors"""
        mock_openai_provider.count_tokens.side_effect = Exception("Token counting failed")
        mock_factory.create_provider.return_value = mock_openai_provider
        
        chunker = HybridMarkdownChunker(chunk_size=200, chunk_overlap=50)
        
        # Test that token counting error is properly handled
        with pytest.raises(TokenizationError, match="Failed to calculate token length"):
            chunker._token_length("Test content")
    
    @patch('src.llm.factory.LLMFactory')
    def test_hybrid_chunker_provider_metadata_enrichment(self, mock_factory, mock_openai_provider, sample_markdown_content):
        """Test that chunks are enriched with LLM provider metadata"""
        mock_factory.create_provider.return_value = mock_openai_provider
        
        chunker = HybridMarkdownChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_document(sample_markdown_content)
        
        for chunk in chunks:
            # Check that LLM provider metadata is added
            assert chunk.metadata.get('llm_provider') == 'openai'
            assert chunk.metadata.get('llm_model') == 'gpt-3.5-turbo'
            assert 'chunk_tokens' in chunk.metadata
    
    @patch('src.llm.factory.LLMFactory')
    def test_hybrid_chunker_different_providers_same_content(self, mock_factory, sample_markdown_content):
        """Test that different providers can process the same content"""
        # Test with OpenAI provider
        mock_openai = Mock(spec=OpenAIProvider)
        mock_openai.count_tokens.return_value = 100
        mock_openai.provider_name = "openai"
        mock_openai.model = "gpt-3.5-turbo"
        
        mock_factory.create_provider.return_value = mock_openai
        chunker_openai = HybridMarkdownChunker(chunk_size=400, chunk_overlap=50)
        chunks_openai = chunker_openai.chunk_document(sample_markdown_content)
        
        # Test with Jina provider
        mock_jina = Mock(spec=JinaProvider)
        mock_jina.count_tokens.return_value = 80
        mock_jina.provider_name = "jina"
        mock_jina.model = "jina-embeddings-v2-base-en"
        
        mock_factory.create_provider.return_value = mock_jina
        chunker_jina = HybridMarkdownChunker(chunk_size=400, chunk_overlap=50)
        chunks_jina = chunker_jina.chunk_document(sample_markdown_content)
        
        # Both should produce chunks
        assert len(chunks_openai) > 0
        assert len(chunks_jina) > 0
        
        # Content should be similar but token counts different
        assert chunks_openai[0].page_content == chunks_jina[0].page_content
        assert chunks_openai[0].metadata['chunk_tokens'] == 100
        assert chunks_jina[0].metadata['chunk_tokens'] == 80
    
    @patch('src.llm.factory.LLMFactory')
    def test_hybrid_chunker_provider_configuration_changes(self, mock_factory):
        """Test that chunker respects provider configuration changes"""
        # First provider
        mock_provider1 = Mock(spec=OpenAIProvider)
        mock_provider1.count_tokens.return_value = 50
        mock_provider1.provider_name = "openai"
        mock_provider1.model = "gpt-3.5-turbo"
        
        mock_factory.create_provider.return_value = mock_provider1
        chunker1 = HybridMarkdownChunker(chunk_size=200, chunk_overlap=50)
        
        assert chunker1.llm_provider == mock_provider1
        assert chunker1._token_length("test") == 50
        
        # Second provider (simulating config change)
        mock_provider2 = Mock(spec=JinaProvider)
        mock_provider2.count_tokens.return_value = 30
        mock_provider2.provider_name = "jina"
        mock_provider2.model = "jina-embeddings-v2-base-en"
        
        mock_factory.create_provider.return_value = mock_provider2
        chunker2 = HybridMarkdownChunker(chunk_size=200, chunk_overlap=50)
        
        assert chunker2.llm_provider == mock_provider2
        assert chunker2._token_length("test") == 30
    
    @patch('src.llm.factory.LLMFactory')
    def test_hybrid_chunker_with_custom_provider_kwargs(self, mock_factory):
        """Test HybridChunker with custom provider configuration"""
        mock_provider = Mock(spec=JinaProvider)
        mock_provider.count_tokens.return_value = 60
        mock_provider.provider_name = "jina"
        mock_provider.model = "custom-jina-model"
        
        mock_factory.create_provider.return_value = mock_provider
        
        chunker = HybridMarkdownChunker(chunk_size=200, chunk_overlap=50)
        
        # Verify the provider is used correctly
        assert chunker.llm_provider == mock_provider
        
        # Test that factory was called (would normally pass through kwargs)
        mock_factory.create_provider.assert_called_once()
    
    @patch('src.llm.factory.LLMFactory')
    @patch('src.chunkers.hybrid_chunker.tiktoken')
    def test_complete_fallback_chain(self, mock_tiktoken, mock_factory):
        """Test complete fallback chain: LLM provider → tiktoken model → tiktoken base"""
        # Mock LLM factory to fail
        mock_factory.create_provider.side_effect = Exception("LLM provider failed")
        
        # Mock tiktoken model-specific to fail
        mock_tiktoken.encoding_for_model.side_effect = Exception("Model not found")
        
        # Mock tiktoken base encoding to succeed
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tiktoken.get_encoding.return_value = mock_tokenizer
        
        chunker = HybridMarkdownChunker(chunk_size=200, chunk_overlap=50)
        
        # Verify complete fallback
        assert chunker.llm_provider is None
        assert chunker.tokenizer is not None
        
        # Test token counting uses final fallback
        token_count = chunker._token_length("Test")
        assert token_count == 3
        mock_tiktoken.get_encoding.assert_called_with("cl100k_base")
    
    @patch('src.llm.factory.LLMFactory')
    @patch('src.chunkers.hybrid_chunker.tiktoken')
    def test_complete_failure_scenario(self, mock_tiktoken, mock_factory):
        """Test complete failure scenario when all tokenization methods fail"""
        # Mock all methods to fail
        mock_factory.create_provider.side_effect = Exception("LLM provider failed")
        mock_tiktoken.encoding_for_model.side_effect = Exception("Model not found")
        mock_tiktoken.get_encoding.side_effect = Exception("Base encoding failed")
        
        # Should raise TokenizationError during initialization
        with pytest.raises(TokenizationError, match="Failed to initialize tokenizer or LLM provider"):
            HybridMarkdownChunker(chunk_size=200, chunk_overlap=50)