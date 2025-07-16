import pytest
from unittest.mock import Mock, patch, MagicMock
from src.llm.providers.base import (
    BaseLLMProvider, LLMResponse, EmbeddingResponse, 
    LLMProviderError, TokenizationError, CompletionError, EmbeddingError
)
from src.llm.providers.openai_provider import OpenAIProvider
from src.llm.providers.anthropic_provider import AnthropicProvider
from src.llm.providers.jina_provider import JinaProvider


class TestBaseLLMProvider:
    """Tests for the base LLM provider abstract class"""
    
    def test_llm_response_dataclass(self):
        """Test LLMResponse dataclass structure"""
        response = LLMResponse(
            content="Test response",
            tokens_used=10,
            model="test-model",
            provider="test-provider",
            metadata={"key": "value"}
        )
        
        assert response.content == "Test response"
        assert response.tokens_used == 10
        assert response.model == "test-model"
        assert response.provider == "test-provider"
        assert response.metadata == {"key": "value"}
    
    def test_embedding_response_dataclass(self):
        """Test EmbeddingResponse dataclass structure"""
        response = EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            tokens_used=5,
            model="embedding-model",
            provider="test-provider",
            metadata={"dimension": 3}
        )
        
        assert response.embeddings == [[0.1, 0.2, 0.3]]
        assert response.tokens_used == 5
        assert response.model == "embedding-model"
        assert response.provider == "test-provider"
        assert response.metadata == {"dimension": 3}
    
    def test_base_provider_initialization(self):
        """Test base provider initialization"""
        # Create a concrete implementation for testing
        class TestProvider(BaseLLMProvider):
            @property
            def provider_name(self):
                return "test"
            
            def _initialize_client(self):
                pass
            
            def generate_completion(self, prompt, **kwargs):
                pass
            
            def generate_embeddings(self, texts, **kwargs):
                pass
            
            def count_tokens(self, text):
                return len(text.split())
            
            def get_max_tokens(self):
                return 4096
        
        provider = TestProvider(api_key="test_key", model="test_model", extra_param="value")
        
        assert provider.api_key == "test_key"
        assert provider.model == "test_model"
        assert provider.config == {"extra_param": "value"}
        assert provider._client is None
    
    def test_is_available_with_api_key(self):
        """Test provider availability with API key"""
        class TestProvider(BaseLLMProvider):
            @property
            def provider_name(self):
                return "test"
            
            def _initialize_client(self):
                pass
            
            def generate_completion(self, prompt, **kwargs):
                pass
            
            def generate_embeddings(self, texts, **kwargs):
                pass
            
            def count_tokens(self, text):
                return len(text.split())
            
            def get_max_tokens(self):
                return 4096
        
        provider = TestProvider(api_key="test_key", model="test_model")
        assert provider.is_available() is True
        
        provider_no_key = TestProvider(api_key="", model="test_model")
        assert provider_no_key.is_available() is False


class TestOpenAIProvider:
    """Tests for OpenAI provider"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client"""
        with patch('src.llm.providers.openai_provider.OpenAI') as mock_client:
            yield mock_client
    
    @pytest.fixture
    def mock_tiktoken(self):
        """Mock tiktoken tokenizer"""
        with patch('src.llm.providers.openai_provider.tiktoken') as mock_tiktoken:
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
            yield mock_tiktoken
    
    def test_openai_provider_initialization(self, mock_openai_client, mock_tiktoken):
        """Test OpenAI provider initialization"""
        provider = OpenAIProvider(api_key="test_key", model="gpt-3.5-turbo")
        
        assert provider.api_key == "test_key"
        assert provider.model == "gpt-3.5-turbo"
        assert provider.provider_name == "openai"
        assert provider.embedding_model == "text-embedding-ada-002"
    
    def test_openai_tokenizer_initialization(self, mock_openai_client, mock_tiktoken):
        """Test tokenizer initialization with fallback"""
        provider = OpenAIProvider(api_key="test_key", model="gpt-3.5-turbo")
        
        # Test successful tokenizer initialization
        tokenizer = provider._get_tokenizer()
        mock_tiktoken.encoding_for_model.assert_called_with("gpt-3.5-turbo")
        assert tokenizer is not None
    
    def test_openai_tokenizer_fallback(self, mock_openai_client):
        """Test tokenizer fallback to cl100k_base"""
        with patch('src.llm.providers.openai_provider.tiktoken') as mock_tiktoken:
            mock_tiktoken.encoding_for_model.side_effect = Exception("Model not found")
            
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_tiktoken.get_encoding.return_value = mock_tokenizer
            
            provider = OpenAIProvider(api_key="test_key", model="unknown-model")
            tokenizer = provider._get_tokenizer()
            
            mock_tiktoken.get_encoding.assert_called_with("cl100k_base")
            assert tokenizer is not None
    
    def test_openai_generate_completion(self, mock_openai_client, mock_tiktoken):
        """Test OpenAI completion generation"""
        # Mock the response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 25
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 10
        
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_client.return_value = mock_client_instance
        
        provider = OpenAIProvider(api_key="test_key", model="gpt-3.5-turbo")
        response = provider.generate_completion("Test prompt")
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.tokens_used == 25
        assert response.model == "gpt-3.5-turbo"
        assert response.provider == "openai"
        assert response.metadata["prompt_tokens"] == 15
        assert response.metadata["completion_tokens"] == 10
    
    def test_openai_generate_embeddings(self, mock_openai_client, mock_tiktoken):
        """Test OpenAI embeddings generation"""
        # Mock the response
        mock_response = Mock()
        mock_response.data = [Mock(), Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_response.data[1].embedding = [0.4, 0.5, 0.6]
        mock_response.usage.total_tokens = 20
        
        mock_client_instance = Mock()
        mock_client_instance.embeddings.create.return_value = mock_response
        mock_openai_client.return_value = mock_client_instance
        
        provider = OpenAIProvider(api_key="test_key")
        response = provider.generate_embeddings(["text1", "text2"])
        
        assert isinstance(response, EmbeddingResponse)
        assert response.embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert response.tokens_used == 20
        assert response.model == "text-embedding-ada-002"
        assert response.provider == "openai"
    
    def test_openai_count_tokens(self, mock_openai_client, mock_tiktoken):
        """Test OpenAI token counting"""
        provider = OpenAIProvider(api_key="test_key", model="gpt-3.5-turbo")
        
        token_count = provider.count_tokens("Hello world")
        
        assert token_count == 5  # Mocked to return 5 tokens
        mock_tiktoken.encoding_for_model.assert_called_with("gpt-3.5-turbo")
    
    def test_openai_get_max_tokens(self, mock_openai_client, mock_tiktoken):
        """Test getting max tokens for different models"""
        provider = OpenAIProvider(api_key="test_key", model="gpt-4")
        assert provider.get_max_tokens() == 8192
        
        provider = OpenAIProvider(api_key="test_key", model="gpt-3.5-turbo")
        assert provider.get_max_tokens() == 4096
        
        provider = OpenAIProvider(api_key="test_key", model="unknown-model")
        assert provider.get_max_tokens() == 4096  # Default
    
    def test_openai_completion_error(self, mock_openai_client, mock_tiktoken):
        """Test OpenAI completion error handling"""
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_client.return_value = mock_client_instance
        
        provider = OpenAIProvider(api_key="test_key", model="gpt-3.5-turbo")
        
        with pytest.raises(CompletionError, match="OpenAI completion failed"):
            provider.generate_completion("Test prompt")
    
    def test_openai_embedding_error(self, mock_openai_client, mock_tiktoken):
        """Test OpenAI embedding error handling"""
        mock_client_instance = Mock()
        mock_client_instance.embeddings.create.side_effect = Exception("API Error")
        mock_openai_client.return_value = mock_client_instance
        
        provider = OpenAIProvider(api_key="test_key")
        
        with pytest.raises(EmbeddingError, match="OpenAI embedding failed"):
            provider.generate_embeddings(["text"])


class TestJinaProvider:
    """Tests for Jina AI provider"""
    
    @pytest.fixture
    def mock_requests(self):
        """Mock requests for HTTP calls"""
        with patch('src.llm.providers.jina_provider.requests') as mock_requests:
            yield mock_requests
    
    def test_jina_provider_initialization(self):
        """Test Jina provider initialization"""
        provider = JinaProvider(api_key="test_key", model="jina-embeddings-v2-base-en")
        
        assert provider.api_key == "test_key"
        assert provider.model == "jina-embeddings-v2-base-en"
        assert provider.provider_name == "jina"
        assert provider.base_url == "https://api.jina.ai/v1"
    
    def test_jina_client_initialization(self):
        """Test Jina client initialization"""
        provider = JinaProvider(api_key="test_key")
        client = provider.get_client()
        
        assert client["base_url"] == "https://api.jina.ai/v1"
        assert client["headers"]["Authorization"] == "Bearer test_key"
        assert client["headers"]["Content-Type"] == "application/json"
    
    def test_jina_generate_embeddings(self, mock_requests):
        """Test Jina embeddings generation"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ],
            "usage": {"total_tokens": 15}
        }
        mock_requests.post.return_value = mock_response
        
        provider = JinaProvider(api_key="test_key")
        response = provider.generate_embeddings(["text1", "text2"])
        
        assert isinstance(response, EmbeddingResponse)
        assert response.embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert response.tokens_used == 15
        assert response.provider == "jina"
        
        # Verify API call
        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        assert "embeddings" in call_args[0][0]
        assert call_args[1]["json"]["input"] == ["text1", "text2"]
    
    def test_jina_generate_completion(self, mock_requests):
        """Test Jina completion generation"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"text": "Generated response"}],
            "usage": {"total_tokens": 20}
        }
        mock_requests.post.return_value = mock_response
        
        provider = JinaProvider(api_key="test_key")
        response = provider.generate_completion("Test prompt")
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Generated response"
        assert response.tokens_used == 20
        assert response.provider == "jina"
    
    def test_jina_count_tokens(self):
        """Test Jina token counting (approximation)"""
        provider = JinaProvider(api_key="test_key")
        
        # Test approximation: ~4 characters per token
        token_count = provider.count_tokens("Hello world test")  # 16 chars
        assert token_count == 4  # 16 // 4
    
    def test_jina_get_max_tokens(self):
        """Test getting max tokens for Jina models"""
        provider = JinaProvider(api_key="test_key", model="jina-embeddings-v2-base-en")
        assert provider.get_max_tokens() == 8192
        
        provider = JinaProvider(api_key="test_key", model="unknown-model")
        assert provider.get_max_tokens() == 8192  # Default
    
    def test_jina_embedding_error(self, mock_requests):
        """Test Jina embedding error handling"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_requests.post.return_value = mock_response
        
        provider = JinaProvider(api_key="test_key")
        
        with pytest.raises(EmbeddingError, match="Jina API error: 400"):
            provider.generate_embeddings(["text"])
    
    def test_jina_completion_error(self, mock_requests):
        """Test Jina completion error handling"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_requests.post.return_value = mock_response
        
        provider = JinaProvider(api_key="test_key")
        
        with pytest.raises(CompletionError, match="Jina API error: 401"):
            provider.generate_completion("Test prompt")


class TestAnthropicProvider:
    """Tests for Anthropic provider"""
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client"""
        with patch('src.llm.providers.anthropic_provider.anthropic') as mock_anthropic:
            yield mock_anthropic
    
    def test_anthropic_provider_initialization(self, mock_anthropic_client):
        """Test Anthropic provider initialization"""
        provider = AnthropicProvider(api_key="test_key", model="claude-3-sonnet-20240229")
        
        assert provider.api_key == "test_key"
        assert provider.model == "claude-3-sonnet-20240229"
        assert provider.provider_name == "anthropic"
    
    def test_anthropic_generate_completion(self, mock_anthropic_client):
        """Test Anthropic completion generation"""
        # Mock the response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Claude response"
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 10
        mock_response.stop_reason = "end_turn"
        
        mock_client_instance = Mock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_anthropic_client.Anthropic.return_value = mock_client_instance
        
        provider = AnthropicProvider(api_key="test_key", model="claude-3-sonnet-20240229")
        response = provider.generate_completion("Test prompt")
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Claude response"
        assert response.tokens_used == 25  # 15 + 10
        assert response.model == "claude-3-sonnet-20240229"
        assert response.provider == "anthropic"
        assert response.metadata["input_tokens"] == 15
        assert response.metadata["output_tokens"] == 10
    
    def test_anthropic_embeddings_not_supported(self, mock_anthropic_client):
        """Test that Anthropic embeddings raise appropriate error"""
        provider = AnthropicProvider(api_key="test_key")
        
        with pytest.raises(EmbeddingError, match="Anthropic doesn't provide direct embeddings API"):
            provider.generate_embeddings(["text"])
    
    def test_anthropic_count_tokens(self, mock_anthropic_client):
        """Test Anthropic token counting (approximation)"""
        provider = AnthropicProvider(api_key="test_key")
        
        # Test approximation: ~4 characters per token
        token_count = provider.count_tokens("Hello world test")  # 16 chars
        assert token_count == 4  # 16 // 4
    
    def test_anthropic_get_max_tokens(self, mock_anthropic_client):
        """Test getting max tokens for Anthropic models"""
        provider = AnthropicProvider(api_key="test_key", model="claude-3-sonnet-20240229")
        assert provider.get_max_tokens() == 200000
        
        provider = AnthropicProvider(api_key="test_key", model="unknown-model")
        assert provider.get_max_tokens() == 200000  # Default
    
    def test_anthropic_completion_error(self, mock_anthropic_client):
        """Test Anthropic completion error handling"""
        mock_client_instance = Mock()
        mock_client_instance.messages.create.side_effect = Exception("API Error")
        mock_anthropic_client.Anthropic.return_value = mock_client_instance
        
        provider = AnthropicProvider(api_key="test_key")
        
        with pytest.raises(CompletionError, match="Anthropic completion failed"):
            provider.generate_completion("Test prompt")