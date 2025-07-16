import pytest
from unittest.mock import Mock, patch, MagicMock
from src.llm.factory import LLMFactory
from src.llm.providers import BaseLLMProvider, OpenAIProvider, AnthropicProvider, JinaProvider


class TestLLMFactory:
    """Tests for LLM Factory"""
    
    def test_available_providers_registration(self):
        """Test that all expected providers are registered"""
        expected_providers = ["openai", "anthropic", "jina"]
        
        for provider_name in expected_providers:
            assert provider_name in LLMFactory._providers
    
    def test_provider_classes_correct(self):
        """Test that provider classes are correctly registered"""
        assert LLMFactory._providers["openai"] == OpenAIProvider
        assert LLMFactory._providers["anthropic"] == AnthropicProvider
        assert LLMFactory._providers["jina"] == JinaProvider
    
    @patch('src.llm.factory.config')
    def test_create_openai_provider(self, mock_config):
        """Test creating OpenAI provider"""
        mock_config.LLM_PROVIDER = "openai"
        mock_config.LLM_MODEL = "gpt-3.5-turbo"
        mock_config.OPENAI_API_KEY = "test_openai_key"
        
        with patch('src.llm.providers.openai_provider.OpenAI'):
            with patch('src.llm.providers.openai_provider.tiktoken'):
                provider = LLMFactory.create_provider()
                
                assert isinstance(provider, OpenAIProvider)
                assert provider.api_key == "test_openai_key"
                assert provider.model == "gpt-3.5-turbo"
    
    @patch('src.llm.factory.config')
    def test_create_anthropic_provider(self, mock_config):
        """Test creating Anthropic provider"""
        mock_config.LLM_PROVIDER = "anthropic"
        mock_config.LLM_MODEL = "claude-3-sonnet-20240229"
        mock_config.ANTHROPIC_API_KEY = "test_anthropic_key"
        
        with patch('src.llm.providers.anthropic_provider.anthropic'):
            provider = LLMFactory.create_provider()
            
            assert isinstance(provider, AnthropicProvider)
            assert provider.api_key == "test_anthropic_key"
            assert provider.model == "claude-3-sonnet-20240229"
    
    @patch('src.llm.factory.config')
    def test_create_jina_provider(self, mock_config):
        """Test creating Jina provider"""
        mock_config.LLM_PROVIDER = "jina"
        mock_config.LLM_MODEL = "jina-embeddings-v2-base-en"
        mock_config.JINA_API_KEY = "test_jina_key"
        
        provider = LLMFactory.create_provider()
        
        assert isinstance(provider, JinaProvider)
        assert provider.api_key == "test_jina_key"
        assert provider.model == "jina-embeddings-v2-base-en"
    
    @patch('src.llm.factory.config')
    def test_create_provider_with_overrides(self, mock_config):
        """Test creating provider with parameter overrides"""
        mock_config.LLM_PROVIDER = "openai"
        mock_config.LLM_MODEL = "gpt-3.5-turbo"
        mock_config.OPENAI_API_KEY = "test_key"
        mock_config.JINA_API_KEY = "test_jina_key"
        
        with patch('src.llm.providers.openai_provider.OpenAI'):
            with patch('src.llm.providers.openai_provider.tiktoken'):
                # Override provider and model
                provider = LLMFactory.create_provider(
                    provider_name="jina",
                    model="custom-jina-model"
                )
                
                assert isinstance(provider, JinaProvider)
                assert provider.api_key == "test_jina_key"
                assert provider.model == "custom-jina-model"
    
    @patch('src.llm.factory.config')
    def test_create_provider_with_kwargs(self, mock_config):
        """Test creating provider with additional kwargs"""
        mock_config.LLM_PROVIDER = "jina"
        mock_config.LLM_MODEL = "jina-embeddings-v2-base-en"
        mock_config.JINA_API_KEY = "test_jina_key"
        
        provider = LLMFactory.create_provider(
            base_url="https://custom.jina.ai",
            embedding_model="custom-embedding-model"
        )
        
        assert isinstance(provider, JinaProvider)
        assert provider.config["base_url"] == "https://custom.jina.ai"
        assert provider.config["embedding_model"] == "custom-embedding-model"
    
    def test_create_provider_unknown_provider(self):
        """Test creating provider with unknown provider name"""
        with pytest.raises(ValueError, match="Unknown provider: unknown"):
            LLMFactory.create_provider(provider_name="unknown")
    
    @patch('src.llm.factory.config')
    def test_create_provider_no_api_key(self, mock_config):
        """Test creating provider with no API key"""
        mock_config.LLM_PROVIDER = "openai"
        mock_config.LLM_MODEL = "gpt-3.5-turbo"
        mock_config.OPENAI_API_KEY = ""  # Empty API key
        
        with pytest.raises(ValueError, match="No API key found for provider: openai"):
            LLMFactory.create_provider()
    
    @patch('src.llm.factory.config')
    def test_get_api_key_mapping(self, mock_config):
        """Test API key mapping for different providers"""
        mock_config.OPENAI_API_KEY = "openai_key"
        mock_config.ANTHROPIC_API_KEY = "anthropic_key"
        mock_config.JINA_API_KEY = "jina_key"
        
        assert LLMFactory._get_api_key("openai") == "openai_key"
        assert LLMFactory._get_api_key("anthropic") == "anthropic_key"
        assert LLMFactory._get_api_key("jina") == "jina_key"
        assert LLMFactory._get_api_key("unknown") == ""
    
    def test_register_new_provider(self):
        """Test registering a new provider"""
        class CustomProvider(BaseLLMProvider):
            @property
            def provider_name(self):
                return "custom"
            
            def _initialize_client(self):
                pass
            
            def generate_completion(self, prompt, **kwargs):
                pass
            
            def generate_embeddings(self, texts, **kwargs):
                pass
            
            def count_tokens(self, text):
                return len(text.split())
            
            def get_max_tokens(self):
                return 2048
        
        LLMFactory.register_provider("custom", CustomProvider)
        
        assert "custom" in LLMFactory._providers
        assert LLMFactory._providers["custom"] == CustomProvider
        
        # Clean up
        del LLMFactory._providers["custom"]
    
    @patch('src.llm.factory.config')
    def test_get_available_providers_with_keys(self, mock_config):
        """Test getting available providers with API keys"""
        mock_config.OPENAI_API_KEY = "openai_key"
        mock_config.ANTHROPIC_API_KEY = ""
        mock_config.JINA_API_KEY = "jina_key"
        
        status = LLMFactory.get_available_providers()
        
        assert status["openai"] is True
        assert status["anthropic"] is False
        assert status["jina"] is True
    
    @patch('src.llm.factory.config')
    def test_get_available_providers_no_keys(self, mock_config):
        """Test getting available providers with no API keys"""
        mock_config.OPENAI_API_KEY = ""
        mock_config.ANTHROPIC_API_KEY = ""
        mock_config.JINA_API_KEY = ""
        
        status = LLMFactory.get_available_providers()
        
        assert status["openai"] is False
        assert status["anthropic"] is False
        assert status["jina"] is False
    
    def test_get_available_providers_with_exception(self):
        """Test getting available providers when API key access raises exception"""
        def mock_get_api_key(provider_name):
            if provider_name == "anthropic":
                raise Exception("Config error")
            elif provider_name == "openai":
                return "openai_key"
            elif provider_name == "jina":
                return "jina_key"
            return ""
        
        with patch.object(LLMFactory, '_get_api_key', side_effect=mock_get_api_key):
            status = LLMFactory.get_available_providers()
            
            assert status["openai"] is True
            assert status["anthropic"] is False  # Should handle exception gracefully
            assert status["jina"] is True
    
    @patch('src.llm.factory.config')
    def test_factory_integration_with_config_defaults(self, mock_config):
        """Test factory integration with configuration defaults"""
        mock_config.LLM_PROVIDER = "openai"
        mock_config.LLM_MODEL = "gpt-4"
        mock_config.OPENAI_API_KEY = "test_key"
        
        with patch('src.llm.providers.openai_provider.OpenAI'):
            with patch('src.llm.providers.openai_provider.tiktoken'):
                provider = LLMFactory.create_provider()
                
                assert provider.provider_name == "openai"
                assert provider.model == "gpt-4"
                assert provider.api_key == "test_key"
    
    def test_factory_class_methods_are_classmethods(self):
        """Test that factory methods are class methods"""
        assert hasattr(LLMFactory.create_provider, '__func__')
        assert hasattr(LLMFactory.register_provider, '__func__')
        assert hasattr(LLMFactory.get_available_providers, '__func__')
        assert hasattr(LLMFactory._get_api_key, '__func__')
    
    def test_factory_providers_dict_immutability(self):
        """Test that modifying _providers dict affects factory"""
        original_providers = LLMFactory._providers.copy()
        
        # Add a temporary provider
        LLMFactory._providers["temp"] = Mock
        
        assert "temp" in LLMFactory._providers
        
        # Clean up
        del LLMFactory._providers["temp"]
        
        # Verify original state
        assert LLMFactory._providers == original_providers