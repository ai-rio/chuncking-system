from typing import Dict, Type, Optional
from src.config.settings import config
from .providers import BaseLLMProvider, OpenAIProvider, AnthropicProvider, JinaProvider


class LLMFactory:
    """Factory class for creating LLM providers"""
    
    _providers: Dict[str, Type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "jina": JinaProvider,
    }
    
    @classmethod
    def create_provider(
        self, 
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """Create LLM provider instance"""
        
        # Use config defaults if not provided
        provider_name = provider_name or config.LLM_PROVIDER
        model = model or config.LLM_MODEL
        
        if provider_name not in self._providers:
            raise ValueError(f"Unknown provider: {provider_name}. Available: {list(self._providers.keys())}")
        
        provider_class = self._providers[provider_name]
        
        # Get API key based on provider
        api_key = self._get_api_key(provider_name)
        
        if not api_key:
            raise ValueError(f"No API key found for provider: {provider_name}")
        
        return provider_class(api_key=api_key, model=model, **kwargs)
    
    @classmethod
    def _get_api_key(cls, provider_name: str) -> str:
        """Get API key for the specified provider"""
        key_mapping = {
            "openai": config.OPENAI_API_KEY,
            "anthropic": config.ANTHROPIC_API_KEY,
            "jina": config.JINA_API_KEY,
        }
        
        return key_mapping.get(provider_name, "")
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLMProvider]):
        """Register a new provider"""
        cls._providers[name] = provider_class
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """Get list of available providers and their status"""
        status = {}
        
        for provider_name in cls._providers.keys():
            try:
                api_key = cls._get_api_key(provider_name)
                status[provider_name] = bool(api_key)
            except Exception:
                status[provider_name] = False
        
        return status