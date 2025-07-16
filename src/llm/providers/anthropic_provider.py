from typing import List, Optional, Dict, Any

from .base import BaseLLMProvider, LLMResponse, EmbeddingResponse, TokenizationError, CompletionError, EmbeddingError

try:
    import anthropic
except ImportError:
    anthropic = None


class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider implementation"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", **kwargs):
        super().__init__(api_key, model, **kwargs)
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    def _initialize_client(self) -> None:
        """Initialize Anthropic client"""
        if anthropic is None:
            raise CompletionError("Anthropic package not installed. Install with: pip install anthropic")
        try:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        except Exception as e:
            raise CompletionError(f"Failed to initialize Anthropic client: {str(e)}")
    
    def generate_completion(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate text completion using Anthropic"""
        try:
            client = self.get_client()
            
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            return LLMResponse(
                content=response.content[0].text,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                model=self.model,
                provider=self.provider_name,
                metadata={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "stop_reason": response.stop_reason
                }
            )
        except Exception as e:
            raise CompletionError(f"Anthropic completion failed: {str(e)}")
    
    def generate_embeddings(
        self, 
        texts: List[str],
        **kwargs
    ) -> EmbeddingResponse:
        """Generate embeddings using Anthropic (not directly supported)"""
        # Anthropic doesn't provide embeddings API directly
        # This could be extended to use a different embedding model
        raise EmbeddingError("Anthropic doesn't provide direct embeddings API")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Anthropic's token counting"""
        try:
            client = self.get_client()
            # Anthropic doesn't have a direct token counting API
            # We'll use an approximation: ~4 characters per token
            return len(text) // 4
        except Exception as e:
            raise TokenizationError(f"Token counting failed: {str(e)}")
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens for the model"""
        token_limits = {
            "claude-3-sonnet-20240229": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-3-5-sonnet-20240620": 200000,
            "claude-3-5-haiku-20241022": 200000
        }
        return token_limits.get(self.model, 200000)