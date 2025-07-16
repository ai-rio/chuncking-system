from typing import List, Optional, Dict, Any

from .base import BaseLLMProvider, LLMResponse, EmbeddingResponse, TokenizationError, CompletionError, EmbeddingError

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.embedding_model = kwargs.get("embedding_model", "text-embedding-ada-002")
        self._tokenizer = None
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def _initialize_client(self) -> None:
        """Initialize OpenAI client"""
        if OpenAI is None:
            raise CompletionError("OpenAI package not installed. Install with: pip install openai")
        try:
            self._client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise CompletionError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _get_tokenizer(self):
        """Get or initialize tokenizer"""
        if tiktoken is None:
            raise TokenizationError("tiktoken package not installed. Install with: pip install tiktoken")
        if self._tokenizer is None:
            try:
                self._tokenizer = tiktoken.encoding_for_model(self.model)
            except Exception:
                try:
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
                except Exception as e:
                    raise TokenizationError(f"Failed to initialize tokenizer: {str(e)}")
        return self._tokenizer
    
    def generate_completion(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate text completion using OpenAI"""
        try:
            client = self.get_client()
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                tokens_used=response.usage.total_tokens,
                model=self.model,
                provider=self.provider_name,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "finish_reason": response.choices[0].finish_reason
                }
            )
        except Exception as e:
            raise CompletionError(f"OpenAI completion failed: {str(e)}")
    
    def generate_embeddings(
        self, 
        texts: List[str],
        **kwargs
    ) -> EmbeddingResponse:
        """Generate embeddings using OpenAI"""
        try:
            client = self.get_client()
            
            response = client.embeddings.create(
                model=self.embedding_model,
                input=texts,
                **kwargs
            )
            
            embeddings = [embedding.embedding for embedding in response.data]
            
            return EmbeddingResponse(
                embeddings=embeddings,
                tokens_used=response.usage.total_tokens,
                model=self.embedding_model,
                provider=self.provider_name,
                metadata={
                    "embedding_dimension": len(embeddings[0]) if embeddings else 0
                }
            )
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        try:
            tokenizer = self._get_tokenizer()
            return len(tokenizer.encode(text))
        except Exception as e:
            raise TokenizationError(f"Token counting failed: {str(e)}")
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens for the model"""
        token_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000
        }
        return token_limits.get(self.model, 4096)