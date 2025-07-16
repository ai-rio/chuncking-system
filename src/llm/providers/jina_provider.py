from typing import List, Optional, Dict, Any
import requests
import json

from .base import BaseLLMProvider, LLMResponse, EmbeddingResponse, TokenizationError, CompletionError, EmbeddingError


class JinaProvider(BaseLLMProvider):
    """Jina AI provider implementation"""
    
    def __init__(self, api_key: str, model: str = "jina-embeddings-v2-base-en", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.base_url = kwargs.get("base_url", "https://api.jina.ai/v1")
        self.embedding_model = kwargs.get("embedding_model", "jina-embeddings-v2-base-en")
    
    @property
    def provider_name(self) -> str:
        return "jina"
    
    def _initialize_client(self) -> None:
        """Initialize Jina client (using requests)"""
        self._client = {
            "base_url": self.base_url,
            "headers": {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        }
    
    def generate_completion(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate text completion using Jina (if available)"""
        # Note: Jina is primarily known for embeddings, but may have completion models
        try:
            client = self.get_client()
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens or 1024,
                "temperature": temperature,
                **kwargs
            }
            
            response = requests.post(
                f"{client['base_url']}/completions",
                headers=client["headers"],
                json=payload
            )
            
            if response.status_code != 200:
                raise CompletionError(f"Jina API error: {response.status_code} - {response.text}")
            
            data = response.json()
            
            return LLMResponse(
                content=data["choices"][0]["text"],
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                model=self.model,
                provider=self.provider_name,
                metadata=data.get("usage", {})
            )
        except Exception as e:
            raise CompletionError(f"Jina completion failed: {str(e)}")
    
    def generate_embeddings(
        self, 
        texts: List[str],
        **kwargs
    ) -> EmbeddingResponse:
        """Generate embeddings using Jina"""
        try:
            client = self.get_client()
            
            payload = {
                "model": self.embedding_model,
                "input": texts,
                **kwargs
            }
            
            response = requests.post(
                f"{client['base_url']}/embeddings",
                headers=client["headers"],
                json=payload
            )
            
            if response.status_code != 200:
                raise EmbeddingError(f"Jina API error: {response.status_code} - {response.text}")
            
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            
            return EmbeddingResponse(
                embeddings=embeddings,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                model=self.embedding_model,
                provider=self.provider_name,
                metadata={
                    "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                    "usage": data.get("usage", {})
                }
            )
        except Exception as e:
            raise EmbeddingError(f"Jina embedding failed: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens (approximate for Jina)"""
        # Jina doesn't provide direct tokenization, so we approximate
        # This is a rough estimate: ~4 characters per token
        return len(text) // 4
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens for the model"""
        token_limits = {
            "jina-embeddings-v2-base-en": 8192,
            "jina-embeddings-v2-small-en": 8192,
            "jina-reranker-v1-base-en": 8192,
            "jina-reranker-v1-turbo-en": 8192
        }
        return token_limits.get(self.model, 8192)