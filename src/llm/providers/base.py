from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standard response format for LLM operations"""
    content: str
    tokens_used: int
    model: str
    provider: str
    metadata: Dict[str, Any] = None


@dataclass
class EmbeddingResponse:
    """Standard response format for embedding operations"""
    embeddings: List[List[float]]
    tokens_used: int
    model: str
    provider: str
    metadata: Dict[str, Any] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.config = kwargs
        self._client = None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider"""
        pass
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the provider-specific client"""
        pass
    
    @abstractmethod
    def generate_completion(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate text completion"""
        pass
    
    @abstractmethod
    def generate_embeddings(
        self, 
        texts: List[str],
        **kwargs
    ) -> EmbeddingResponse:
        """Generate embeddings for texts"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass
    
    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get maximum tokens for the model"""
        pass
    
    def is_available(self) -> bool:
        """Check if the provider is available and configured"""
        return bool(self.api_key)
    
    def get_client(self):
        """Get or initialize the client"""
        if self._client is None:
            self._initialize_client()
        return self._client


class LLMProviderError(Exception):
    """Base exception for LLM provider errors"""
    pass


class TokenizationError(LLMProviderError):
    """Exception raised when tokenization fails"""
    pass


class CompletionError(LLMProviderError):
    """Exception raised when completion generation fails"""
    pass


class EmbeddingError(LLMProviderError):
    """Exception raised when embedding generation fails"""
    pass