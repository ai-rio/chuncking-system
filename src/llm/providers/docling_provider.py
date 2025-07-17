from typing import List, Optional, Dict, Any
import requests
import json
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.document import ConversionResult
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    DocumentConverter = None
    ConversionResult = None

from .base import (
    BaseLLMProvider, 
    LLMResponse, 
    EmbeddingResponse, 
    LLMProviderError,
    CompletionError,
    EmbeddingError
)


class DoclingProvider(BaseLLMProvider):
    """
    Docling AI provider implementation for multi-format document processing.
    
    Integrates Docling's advanced document AI capabilities through LLM provider interface.
    Supports text completion, embeddings, and document processing workflows.
    """
    
    def __init__(self, api_key: str = None, model: str = "docling-v1", **kwargs):
        """
        Initialize DoclingProvider.
        
        Args:
            api_key: Optional API key (not needed for local docling)
            model: Model name to use (default: docling-v1)
            **kwargs: Additional configuration
        """
        super().__init__(api_key or "local", model, **kwargs)
        
        if not DOCLING_AVAILABLE:
            raise LLMProviderError("Docling library not available. Install with: pip install docling")
        
        # Set base URL for API calls
        self.base_url = kwargs.get("base_url", "https://api.docling.ai/v1")
        self.embedding_model = kwargs.get("embedding_model", "docling-embeddings-v1")
        
        # Initialize DocumentConverter
        self.converter = DocumentConverter()
        
        # Store configuration for compatibility
        self.config["docling_available"] = DOCLING_AVAILABLE
        self.config["base_url"] = self.base_url
    
    @property
    def provider_name(self) -> str:
        """Return the name of the provider"""
        return "docling"
    
    def _initialize_client(self) -> None:
        """Initialize the Docling client configuration"""
        self._client = {
            "base_url": self.base_url,
            "headers": {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "docling-chunking-system/1.0"
            }
        }
    
    def generate_completion(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text completion using Docling API.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional completion parameters
            
        Returns:
            LLMResponse containing generated text and metadata
            
        Raises:
            CompletionError: If completion generation fails
        """
        try:
            client = self.get_client()
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens or 1024,
                "temperature": temperature,
                **kwargs
            }
            
            response = requests.post(
                f"{client['base_url']}/chat/completions",
                headers=client["headers"],
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise CompletionError(
                    f"Docling API error: {response.status_code} - {response.text}"
                )
            
            data = response.json()
            
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                model=self.model,
                provider=self.provider_name,
                metadata={
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "usage": data.get("usage", {})
                }
            )
            
        except (OSError, IOError) as e:
            # Catch network-related errors (RequestException inherits from OSError)
            raise CompletionError(f"Docling completion request failed: {str(e)}")
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            raise CompletionError(f"Docling completion response parsing failed: {str(e)}")
        except Exception as e:
            raise CompletionError(f"Docling completion failed: {str(e)}")
    
    def generate_embeddings(
        self, 
        texts: List[str],
        **kwargs
    ) -> EmbeddingResponse:
        """
        Generate embeddings using Docling API.
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional embedding parameters
            
        Returns:
            EmbeddingResponse containing embeddings and metadata
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
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
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise EmbeddingError(
                    f"Docling API error: {response.status_code} - {response.text}"
                )
            
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            
            return EmbeddingResponse(
                embeddings=embeddings,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                model=self.embedding_model,
                provider=self.provider_name,
                metadata={
                    "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                    "usage": data.get("usage", {}),
                    "model_info": data.get("model", {})
                }
            )
            
        except (OSError, IOError) as e:
            # Catch network-related errors (RequestException inherits from OSError)
            raise EmbeddingError(f"Docling embedding request failed: {str(e)}")
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            raise EmbeddingError(f"Docling embedding response parsing failed: {str(e)}")
        except Exception as e:
            raise EmbeddingError(f"Docling embedding failed: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate for Docling).
        
        Uses a simple heuristic since Docling doesn't provide direct tokenization.
        This is an approximation based on average token-to-character ratios.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate number of tokens
        """
        if not text or not text.strip():
            return 0
        
        # Rough approximation: ~4 characters per token for English text
        # This is conservative and should be close to most tokenizers
        char_count = len(text)
        
        # Account for whitespace and punctuation differently
        word_count = len(text.split())
        
        # Use character-based estimate with word-based lower bound
        char_based_estimate = max(1, char_count // 4)
        word_based_estimate = max(1, int(word_count * 1.3))  # ~1.3 tokens per word
        
        # Return the more conservative (higher) estimate
        return max(char_based_estimate, word_based_estimate)
    
    def get_max_tokens(self) -> int:
        """
        Get maximum tokens for the model.
        
        Returns:
            Maximum token limit for the current model
        """
        # Token limits for different Docling models
        # These are reasonable defaults that can be adjusted based on actual API limits
        token_limits = {
            "docling-v1": 8192,
            "docling-large": 16384,
            "docling-small": 4096,
            "docling-embeddings-v1": 8192,
            "docling-document-processor": 32768,
        }
        
        return token_limits.get(self.model, 8192)  # Default to 8K tokens
    
    def is_available(self) -> bool:
        """
        Check if the provider is available and configured.
        
        Returns:
            True if provider has valid configuration, False otherwise
        """
        # For DoclingProvider, we need a real API key, not the default "local"
        return bool(self.api_key and self.api_key.strip() and self.api_key != "local")
    
    def process_document(
        self, 
        file_path: str, 
        document_type: str = "auto",
        **kwargs
    ) -> ConversionResult:
        """
        Process document using Docling DocumentConverter.
        
        This method provides document-specific processing that leverages
        Docling's advanced document understanding capabilities.
        
        Args:
            file_path: Path to the document file
            document_type: Type of document (pdf, docx, html, etc.) or "auto"
            **kwargs: Additional processing parameters
            
        Returns:
            ConversionResult containing the processed document
            
        Note:
            This is a specialized method for document processing workflows.
            It extends beyond standard LLM provider interface for Docling-specific features.
        """
        try:
            if not DOCLING_AVAILABLE:
                raise LLMProviderError("Docling library not available")
            
            # Convert file path to Path object
            doc_path = Path(file_path)
            
            if not doc_path.exists():
                raise LLMProviderError(f"Document file not found: {file_path}")
            
            # Use DocumentConverter to process the document
            result = self.converter.convert(doc_path)
            
            return result
            
        except Exception as e:
            raise LLMProviderError(f"Docling document processing failed: {str(e)}")