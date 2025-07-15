"""
Custom exception classes for the document chunking system.

This module defines specific exception types for different error scenarios
that can occur during document processing and chunking operations.
"""

from typing import Optional, Any, Dict


class ChunkingError(Exception):
    """Base exception for all chunking-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (Details: {detail_str})"
        return self.message


class ConfigurationError(ChunkingError):
    """Raised when there are configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, config_value: Optional[Any] = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = config_value
        super().__init__(message, details)


class ValidationError(ChunkingError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        super().__init__(message, details)


class FileHandlingError(ChunkingError):
    """Raised when file operations fail."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None):
        details = {}
        if file_path:
            details["file_path"] = file_path
        if operation:
            details["operation"] = operation
        super().__init__(message, details)


class ProcessingError(ChunkingError):
    """Raised when document processing fails."""
    
    def __init__(self, message: str, stage: Optional[str] = None, chunk_index: Optional[int] = None):
        details = {}
        if stage:
            details["stage"] = stage
        if chunk_index is not None:
            details["chunk_index"] = chunk_index
        super().__init__(message, details)


class QualityEvaluationError(ChunkingError):
    """Raised when chunk quality evaluation fails."""
    
    def __init__(self, message: str, metric: Optional[str] = None, chunk_count: Optional[int] = None):
        details = {}
        if metric:
            details["metric"] = metric
        if chunk_count is not None:
            details["chunk_count"] = chunk_count
        super().__init__(message, details)


class TokenizationError(ChunkingError):
    """Raised when tokenization operations fail."""
    
    def __init__(self, message: str, model: Optional[str] = None, text_length: Optional[int] = None):
        details = {}
        if model:
            details["model"] = model
        if text_length is not None:
            details["text_length"] = text_length
        super().__init__(message, details)


class MetadataError(ChunkingError):
    """Raised when metadata operations fail."""
    
    def __init__(self, message: str, metadata_key: Optional[str] = None, operation: Optional[str] = None):
        details = {}
        if metadata_key:
            details["metadata_key"] = metadata_key
        if operation:
            details["operation"] = operation
        super().__init__(message, details)


class MemoryError(ChunkingError):
    """Raised when memory constraints are exceeded."""
    
    def __init__(self, message: str, memory_used: Optional[str] = None, operation: Optional[str] = None):
        details = {}
        if memory_used:
            details["memory_used"] = memory_used
        if operation:
            details["operation"] = operation
        super().__init__(message, details)


class BatchProcessingError(ChunkingError):
    """Raised when batch processing operations fail."""
    
    def __init__(self, message: str, batch_size: Optional[int] = None, failed_files: Optional[list] = None):
        details = {}
        if batch_size is not None:
            details["batch_size"] = batch_size
        if failed_files:
            details["failed_files"] = failed_files
        super().__init__(message, details)


class SemanticProcessingError(ChunkingError):
    """Raised when semantic processing operations fail."""
    
    def __init__(self, message: str, vectorizer: Optional[str] = None, similarity_method: Optional[str] = None):
        details = {}
        if vectorizer:
            details["vectorizer"] = vectorizer
        if similarity_method:
            details["similarity_method"] = similarity_method
        super().__init__(message, details)


class SecurityError(ChunkingError):
    """Raised when security validation fails."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, security_check: Optional[str] = None):
        details = {}
        if file_path:
            details["file_path"] = file_path
        if security_check:
            details["security_check"] = security_check
        super().__init__(message, details)