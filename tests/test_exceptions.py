
import pytest
from src.exceptions import (
    ChunkingError,
    ConfigurationError,
    ValidationError,
    FileHandlingError,
    ProcessingError,
    QualityEvaluationError,
    TokenizationError,
    MetadataError,
    MemoryError,
    BatchProcessingError,
    SemanticProcessingError,
    SecurityError
)

def test_chunking_error():
    with pytest.raises(ChunkingError) as e:
        raise ChunkingError("Test message", details={"key": "value"})
    assert "Test message" in str(e.value)
    assert "key=value" in str(e.value)

def test_configuration_error():
    with pytest.raises(ConfigurationError) as e:
        raise ConfigurationError("Config error", config_key="timeout", config_value=30)
    assert "Config error" in str(e.value)
    assert "config_key=timeout" in str(e.value)
    assert "config_value=30" in str(e.value)

def test_validation_error():
    with pytest.raises(ValidationError) as e:
        raise ValidationError("Validation failed", field="username", value="testuser")
    assert "Validation failed" in str(e.value)
    assert "field=username" in str(e.value)
    assert "value=testuser" in str(e.value)

def test_file_handling_error():
    with pytest.raises(FileHandlingError) as e:
        raise FileHandlingError("File not found", file_path="/path/to/file", operation="read")
    assert "File not found" in str(e.value)
    assert "file_path=/path/to/file" in str(e.value)
    assert "operation=read" in str(e.value)

def test_processing_error():
    with pytest.raises(ProcessingError) as e:
        raise ProcessingError("Processing failed", stage="chunking", chunk_index=5)
    assert "Processing failed" in str(e.value)
    assert "stage=chunking" in str(e.value)
    assert "chunk_index=5" in str(e.value)

def test_quality_evaluation_error():
    with pytest.raises(QualityEvaluationError) as e:
        raise QualityEvaluationError("Quality check failed", metric="cohesion", chunk_count=10)
    assert "Quality check failed" in str(e.value)
    assert "metric=cohesion" in str(e.value)
    assert "chunk_count=10" in str(e.value)

def test_tokenization_error():
    with pytest.raises(TokenizationError) as e:
        raise TokenizationError("Tokenization failed", model="bert-base-uncased", text_length=1000)
    assert "Tokenization failed" in str(e.value)
    assert "model=bert-base-uncased" in str(e.value)
    assert "text_length=1000" in str(e.value)

def test_metadata_error():
    with pytest.raises(MetadataError) as e:
        raise MetadataError("Metadata error", metadata_key="author", operation="extract")
    assert "Metadata error" in str(e.value)
    assert "metadata_key=author" in str(e.value)
    assert "operation=extract" in str(e.value)

def test_memory_error():
    with pytest.raises(MemoryError) as e:
        raise MemoryError("Out of memory", memory_used="1GB", operation="chunking")
    assert "Out of memory" in str(e.value)
    assert "memory_used=1GB" in str(e.value)
    assert "operation=chunking" in str(e.value)

def test_batch_processing_error():
    with pytest.raises(BatchProcessingError) as e:
        raise BatchProcessingError("Batch failed", batch_size=10, failed_files=["file1.txt"])
    assert "Batch failed" in str(e.value)
    assert "batch_size=10" in str(e.value)
    assert "failed_files=['file1.txt']" in str(e.value)

def test_semantic_processing_error():
    with pytest.raises(SemanticProcessingError) as e:
        raise SemanticProcessingError("Semantic error", vectorizer="tfidf", similarity_method="cosine")
    assert "Semantic error" in str(e.value)
    assert "vectorizer=tfidf" in str(e.value)
    assert "similarity_method=cosine" in str(e.value)

def test_security_error():
    with pytest.raises(SecurityError) as e:
        raise SecurityError("Security violation", file_path="/path/to/file", security_check="xss")
    assert "Security violation" in str(e.value)
    assert "file_path=/path/to/file" in str(e.value)
    assert "security_check=xss" in str(e.value)
