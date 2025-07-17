"""
Mock test for DoclingProcessor without actual docling library installed
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


class MockDoclingProcessor:
    """Mock DoclingProcessor for testing purposes."""
    
    def __init__(self, provider=None, chunker_tokenizer="mock-tokenizer"):
        self.provider = provider
        self.chunker_tokenizer = chunker_tokenizer
        
    def process_document(self, file_path, format_type="auto"):
        """Mock process_document method."""
        return [
            Document(
                page_content=f"Mock content from {file_path}",
                metadata={
                    "source": file_path,
                    "format": format_type,
                    "chunk_index": 0,
                    "file_size": 1024,
                    "processing_time": 0.1
                }
            )
        ]
    
    def get_supported_formats(self):
        return ["pdf", "docx", "pptx", "html", "image"]
    
    def get_processor_info(self):
        return {
            "processor_name": "DoclingProcessor",
            "library": "docling",
            "supported_formats": self.get_supported_formats()
        }


def test_mock_docling_processor():
    """Test that the mock DoclingProcessor works as expected."""
    processor = MockDoclingProcessor()
    
    # Test processing a document
    documents = processor.process_document("test.pdf", "pdf")
    
    assert len(documents) == 1
    assert documents[0].page_content == "Mock content from test.pdf"
    assert documents[0].metadata["format"] == "pdf"
    assert documents[0].metadata["source"] == "test.pdf"
    
    # Test processor info
    info = processor.get_processor_info()
    assert info["processor_name"] == "DoclingProcessor"
    assert info["library"] == "docling"
    assert "pdf" in info["supported_formats"]


def test_production_pipeline_with_mock():
    """Test production pipeline with mock DoclingProcessor."""
    import tempfile
    import os
    from src.orchestration.production_pipeline import ProductionPipeline, PipelineConfig
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        temp_file.write(b"Test PDF content")
        test_file_path = temp_file.name
    
    try:
        # Mock the DoclingProcessor import
        with patch('src.chunkers.docling_processor.DoclingProcessor', MockDoclingProcessor):
            config = PipelineConfig(
                max_concurrent_files=2,
                performance_monitoring_enabled=True,
                quality_evaluation_enabled=True,
                error_recovery_enabled=True
            )
            
            pipeline = ProductionPipeline(config)
            
            # Test single document processing
            result = pipeline.process_single_document(test_file_path)
            
            assert result.success
            assert result.format_type == "pdf"
            assert len(result.chunks) == 1
            assert result.chunks[0].page_content == f"Mock content from {test_file_path}"
            
            # Test production readiness validation
            readiness = pipeline.validate_production_readiness()
            assert readiness["readiness_score"] > 0
            assert readiness["checks"]["processing_capability"]
    finally:
        # Clean up the temporary file
        if os.path.exists(test_file_path):
            os.unlink(test_file_path)


if __name__ == "__main__":
    test_mock_docling_processor()
    test_production_pipeline_with_mock()
    print("✅ Mock tests passed! System is ready for actual Docling integration.")