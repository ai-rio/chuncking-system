"""
Integration tests for the chunking system.
Tests end-to-end workflows and component interactions.
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, Mock

from langchain_core.documents import Document

# Import components to test integration
from src.chunkers.hybrid_chunker import HybridMarkdownChunker
from src.chunkers.evaluators import ChunkQualityEvaluator
from src.utils.file_handler import FileHandler
from src.utils.metadata_enricher import MetadataEnricher


class TestIntegration:
    """Integration tests for the chunking system."""

    def test_end_to_end_document_processing(self, temp_dir, sample_markdown_file):
        """Test complete document processing workflow."""
        # Initialize components
        chunker = HybridMarkdownChunker(chunk_size=200, chunk_overlap=50)
        evaluator = ChunkQualityEvaluator()
        
        # Read document
        content = sample_markdown_file.read_text()
        metadata = {
            'source_file': str(sample_markdown_file),
            'file_size': len(content),
            'book_title': 'Test Document'
        }
        
        # Chunk document
        chunks = chunker.chunk_document(content, metadata)
        
        # Enrich metadata
        enriched_chunks = []
        for chunk in chunks:
            enriched_chunk = MetadataEnricher.enrich_chunk(chunk, metadata)
            enriched_chunks.append(enriched_chunk)
        
        # Evaluate quality
        quality_metrics = evaluator.evaluate_chunks(enriched_chunks)
        
        # Save chunks
        output_path = temp_dir / "integrated_chunks.json"
        FileHandler.save_chunks(enriched_chunks, str(output_path), 'json')
        
        # Assertions
        assert len(enriched_chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in enriched_chunks)
        assert quality_metrics['overall_score'] >= 0
        assert output_path.exists()
        
        # Verify metadata enrichment worked
        for chunk in enriched_chunks:
            assert 'chunk_id' in chunk.metadata
            assert 'processed_at' in chunk.metadata
            assert 'source_file' in chunk.metadata
            assert chunk.metadata['source_file'] == str(sample_markdown_file)

    def test_chunker_evaluator_integration(self, sample_chunks):
        """Test integration between chunker and evaluator."""
        chunker = HybridMarkdownChunker(chunk_size=200, chunk_overlap=50)
        evaluator = ChunkQualityEvaluator()
        
        # Process content through chunker
        content = "# Test Header\n\nThis is test content for integration testing. " * 20
        chunks = chunker.chunk_document(content)
        
        # Evaluate the chunks
        metrics = evaluator.evaluate_chunks(chunks)
        
        assert 'total_chunks' in metrics
        assert metrics['total_chunks'] == len(chunks)
        assert 'overall_score' in metrics
        assert 0 <= metrics['overall_score'] <= 100

    def test_file_handler_chunker_integration(self, temp_dir):
        """Test integration between file handler and chunker."""
        # Create test files
        file1 = temp_dir / "test1.md"
        file2 = temp_dir / "test2.md"
        file1.write_text("# Document 1\nContent for document 1")
        file2.write_text("# Document 2\nContent for document 2")
        
        # Find files
        files = FileHandler.find_markdown_files(str(temp_dir))
        
        # Process files with chunker
        chunker = HybridMarkdownChunker()
        all_chunks = []
        
        for file_path in files:
            with open(file_path, 'r') as f:
                content = f.read()
            chunks = chunker.chunk_document(content, {'source': file_path})
            all_chunks.extend(chunks)
        
        # Save and reload
        output_path = temp_dir / "all_chunks.json"
        FileHandler.save_chunks(all_chunks, str(output_path), 'json')
        loaded_chunks = FileHandler.load_chunks(str(output_path))
        
        assert len(loaded_chunks) == len(all_chunks)
        assert all(isinstance(chunk, Document) for chunk in loaded_chunks)

    def test_metadata_enricher_integration(self, sample_chunks):
        """Test metadata enricher integration with other components."""
        evaluator = ChunkQualityEvaluator()
        
        # Enrich metadata for chunks
        enriched_chunks = []
        document_info = {'source_file': 'test.md', 'author': 'Test Author'}
        
        for chunk in sample_chunks:
            enriched = MetadataEnricher.enrich_chunk(chunk, document_info)
            enriched_chunks.append(enriched)
        
        # Evaluate enriched chunks
        metrics = evaluator.evaluate_chunks(enriched_chunks)
        
        # Verify enrichment worked
        for chunk in enriched_chunks:
            assert 'chunk_id' in chunk.metadata
            assert 'source_file' in chunk.metadata
            assert chunk.metadata['source_file'] == 'test.md'
        
        # Verify evaluation still works
        assert 'overall_score' in metrics

    def test_batch_processing_integration(self, temp_dir):
        """Test batch processing integration."""
        # Create multiple test files
        files = []
        for i in range(5):
            file_path = temp_dir / f"batch_test_{i}.md"
            content = f"# Document {i}\n\nContent for document {i}. " * 10
            file_path.write_text(content)
            files.append(str(file_path))
        
        # Batch process
        chunker = HybridMarkdownChunker(chunk_size=150, chunk_overlap=50)
        evaluator = ChunkQualityEvaluator()
        
        # Process files
        progress_calls = []
        def progress_callback(current, total, filename):
            progress_calls.append((current, total, filename))
        
        results = chunker.batch_process_files(files, progress_callback)
        
        # Evaluate all chunks
        all_chunks = []
        for file_path, chunks in results.items():
            all_chunks.extend(chunks)
        
        overall_metrics = evaluator.evaluate_chunks(all_chunks)
        
        # Assertions
        assert len(results) == 5
        assert len(progress_calls) == 5
        assert len(all_chunks) > 0
        assert 'overall_score' in overall_metrics

    def test_different_output_formats_integration(self, temp_dir, sample_chunks):
        """Test integration with different output formats."""
        # Test all supported formats
        formats = ['json', 'csv', 'pickle']
        
        for fmt in formats:
            output_path = temp_dir / f"chunks.{fmt}"
            
            # Save chunks
            FileHandler.save_chunks(sample_chunks, str(output_path), fmt)
            assert output_path.exists()
            
            # Load back (only for json and pickle)
            if fmt in ['json', 'pickle']:
                loaded_chunks = FileHandler.load_chunks(str(output_path))
                assert len(loaded_chunks) == len(sample_chunks)

    def test_quality_evaluation_workflow(self, temp_dir):
        """Test complete quality evaluation workflow."""
        # Create document with various quality issues
        content = """
# Good Header

This is good content with proper structure.

## Another Header

More good content here.

```python
def code_example():
    return "good code"
```

## Issues Section

a  # Very short content
"""
        
        # Process through full pipeline
        chunker = HybridMarkdownChunker(chunk_size=200, chunk_overlap=50)
        evaluator = ChunkQualityEvaluator()
        
        chunks = chunker.chunk_document(content)
        metrics = evaluator.evaluate_chunks(chunks)
        
        # Generate report
        report_path = temp_dir / "quality_report.md"
        report = evaluator.generate_report(chunks, str(report_path))
        
        # Verify report generation
        assert report_path.exists()
        assert "# Chunk Quality Evaluation Report" in report
        assert "## Recommendations" in report

    def test_error_handling_integration(self, temp_dir):
        """Test error handling across components."""
        chunker = HybridMarkdownChunker()
        evaluator = ChunkQualityEvaluator()
        
        # Test with empty content
        empty_chunks = chunker.chunk_document("")
        empty_metrics = evaluator.evaluate_chunks(empty_chunks)
        
        assert empty_chunks == []
        assert 'error' in empty_metrics
        
        # Test with invalid file operations
        invalid_path = temp_dir / "nonexistent" / "file.json"
        chunks = [Document(page_content="test", metadata={})]
        
        # Should create directories
        FileHandler.save_chunks(chunks, str(invalid_path), 'json')
        assert invalid_path.exists()

    def test_memory_cleanup_integration(self, temp_dir):
        """Test memory cleanup during batch processing."""
        # Create many small files to trigger cleanup
        files = []
        for i in range(15):  # More than BATCH_SIZE
            file_path = temp_dir / f"memory_test_{i}.md"
            file_path.write_text(f"# File {i}\nContent {i}")
            files.append(str(file_path))
        
        chunker = HybridMarkdownChunker()
        
        with patch('gc.collect') as mock_gc:
            results = chunker.batch_process_files(files)
            
            # Should have called gc.collect due to batch size
            assert mock_gc.called
            assert len(results) == 15

    def test_configuration_integration(self):
        """Test that configuration affects all components."""
        from src.config.settings import config
        
        # Test chunker respects config
        chunker = HybridMarkdownChunker()
        assert chunker.chunk_size == config.DEFAULT_CHUNK_SIZE
        assert chunker.chunk_overlap == config.DEFAULT_CHUNK_OVERLAP
        
        # Test custom config
        custom_chunker = HybridMarkdownChunker(chunk_size=500, chunk_overlap=100)
        assert custom_chunker.chunk_size == 500
        assert custom_chunker.chunk_overlap == 100

    def test_unicode_content_integration(self, temp_dir):
        """Test handling of unicode content across components."""
        unicode_content = """
# 测试文档 (Test Document)

这是一个包含中文的测试文档。This is a test document with Chinese characters.

## Section avec français

Contenu en français avec des caractères spéciaux: été, naïve, œuvre.

## セクション（日本語）

日本語のコンテンツです。

🌍 Emoji content: 🎉 🚀 ✨
"""
        
        # Full pipeline test
        chunker = HybridMarkdownChunker(chunk_size=200)
        evaluator = ChunkQualityEvaluator()
        
        chunks = chunker.chunk_document(unicode_content)
        enriched_chunks = [
            MetadataEnricher.enrich_chunk(chunk) for chunk in chunks
        ]
        
        # Save and load
        output_path = temp_dir / "unicode_chunks.json"
        FileHandler.save_chunks(enriched_chunks, str(output_path), 'json')
        loaded_chunks = FileHandler.load_chunks(str(output_path))
        
        # Evaluate
        metrics = evaluator.evaluate_chunks(loaded_chunks)
        
        # Verify unicode preservation
        assert any("测试文档" in chunk.page_content for chunk in loaded_chunks)
        assert any("français" in chunk.page_content for chunk in loaded_chunks)
        assert any("🌍" in chunk.page_content for chunk in loaded_chunks)
        assert 'overall_score' in metrics

    def test_large_document_integration(self, temp_dir):
        """Test integration with large document processing."""
        # Create a large document
        large_content = """
# Large Document Test

This is a test of processing a large document that will generate many chunks.
""" + "\n\n".join([f"## Section {i}\n\nContent for section {i}. " + "Text content. " * 50 for i in range(20)])
        
        large_file = temp_dir / "large_document.md"
        large_file.write_text(large_content)
        
        # Process through full pipeline
        chunker = HybridMarkdownChunker(chunk_size=300)
        evaluator = ChunkQualityEvaluator()
        
        chunks = chunker.chunk_document(large_content, {'source': str(large_file)})
        metrics = evaluator.evaluate_chunks(chunks)
        
        # Save in different formats
        json_path = temp_dir / "large_chunks.json"
        csv_path = temp_dir / "large_chunks.csv"
        
        FileHandler.save_chunks(chunks, str(json_path), 'json')
        FileHandler.save_chunks(chunks, str(csv_path), 'csv')
        
        # Verify processing
        assert len(chunks) > 10  # Should generate many chunks
        assert metrics['overall_score'] > 0
        assert json_path.exists()
        assert csv_path.exists()

    def test_component_compatibility(self):
        """Test that all components work together with consistent interfaces."""
        # Test Document object compatibility
        test_doc = Document(
            page_content="Test content",
            metadata={"test": "value"}
        )
        
        # All components should accept Document objects
        enriched = MetadataEnricher.enrich_chunk(test_doc)
        assert isinstance(enriched, Document)
        
        # Evaluator should accept list of Documents
        evaluator = ChunkQualityEvaluator()
        metrics = evaluator.evaluate_chunks([enriched])
        assert isinstance(metrics, dict)
        
        # FileHandler should work with Documents
        temp_file = "/tmp/test_compatibility.json"
        try:
            FileHandler.save_chunks([enriched], temp_file, 'json')
            loaded = FileHandler.load_chunks(temp_file)
            assert isinstance(loaded[0], Document)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)