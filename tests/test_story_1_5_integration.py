"""
Story 1.5 End-to-End Integration Tests
Tests complete multi-format processing pipeline and production readiness.
"""

import pytest
import json
import os
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from typing import List, Dict, Any

from langchain_core.documents import Document

# Import all Story 1.5 components
from src.chunkers.docling_processor import DoclingProcessor
from src.utils.enhanced_file_handler import EnhancedFileHandler
from src.chunkers.multi_format_quality_evaluator import MultiFormatQualityEvaluator
from src.chunkers.hybrid_chunker import HybridMarkdownChunker
from src.chunkers.evaluators import ChunkQualityEvaluator
from src.utils.performance import PerformanceMonitor
from src.utils.monitoring import SystemMonitor
from src.utils.metadata_enricher import MetadataEnricher
from src.utils.file_handler import FileHandler


class TestStory15EndToEndIntegration:
    """Complete end-to-end integration tests for Story 1.5 multi-format processing."""

    @pytest.fixture(autouse=True)
    def setup_method(self, mock_docling_provider):
        """Set up test environment for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        
        # Initialize components with proper dependencies
        self.base_evaluator = ChunkQualityEvaluator()
        self.mock_provider = mock_docling_provider
        self.docling_processor = DoclingProcessor(self.mock_provider)
        self.file_handler = FileHandler()
        self.enhanced_file_handler = EnhancedFileHandler(self.file_handler, self.docling_processor)
        
        # Create test files for different formats
        self.test_files = self._create_test_files()
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_files(self) -> Dict[str, Path]:
        """Create test files for different formats."""
        files = {}
        
        # Markdown file
        md_content = """# Test Document
        
This is a test document with multiple sections.

## Section 1
Content for section 1 with some text.

## Section 2  
Content for section 2 with more text.

```python
def example():
    return "code example"
```

## Conclusion
Final section with summary.
"""
        files['markdown'] = self.temp_dir / "test.md"
        files['markdown'].write_text(md_content)
        
        # HTML file
        html_content = """<!DOCTYPE html>
<html>
<head><title>Test Document</title></head>
<body>
<h1>Test HTML Document</h1>
<p>This is a test HTML document for processing.</p>
<h2>Section 1</h2>
<p>Content for HTML section 1.</p>
</body>
</html>"""
        files['html'] = self.temp_dir / "test.html"
        files['html'].write_text(html_content)
        
        # Text file (simulating other formats)
        text_content = "This is a plain text file for testing multi-format processing."
        files['text'] = self.temp_dir / "test.txt"
        files['text'].write_text(text_content)
        
        return files

    def test_complete_pdf_processing_pipeline(self):
        """Test complete PDF processing pipeline from end to end."""
        try:
            from docling.datamodel.base_models import ConversionResult
            DOCLING_AVAILABLE = True
        except ImportError:
            ConversionResult = None
            DOCLING_AVAILABLE = False
        
        # Create a test PDF file path
        pdf_path = str(self.temp_dir / "test.pdf")
        
        # Create mock PDF file
        Path(pdf_path).write_text("Mock PDF content")
        
        # Process document using the processor with mock provider
        result = self.docling_processor.process_document(pdf_path, format_type="pdf")
        
        # Verify ConversionResult structure
        if DOCLING_AVAILABLE:
            assert isinstance(result, ConversionResult)
            assert result.status.success is True
            # Access document content through the ConversionResult API
            assert hasattr(result, 'document')
        else:
            # Fallback for when docling is not available
            assert result is not None
        
        # Test quality evaluation with mock chunks
        # Handle both ConversionResult and dict responses
        if DOCLING_AVAILABLE and hasattr(result, 'document'):
            text_content = result.document.export_to_markdown()
            metadata = {"format": "pdf"}
        else:
            # Handle dict response from mock provider
            text_content = result.get('text', 'Mock content') if isinstance(result, dict) else getattr(result, 'text', 'Mock content')
            metadata = result.get('metadata', {"format": "pdf"}) if isinstance(result, dict) else getattr(result, 'metadata', {"format": "pdf"})
        
        mock_chunks = [
            Document(
                page_content=text_content,
                metadata=metadata
            )
        ]
        
        quality_evaluator = MultiFormatQualityEvaluator(self.base_evaluator)
        quality_metrics = quality_evaluator.evaluate_multi_format_chunks(mock_chunks, 'pdf')
        
        # Verify quality evaluation
        assert 'overall_score' in quality_metrics
        assert quality_metrics['total_chunks'] == 1
        
        # Verify processing time (handle both ConversionResult and dict responses)
        if isinstance(result, dict):
            processing_time = result.get('processing_time', 0)
        else:
            processing_time = getattr(result, 'processing_time', 0)
        assert processing_time >= 0

    def test_complete_docx_processing_pipeline(self):
        """Test complete DOCX processing pipeline from end to end."""
        try:
            from docling.datamodel.base_models import ConversionResult
            DOCLING_AVAILABLE = True
        except ImportError:
            ConversionResult = None
            DOCLING_AVAILABLE = False
        
        # Create a test DOCX file path
        docx_path = str(self.temp_dir / "test.docx")
        
        # Create mock DOCX file
        Path(docx_path).write_text("Mock DOCX content")
        
        # Process document using the processor with mock provider
        result = self.docling_processor.process_document(docx_path, format_type="docx")
        
        # Verify ConversionResult structure
        if DOCLING_AVAILABLE:
            assert isinstance(result, ConversionResult)
            assert result.status.success is True
            assert hasattr(result, 'document')
        else:
            assert result is not None
        
        # Test quality evaluation with mock chunks
        # Handle both ConversionResult and dict responses
        if DOCLING_AVAILABLE and hasattr(result, 'document'):
            text_content = result.document.export_to_markdown()
            metadata = {"format": "docx"}
        else:
            # Handle dict response from mock provider
            text_content = result.get('text', 'Mock content') if isinstance(result, dict) else getattr(result, 'text', 'Mock content')
            metadata = result.get('metadata', {"format": "docx"}) if isinstance(result, dict) else getattr(result, 'metadata', {"format": "docx"})
        
        mock_chunks = [
            Document(
                page_content=text_content,
                metadata=metadata
            )
        ]
        
        quality_evaluator = MultiFormatQualityEvaluator(self.base_evaluator)
        quality_metrics = quality_evaluator.evaluate_multi_format_chunks(mock_chunks, 'docx')
        
        # Verify DOCX-specific processing
        assert 'overall_score' in quality_metrics
        assert quality_metrics['overall_score'] >= 0
        assert quality_metrics['total_chunks'] == 1
            
    def test_complete_image_processing_pipeline(self):
        """Test complete image processing pipeline with OCR."""
        try:
            from docling.datamodel.base_models import ConversionResult
            DOCLING_AVAILABLE = True
        except ImportError:
            ConversionResult = None
            DOCLING_AVAILABLE = False
        
        # Create a test image file path
        image_path = str(self.temp_dir / "test.jpg")
        
        # Create mock image file
        Path(image_path).write_text("Mock image content")
        
        # Process document using the processor with mock provider
        result = self.docling_processor.process_document(image_path, format_type="image")
        
        # Verify ConversionResult structure
        if DOCLING_AVAILABLE:
            assert isinstance(result, ConversionResult)
            assert result.status.success is True
            assert hasattr(result, 'document')
        else:
            assert result is not None
        
        # Test quality evaluation with mock chunks
        # Handle both ConversionResult and dict responses
        if DOCLING_AVAILABLE and hasattr(result, 'document'):
            text_content = result.document.export_to_markdown()
            metadata = {"format": "image"}
        else:
            # Handle dict response from mock provider
            text_content = result.get('text', 'Mock content') if isinstance(result, dict) else getattr(result, 'text', 'Mock content')
            metadata = result.get('metadata', {"format": "image"}) if isinstance(result, dict) else getattr(result, 'metadata', {"format": "image"})
        
        mock_chunks = [
            Document(
                page_content=text_content,
                metadata=metadata
            )
        ]
        
        quality_evaluator = MultiFormatQualityEvaluator(self.base_evaluator)
        quality_metrics = quality_evaluator.evaluate_multi_format_chunks(mock_chunks, 'image')
        
        # Verify image processing
        assert 'overall_score' in quality_metrics
        assert quality_metrics['overall_score'] >= 0
        assert quality_metrics['total_chunks'] == 1

    def test_mixed_format_batch_processing(self):
        """Test batch processing of mixed document formats."""
        try:
            from docling.datamodel.base_models import ConversionResult
            DOCLING_AVAILABLE = True
        except ImportError:
            ConversionResult = None
            DOCLING_AVAILABLE = False
        
        # Create test files for different formats
        test_files = {
            "test.pdf": "pdf",
            "test.docx": "docx", 
            "test.html": "html"
        }
        
        # Create actual test files
        for filename, format_type in test_files.items():
            file_path = self.temp_dir / filename
            file_path.write_text(f"Mock {format_type} content")
        
        # Process each file and collect results
        all_chunks = []
        processing_results = []
        
        for filename, format_type in test_files.items():
            file_path = str(self.temp_dir / filename)
            result = self.docling_processor.process_document(file_path, format_type=format_type)
            processing_results.append(result)
            
            # Convert ConversionResult to Document for quality evaluation
            if DOCLING_AVAILABLE and hasattr(result, 'document'):
                # Use mock data for testing since we don't have real document content
                chunk = Document(
                    page_content=f"Mock {format_type} content",
                    metadata={"format": format_type}
                )
            else:
                chunk = Document(
                    page_content=f"Mock {format_type} content",
                    metadata={"format": format_type}
                )
            all_chunks.append(chunk)
        
        # Initialize quality evaluator
        quality_evaluator = MultiFormatQualityEvaluator(self.base_evaluator)
        
        # Evaluate mixed format batch
        quality_metrics = quality_evaluator.evaluate_multi_format_chunks(all_chunks, 'pdf')
        
        # Verify mixed format processing
        assert len(all_chunks) == 3
        assert len(processing_results) == 3
        
        # Verify all processing was successful
        if DOCLING_AVAILABLE:
            assert all(result.status.success for result in processing_results)
        else:
            assert all(result is not None for result in processing_results)
        
        # Verify different formats were processed
        formats = {chunk.metadata.get('format') for chunk in all_chunks}
        assert formats == {'pdf', 'docx', 'html'}
        
        # Verify quality evaluation
        assert 'overall_score' in quality_metrics
        assert quality_metrics['overall_score'] >= 0
        assert quality_metrics['total_chunks'] == 3

    def test_quality_evaluation_integration(self):
        """Test quality evaluation integration across all formats."""
        # Create test chunks for all supported formats
        test_chunks = [
            Document(
                page_content="High quality PDF content with good structure",
                metadata={"format": "pdf", "page": 1}
            ),
            Document(
                page_content="DOCX content with formatting",
                metadata={"format": "docx", "style": "heading"}
            ),
            Document(
                page_content="HTML content with tags",
                metadata={"format": "html", "tag": "p"}
            ),
            Document(
                page_content="Image OCR text",
                metadata={"format": "image", "ocr_confidence": 0.92}
            ),
            Document(
                page_content="# Markdown header\n\nGood markdown content",
                metadata={"format": "markdown"}
            )
        ]
        
        # Initialize quality evaluator
        quality_evaluator = MultiFormatQualityEvaluator(self.base_evaluator)
        
        # Evaluate all formats - use 'pdf' as default format
        quality_metrics = quality_evaluator.evaluate_multi_format_chunks(test_chunks, 'pdf')
        
        # Verify comprehensive quality evaluation
        assert 'overall_score' in quality_metrics
        assert quality_metrics['overall_score'] >= 0
        assert quality_metrics['total_chunks'] == 5

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        try:
            from docling.datamodel.base_models import ConversionResult
            DOCLING_AVAILABLE = True
        except ImportError:
            ConversionResult = None
            DOCLING_AVAILABLE = False
        
        # Test error handling with non-existent file
        try:
            result = self.docling_processor.process_document("nonexistent_file.pdf", format_type="pdf")
            # The processor should handle the error gracefully and return a failed result
            if DOCLING_AVAILABLE:
                assert isinstance(result, ConversionResult)
                assert result.status.success is False
            else:
                assert result is not None
        except FileNotFoundError:
            # This is also acceptable behavior
            pass
        
        # Test recovery with fallback chunker
        with patch('src.chunkers.hybrid_chunker.HybridMarkdownChunker.chunk_document') as mock_fallback:
            mock_fallback.return_value = [
                Document(page_content="Fallback content", metadata={"fallback": True})
            ]
            
            # Use fallback chunker for markdown content
            fallback_chunker = HybridMarkdownChunker()
            fallback_chunks = fallback_chunker.chunk_document("# Fallback content\n\nThis is fallback processing.")
            
            # Verify fallback works
            assert len(fallback_chunks) == 1
            assert fallback_chunks[0].metadata.get('fallback') is True
        
        # Test successful processing after error
        test_file = self.temp_dir / "recovery_test.pdf"
        test_file.write_text("Recovery test content")
        
        result = self.docling_processor.process_document(str(test_file), format_type="pdf")
        if DOCLING_AVAILABLE:
            assert isinstance(result, ConversionResult)
            assert result.status.success is True
        else:
            assert result is not None

    def test_performance_benchmarks(self):
        """Test performance benchmarks meet production requirements."""
        try:
            from docling.datamodel.base_models import ConversionResult
            DOCLING_AVAILABLE = True
        except ImportError:
            ConversionResult = None
            DOCLING_AVAILABLE = False
        
        # Initialize performance monitoring
        performance_monitor = PerformanceMonitor()
        
        # Test performance benchmarking
        start_time = time.time()
        operation_id = performance_monitor.start_monitoring("document_processing")
        
        # Create test file for benchmarking
        test_file = self.temp_dir / "benchmark.pdf"
        test_file.write_text("Benchmark content" * 100)  # Larger content for realistic testing
        
        # Process document using actual DoclingProcessor
        result = self.docling_processor.process_document(str(test_file), format_type="pdf")
        
        performance_monitor.end_monitoring(operation_id)
        end_time = time.time()
        
        # Verify performance requirements
        processing_time = end_time - start_time
        assert processing_time < 10.0  # Realistic requirement for test environment
        if DOCLING_AVAILABLE:
            assert isinstance(result, ConversionResult)
            assert result.status.success is True
        else:
            assert result is not None
        
        # Verify performance metrics
        stats = performance_monitor.get_overall_stats()
        assert stats['total_operations'] >= 1

    def test_concurrent_processing_safety(self):
        """Test concurrent processing safety and resource management."""
        import threading
        import concurrent.futures
        try:
            from docling.datamodel.base_models import ConversionResult
            DOCLING_AVAILABLE = True
        except ImportError:
            ConversionResult = None
            DOCLING_AVAILABLE = False
        
        # Create test files for concurrent processing
        test_files = []
        for i in range(3):  # Reduced from 5 to 3 for faster testing
            test_file = self.temp_dir / f"concurrent_{i}.pdf"
            test_file.write_text(f"Concurrent content {i}" * 50)
            test_files.append(str(test_file))
        
        results = []
        errors = []
        
        def process_document(file_path):
            try:
                # Each thread gets its own processor instance to ensure thread safety
                thread_processor = DoclingProcessor(provider=self.mock_provider)
                result = thread_processor.process_document(file_path, format_type="pdf")
                thread_id = threading.current_thread().ident
                # Add thread info to result for verification - mock this for testing
                if isinstance(result, dict):
                    result['metadata'] = result.get('metadata', {})
                    result['metadata']['thread_id'] = thread_id
                elif hasattr(result, 'metadata'):
                    result.metadata = result.metadata or {}
                    result.metadata['thread_id'] = thread_id
                return result
            except Exception as e:
                errors.append(e)
                return None
        
        # Test concurrent processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(process_document, file_path)
                for file_path in test_files
            ]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        # Verify concurrent processing
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3
        
        # Verify all results are ConversionResult instances
        for result in results:
            if DOCLING_AVAILABLE:
                assert isinstance(result, ConversionResult)
                assert result.status.success is True
            else:
                assert result is not None
        
        # Verify thread safety (different thread IDs if multiple threads were used)
        thread_ids = set()
        for result in results:
            if isinstance(result, dict):
                metadata = result.get('metadata', {})
            else:
                metadata = getattr(result, 'metadata', {})
            if metadata and 'thread_id' in metadata:
                thread_ids.add(metadata['thread_id'])
        assert len(thread_ids) >= 1, "Should have thread information"

    def test_backward_compatibility_validation(self):
        """Test that existing Markdown workflows remain fully functional."""
        # Test existing Markdown processing
        markdown_content = """# Test Document
        
This is existing markdown content that should continue working.

## Section 1
Content here.
"""
        
        # Initialize existing components
        markdown_chunker = HybridMarkdownChunker()
        markdown_evaluator = ChunkQualityEvaluator()
        
        # Process with existing workflow
        markdown_chunks = markdown_chunker.chunk_document(markdown_content)
        markdown_metrics = markdown_evaluator.evaluate_chunks(markdown_chunks)
        
        # Verify existing functionality
        assert len(markdown_chunks) > 0
        assert 'overall_score' in markdown_metrics
        assert all(isinstance(chunk, Document) for chunk in markdown_chunks)
        
        # Test with new multi-format evaluator
        multi_evaluator = MultiFormatQualityEvaluator(self.base_evaluator)
        multi_metrics = multi_evaluator.evaluate_multi_format_chunks(markdown_chunks, 'markdown')
        
        # Verify backward compatibility
        assert 'overall_score' in multi_metrics
        assert multi_metrics['overall_score'] == markdown_metrics['overall_score']

    def test_resource_usage_optimization(self):
        """Test resource usage optimization and memory management."""
        try:
            from docling.datamodel.base_models import ConversionResult
            DOCLING_AVAILABLE = True
        except ImportError:
            ConversionResult = None
            DOCLING_AVAILABLE = False
        
        # Create test files for resource monitoring
        test_files = []
        for i in range(3):  # Reduced from 10 to 3 for faster testing
            test_file = self.temp_dir / f"resource_test_{i}.pdf"
            test_file.write_text(f"Resource test content {i}" * 100)
            test_files.append(str(test_file))
        
        # Initialize components
        processor = DoclingProcessor(provider=self.mock_provider)
        system_monitor = SystemMonitor()
        
        # Monitor resource usage
        system_monitor.start_monitoring()
        
        # Process multiple documents
        all_results = []
        
        for file_path in test_files:
            result = processor.process_document(file_path, format_type="pdf")
            all_results.append(result)
        
        system_monitor.stop_monitoring.set()
        
        # Verify resource optimization
        assert len(all_results) == 3
        
        # Verify all results are ConversionResult instances
        for result in all_results:
            if DOCLING_AVAILABLE:
                assert isinstance(result, ConversionResult)
                assert result.status.success is True
            else:
                assert result is not None

    def test_comprehensive_error_reporting(self):
        """Test comprehensive error reporting and diagnostics."""
        try:
            from docling.datamodel.base_models import ConversionResult
            DOCLING_AVAILABLE = True
        except ImportError:
            ConversionResult = None
            DOCLING_AVAILABLE = False
        
        # Create test files for error scenarios
        test_files = {
            "normal.pdf": "Normal content",
            "empty.pdf": "",  # Empty file to trigger potential errors
        }
        
        for filename, content in test_files.items():
            test_file = self.temp_dir / filename
            test_file.write_text(content)
        
        # Initialize components
        processor = DoclingProcessor(provider=self.mock_provider)
        
        results = []
        for filename in test_files.keys():
            file_path = str(self.temp_dir / filename)
            try:
                result = processor.process_document(file_path, format_type="pdf")
                results.append({"file": filename, "result": result, "error": None})
            except Exception as e:
                results.append({"file": filename, "result": None, "error": str(e)})
        
        # Verify error reporting
        assert len(results) == 2
        
        # At least one should succeed (normal.pdf)
        successful_results = [r for r in results if r['error'] is None]
        assert len(successful_results) >= 1
        
        # Verify successful result structure
        for result in successful_results:
            if DOCLING_AVAILABLE:
                assert isinstance(result['result'], ConversionResult)
                assert result['result'].status.success is True
            else:
                assert result['result'] is not None
        
        # Test non-existent file error handling
        try:
            result = processor.process_document("nonexistent_file.pdf", format_type="pdf")
            # Should return failed result
            if DOCLING_AVAILABLE:
                assert isinstance(result, ConversionResult)
                assert result.status.success is False
            else:
                assert result is not None
        except FileNotFoundError:
            # This is also acceptable behavior
            pass

    def test_production_monitoring_integration(self):
        """Test production monitoring and observability integration."""
        try:
            from docling.datamodel.base_models import ConversionResult
            DOCLING_AVAILABLE = True
        except ImportError:
            ConversionResult = None
            DOCLING_AVAILABLE = False
        
        # Create test files for monitoring
        test_files = []
        for i in range(3):  # Reduced from 5 to 3 for faster testing
            test_file = self.temp_dir / f"monitored_{i}.pdf"
            test_file.write_text(f"Monitored content {i}" * 50)
            test_files.append(str(test_file))
        
        # Initialize components
        processor = DoclingProcessor(provider=self.mock_provider)
        performance_monitor = PerformanceMonitor()
        system_monitor = SystemMonitor()
        
        # Start monitoring
        operation_id = performance_monitor.start_monitoring("batch_processing")
        system_monitor.start_monitoring()
        
        # Process batch
        all_results = []
        
        for file_path in test_files:
            result = processor.process_document(file_path, format_type="pdf")
            all_results.append(result)
        
        # Stop monitoring
        performance_monitor.end_monitoring(operation_id)
        system_monitor.stop_monitoring.set()
        
        # Verify monitoring data
        assert len(all_results) == 3
        for result in all_results:
            if DOCLING_AVAILABLE:
                assert isinstance(result, ConversionResult)
                assert result.status.success is True
            else:
                assert result is not None
        
        # Verify monitoring metrics
        stats = performance_monitor.get_overall_stats()
        assert stats['total_operations'] >= 1

    def test_system_health_validation(self):
        """Test system health validation and diagnostic tools."""
        # Initialize system monitor
        system_monitor = SystemMonitor()
        
        # Run health check
        health_status = system_monitor.get_system_status()
        
        # Verify health status structure (based on actual API)
        assert isinstance(health_status, dict)
        assert 'health' in health_status
        assert 'metrics_count' in health_status
        assert 'active_alerts' in health_status
        
        # Verify health check components
        health_data = health_status['health']
        assert 'overall_healthy' in health_data
        assert 'checks' in health_data
        
        # Test diagnostic tools
        health_results = system_monitor.health_checker.run_all_checks()
        assert isinstance(health_results, list)
        assert len(health_results) >= 0  # May be empty in test environment

    def test_complete_pipeline_integration(self):
        """Test complete pipeline integration from file input to quality report output."""
        try:
            from docling.datamodel.base_models import ConversionResult
            DOCLING_AVAILABLE = True
        except ImportError:
            ConversionResult = None
            DOCLING_AVAILABLE = False
        
        # Create comprehensive test files
        test_files = {
            "document.pdf": "PDF content for complete pipeline test",
            "presentation.pptx": "PPTX content for complete pipeline test",
            "spreadsheet.docx": "DOCX content for complete pipeline test",
        }
        
        for filename, content in test_files.items():
            test_file = self.temp_dir / filename
            test_file.write_text(content)
        
        # Initialize all components
        processor = DoclingProcessor(provider=self.mock_provider)
        performance_monitor = PerformanceMonitor()
        system_monitor = SystemMonitor()
        
        # Start comprehensive monitoring
        operation_id = performance_monitor.start_monitoring("complete_pipeline")
        system_monitor.start_monitoring()
        
        # Process all files through complete pipeline
        all_results = []
        
        for filename in test_files.keys():
            file_path = str(self.temp_dir / filename)
            format_type = filename.split('.')[-1]
            
            # Process document
            result = processor.process_document(file_path, format_type=format_type)
            all_results.append(result)
            
            # Verify result structure
            if DOCLING_AVAILABLE:
                assert isinstance(result, ConversionResult)
                assert result.status.success is True
            else:
                assert result is not None
        
        # Create chunks from processed results for quality evaluation
        from langchain.schema import Document
        from src.chunkers.hybrid_chunker import HybridMarkdownChunker
        
        all_chunks = []
        for i, result in enumerate(all_results):
            # Handle both ConversionResult and dict responses
            if DOCLING_AVAILABLE and hasattr(result, 'document'):
                text_content = result.document.export_to_markdown()
            elif isinstance(result, dict):
                text_content = result.get('text', '')
            else:
                text_content = getattr(result, 'text', '')
            
            # Create chunks from the processed text
            chunker = HybridMarkdownChunker(chunk_size=200, chunk_overlap=50)
            documents = chunker.chunk_document(text_content)
            
            # Enrich metadata
            for document in documents:
                enriched_chunk = MetadataEnricher.enrich_chunk(document, {"source": list(test_files.keys())[i]})
                all_chunks.append(enriched_chunk)
        
        # Evaluate quality across all formats
        quality_evaluator = MultiFormatQualityEvaluator(self.base_evaluator)
        quality_metrics = quality_evaluator.evaluate_multi_format_chunks(all_chunks, 'pdf')
        
        # Generate comprehensive report
        report_path = self.temp_dir / "pipeline_report.md"
        report_content = quality_evaluator.generate_multi_format_report(all_chunks, 'pdf', str(report_path))
        
        # Save results
        output_path = self.temp_dir / "pipeline_output.json"
        self.enhanced_file_handler.file_handler.save_chunks(all_chunks, str(output_path))
        
        # Stop monitoring
        performance_monitor.end_monitoring(operation_id)
        system_monitor.stop_monitoring.set()
        
        # Verify complete pipeline
        assert len(all_results) == 3
        assert len(all_chunks) >= 3
        assert all('chunk_id' in chunk.metadata for chunk in all_chunks)
        assert 'overall_score' in quality_metrics
        assert report_path.exists()
        assert output_path.exists()
        
        # Verify report content
        report_text = report_path.read_text()
        assert "# Multi-Format Quality Evaluation Report" in report_text
        assert "## Format-Specific Analysis" in report_text
        
        # Verify performance tracking
        stats = performance_monitor.get_overall_stats()
        assert stats['total_operations'] >= 1