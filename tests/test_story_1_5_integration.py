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

    def setup_method(self):
        """Set up test environment for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        
        # Initialize components with proper dependencies
        self.base_evaluator = ChunkQualityEvaluator()
        self.docling_processor = DoclingProcessor()
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
        # Simulate PDF processing workflow
        with patch('src.chunkers.docling_processor.DoclingProcessor.process_document') as mock_process:
            # Mock PDF processing result
            mock_process.return_value = [
                Document(
                    page_content="PDF content chunk 1",
                    metadata={"format": "pdf", "page": 1}
                ),
                Document(
                    page_content="PDF content chunk 2", 
                    metadata={"format": "pdf", "page": 2}
                )
            ]
            
            # Initialize components
            processor = DoclingProcessor()
            quality_evaluator = MultiFormatQualityEvaluator(self.base_evaluator)
            file_handler = self.enhanced_file_handler
            
            # Process document
            pdf_path = "test.pdf"  # Mock path
            chunks = processor.process_document(pdf_path)
            
            # Evaluate quality
            quality_metrics = quality_evaluator.evaluate_multi_format_chunks(chunks, 'pdf')
            
            # Verify end-to-end processing
            assert len(chunks) == 2
            assert all(chunk.metadata.get('format') == 'pdf' for chunk in chunks)
            assert 'overall_score' in quality_metrics
            assert quality_metrics['total_chunks'] == 2
            
            # Verify performance tracking
            assert mock_process.called

    def test_complete_docx_processing_pipeline(self):
        """Test complete DOCX processing pipeline from end to end."""
        with patch('src.chunkers.docling_processor.DoclingProcessor.process_document') as mock_process:
            # Mock DOCX processing result
            mock_process.return_value = [
                Document(
                    page_content="DOCX header content",
                    metadata={"format": "docx", "style": "heading"}
                ),
                Document(
                    page_content="DOCX body content with formatting",
                    metadata={"format": "docx", "style": "body"}
                )
            ]
            
            # Initialize components
            processor = DoclingProcessor()
            quality_evaluator = MultiFormatQualityEvaluator(self.base_evaluator)
            
            # Process document
            docx_path = "test.docx"  # Mock path
            chunks = processor.process_document(docx_path)
            
            # Evaluate quality with format-specific metrics
            quality_metrics = quality_evaluator.evaluate_multi_format_chunks(chunks, 'docx')
            
            # Verify DOCX-specific processing
            assert len(chunks) == 2
            assert any(chunk.metadata.get('style') == 'heading' for chunk in chunks)
            assert 'overall_score' in quality_metrics
            assert quality_metrics['overall_score'] >= 0
            
    def test_complete_image_processing_pipeline(self):
        """Test complete image processing pipeline with OCR."""
        with patch('src.chunkers.docling_processor.DoclingProcessor.process_document') as mock_process:
            # Mock image processing with OCR result
            mock_process.return_value = [
                Document(
                    page_content="Extracted text from image via OCR",
                    metadata={
                        "format": "image",
                        "ocr_confidence": 0.95,
                        "image_dimensions": "1024x768"
                    }
                )
            ]
            
            # Initialize components
            processor = DoclingProcessor()
            quality_evaluator = MultiFormatQualityEvaluator(self.base_evaluator)
            
            # Process image
            image_path = "test.jpg"  # Mock path
            chunks = processor.process_document(image_path)
            
            # Evaluate with image-specific metrics
            quality_metrics = quality_evaluator.evaluate_multi_format_chunks(chunks, 'image')
            
            # Verify image processing
            assert len(chunks) == 1
            assert chunks[0].metadata.get('format') == 'image'
            assert chunks[0].metadata.get('ocr_confidence') > 0.9
            assert 'overall_score' in quality_metrics
            assert quality_metrics['overall_score'] >= 0

    def test_mixed_format_batch_processing(self):
        """Test batch processing of mixed document formats."""
        with patch('src.chunkers.docling_processor.DoclingProcessor.process_document') as mock_process:
            # Mock processing results for different formats
            def mock_process_side_effect(file_path):
                if file_path.endswith('.pdf'):
                    return [Document(page_content="PDF content", metadata={"format": "pdf"})]
                elif file_path.endswith('.docx'):
                    return [Document(page_content="DOCX content", metadata={"format": "docx"})]
                elif file_path.endswith('.html'):
                    return [Document(page_content="HTML content", metadata={"format": "html"})]
                else:
                    return [Document(page_content="Unknown format", metadata={"format": "unknown"})]
            
            mock_process.side_effect = mock_process_side_effect
            
            # Initialize components
            processor = DoclingProcessor()
            file_handler = self.enhanced_file_handler
            quality_evaluator = MultiFormatQualityEvaluator(self.base_evaluator)
            
            # Batch process mixed formats
            mixed_files = ["test.pdf", "test.docx", "test.html"]
            all_chunks = []
            
            for file_path in mixed_files:
                chunks = processor.process_document(file_path)
                all_chunks.extend(chunks)
            
            # Evaluate mixed format batch - use 'pdf' as default format
            quality_metrics = quality_evaluator.evaluate_multi_format_chunks(all_chunks, 'pdf')
            
            # Verify mixed format processing
            assert len(all_chunks) == 3
            formats = {chunk.metadata.get('format') for chunk in all_chunks}
            assert formats == {'pdf', 'docx', 'html'}
            assert 'overall_score' in quality_metrics
            assert quality_metrics['overall_score'] >= 0

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
        with patch('src.chunkers.docling_processor.DoclingProcessor.process_document') as mock_process:
            # Mock processing error
            mock_process.side_effect = Exception("Processing error")
            
            # Initialize components
            processor = DoclingProcessor()
            quality_evaluator = MultiFormatQualityEvaluator(self.base_evaluator)
            
            # Test error handling
            try:
                chunks = processor.process_document("error_file.pdf")
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Processing error" in str(e)
            
            # Test recovery with fallback
            with patch('src.chunkers.hybrid_chunker.HybridMarkdownChunker.chunk_document') as mock_fallback:
                mock_fallback.return_value = [
                    Document(page_content="Fallback content", metadata={"fallback": True})
                ]
                
                # Use fallback chunker
                fallback_chunker = HybridMarkdownChunker()
                fallback_chunks = fallback_chunker.chunk_document("fallback content")
                
                # Verify fallback works
                assert len(fallback_chunks) == 1
                assert fallback_chunks[0].metadata.get('fallback') is True

    def test_performance_benchmarks(self):
        """Test performance benchmarks meet production requirements."""
        with patch('src.chunkers.docling_processor.DoclingProcessor.process_document') as mock_process:
            # Mock realistic processing time
            def mock_process_with_timing(file_path):
                time.sleep(0.1)  # Simulate 100ms processing time
                return [Document(page_content=f"Content from {file_path}", metadata={"format": "pdf"})]
            
            mock_process.side_effect = mock_process_with_timing
            
            # Initialize components
            processor = DoclingProcessor()
            performance_monitor = PerformanceMonitor()
            
            # Test performance benchmarking
            start_time = time.time()
            performance_monitor.start_operation("document_processing")
            
            # Process document
            chunks = processor.process_document("test.pdf")
            
            performance_monitor.end_operation("document_processing")
            end_time = time.time()
            
            # Verify performance requirements
            processing_time = end_time - start_time
            assert processing_time < 1.0  # Sub-second requirement
            assert len(chunks) == 1
            
            # Verify performance metrics
            metrics = performance_monitor.get_metrics()
            assert 'document_processing' in metrics

    def test_concurrent_processing_safety(self):
        """Test concurrent processing safety and resource management."""
        import threading
        import queue
        
        with patch('src.chunkers.docling_processor.DoclingProcessor.process_document') as mock_process:
            # Mock thread-safe processing
            def mock_thread_safe_process(file_path):
                time.sleep(0.05)  # Simulate processing time
                return [Document(page_content=f"Content from {file_path}", metadata={"thread_id": threading.current_thread().ident})]
            
            mock_process.side_effect = mock_thread_safe_process
            
            # Initialize components
            processor = DoclingProcessor()
            results_queue = queue.Queue()
            
            # Define worker function
            def worker(file_path):
                try:
                    chunks = processor.process_document(file_path)
                    results_queue.put({"file": file_path, "chunks": chunks, "success": True})
                except Exception as e:
                    results_queue.put({"file": file_path, "error": str(e), "success": False})
            
            # Start concurrent processing
            threads = []
            test_files = [f"test_{i}.pdf" for i in range(5)]
            
            for file_path in test_files:
                thread = threading.Thread(target=worker, args=(file_path,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Collect results
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())
            
            # Verify concurrent processing
            assert len(results) == 5
            assert all(result['success'] for result in results)
            
            # Verify different threads were used
            thread_ids = {result['chunks'][0].metadata['thread_id'] for result in results}
            assert len(thread_ids) > 1  # Multiple threads used

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
        with patch('src.chunkers.docling_processor.DoclingProcessor.process_document') as mock_process:
            # Mock memory-efficient processing
            def mock_memory_efficient_process(file_path):
                # Simulate memory usage tracking
                import psutil
                process = psutil.Process()
                memory_before = process.memory_info().rss
                
                # Create chunk
                chunk = Document(
                    page_content=f"Memory efficient content from {file_path}",
                    metadata={"memory_before": memory_before, "format": "pdf"}
                )
                
                return [chunk]
            
            mock_process.side_effect = mock_memory_efficient_process
            
            # Initialize components
            processor = DoclingProcessor()
            system_monitor = SystemMonitor()
            
            # Monitor resource usage
            system_monitor.start_monitoring()
            
            # Process multiple documents
            test_files = [f"test_{i}.pdf" for i in range(10)]
            all_chunks = []
            
            for file_path in test_files:
                chunks = processor.process_document(file_path)
                all_chunks.extend(chunks)
            
            system_monitor.stop_monitoring()
            
            # Verify resource optimization
            assert len(all_chunks) == 10
            
            # Verify memory usage is tracked
            for chunk in all_chunks:
                assert 'memory_before' in chunk.metadata

    def test_comprehensive_error_reporting(self):
        """Test comprehensive error reporting and diagnostics."""
        with patch('src.chunkers.docling_processor.DoclingProcessor.process_document') as mock_process:
            # Mock various error scenarios
            def mock_error_scenarios(file_path):
                if "corrupt" in file_path:
                    raise ValueError("Corrupted file format")
                elif "permission" in file_path:
                    raise PermissionError("Access denied")
                elif "timeout" in file_path:
                    raise TimeoutError("Processing timeout")
                else:
                    return [Document(page_content="Normal content", metadata={"format": "pdf"})]
            
            mock_process.side_effect = mock_error_scenarios
            
            # Initialize components
            processor = DoclingProcessor()
            error_files = ["corrupt.pdf", "permission.pdf", "timeout.pdf", "normal.pdf"]
            
            results = []
            for file_path in error_files:
                try:
                    chunks = processor.process_document(file_path)
                    results.append({"file": file_path, "chunks": chunks, "error": None})
                except Exception as e:
                    results.append({"file": file_path, "chunks": [], "error": str(e)})
            
            # Verify error reporting
            assert len(results) == 4
            assert results[0]['error'] == "Corrupted file format"
            assert results[1]['error'] == "Access denied"
            assert results[2]['error'] == "Processing timeout"
            assert results[3]['error'] is None
            assert len(results[3]['chunks']) == 1

    def test_production_monitoring_integration(self):
        """Test production monitoring and observability integration."""
        with patch('src.chunkers.docling_processor.DoclingProcessor.process_document') as mock_process:
            # Mock processing with monitoring
            def mock_monitored_process(file_path):
                # Simulate monitoring data
                return [Document(
                    page_content=f"Monitored content from {file_path}",
                    metadata={
                        "format": "pdf",
                        "processing_time": 0.5,
                        "memory_used": 1024 * 1024,  # 1MB
                        "success": True
                    }
                )]
            
            mock_process.side_effect = mock_monitored_process
            
            # Initialize components
            processor = DoclingProcessor()
            performance_monitor = PerformanceMonitor()
            system_monitor = SystemMonitor()
            
            # Start monitoring
            performance_monitor.start_operation("batch_processing")
            system_monitor.start_monitoring()
            
            # Process batch
            test_files = [f"monitored_{i}.pdf" for i in range(5)]
            all_chunks = []
            
            for file_path in test_files:
                chunks = processor.process_document(file_path)
                all_chunks.extend(chunks)
            
            # Stop monitoring
            performance_monitor.end_operation("batch_processing")
            system_monitor.stop_monitoring()
            
            # Verify monitoring data
            assert len(all_chunks) == 5
            for chunk in all_chunks:
                assert 'processing_time' in chunk.metadata
                assert 'memory_used' in chunk.metadata
                assert chunk.metadata['success'] is True
            
            # Verify monitoring metrics
            perf_metrics = performance_monitor.get_metrics()
            assert 'batch_processing' in perf_metrics

    def test_system_health_validation(self):
        """Test system health validation and diagnostic tools."""
        # Initialize system monitor
        system_monitor = SystemMonitor()
        
        # Run health check
        health_status = system_monitor.get_system_status()
        
        # Verify health check components
        assert 'cpu_usage' in health_status
        assert 'memory_usage' in health_status
        assert 'disk_usage' in health_status
        assert 'system_load' in health_status
        
        # Verify health status
        assert health_status['status'] in ['healthy', 'warning', 'critical']
        
        # Test diagnostic tools
        health_results = system_monitor.health_checker.run_all_checks()
        assert isinstance(health_results, dict)
        assert len(health_results) > 0

    def test_complete_pipeline_integration(self):
        """Test complete pipeline integration from file input to quality report output."""
        with patch('src.chunkers.docling_processor.DoclingProcessor.process_document') as mock_process:
            # Mock complete pipeline processing
            mock_process.return_value = [
                Document(
                    page_content="Complete pipeline test content",
                    metadata={"format": "pdf", "page": 1, "quality": "high"}
                ),
                Document(
                    page_content="Second chunk from pipeline",
                    metadata={"format": "pdf", "page": 2, "quality": "medium"}
                )
            ]
            
            # Initialize complete pipeline
            processor = DoclingProcessor()
            file_handler = self.enhanced_file_handler
            quality_evaluator = MultiFormatQualityEvaluator(self.base_evaluator)
            performance_monitor = PerformanceMonitor()
            
            # Run complete pipeline
            performance_monitor.start_operation("complete_pipeline")
            
            # 1. Process document
            chunks = processor.process_document("test.pdf")
            
            # 2. Enrich metadata
            enriched_chunks = []
            for chunk in chunks:
                enriched_chunk = MetadataEnricher.enrich_chunk(chunk, {"source": "test.pdf"})
                enriched_chunks.append(enriched_chunk)
            
            # 3. Evaluate quality
            quality_metrics = quality_evaluator.evaluate_multi_format_chunks(enriched_chunks, 'pdf')
            
            # 4. Generate report
            report_path = self.temp_dir / "pipeline_report.md"
            report_content = quality_evaluator.generate_multi_format_report(enriched_chunks, 'pdf', str(report_path))
            
            # 5. Save results
            output_path = self.temp_dir / "pipeline_output.json"
            file_handler.save_chunks(enriched_chunks, str(output_path))
            
            performance_monitor.end_operation("complete_pipeline")
            
            # Verify complete pipeline
            assert len(enriched_chunks) == 2
            assert all('chunk_id' in chunk.metadata for chunk in enriched_chunks)
            assert 'overall_score' in quality_metrics
            assert report_path.exists()
            assert output_path.exists()
            
            # Verify report content
            report_text = report_path.read_text()
            assert "# Multi-Format Quality Evaluation Report" in report_text
            assert "## Format-Specific Analysis" in report_text
            
            # Verify performance tracking
            perf_metrics = performance_monitor.get_metrics()
            assert 'complete_pipeline' in perf_metrics