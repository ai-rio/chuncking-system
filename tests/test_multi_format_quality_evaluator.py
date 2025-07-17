"""
Unit tests for the MultiFormatQualityEvaluator class.
Comprehensive test suite for multi-format quality evaluation following TDD principles.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import time

from langchain_core.documents import Document

from src.chunkers.evaluators import ChunkQualityEvaluator
from src.chunkers.multi_format_quality_evaluator import MultiFormatQualityEvaluator


class TestMultiFormatQualityEvaluator:
    """Test cases for MultiFormatQualityEvaluator following Story 1.4 requirements."""

    @pytest.fixture
    def base_evaluator(self) -> ChunkQualityEvaluator:
        """Create a base ChunkQualityEvaluator instance."""
        return ChunkQualityEvaluator()

    @pytest.fixture
    def multi_format_evaluator(self, base_evaluator) -> MultiFormatQualityEvaluator:
        """Create a MultiFormatQualityEvaluator instance for testing."""
        return MultiFormatQualityEvaluator(base_evaluator)

    @pytest.fixture
    def pdf_chunks(self):
        """Sample PDF chunks with typical PDF structure."""
        return [
            Document(
                page_content="Executive Summary\n\nThis document outlines the key findings from our research project.",
                metadata={
                    "source": "test.pdf",
                    "page": 1,
                    "format_type": "pdf",
                    "has_images": False,
                    "has_tables": False,
                    "structure_preserved": True
                }
            ),
            Document(
                page_content="Introduction\n\nThe following sections detail our methodology and results.",
                metadata={
                    "source": "test.pdf",
                    "page": 2,
                    "format_type": "pdf",
                    "has_images": True,
                    "has_tables": True,
                    "structure_preserved": True
                }
            )
        ]

    @pytest.fixture
    def docx_chunks(self):
        """Sample DOCX chunks with Word document structure."""
        return [
            Document(
                page_content="Document Title\n\nThis is a Word document with formatted content.",
                metadata={
                    "source": "test.docx",
                    "format_type": "docx",
                    "has_images": False,
                    "has_tables": True,
                    "structure_preserved": True,
                    "heading_level": 1
                }
            ),
            Document(
                page_content="Section 1: Overview\n\nDetailed explanation of the topic.",
                metadata={
                    "source": "test.docx",
                    "format_type": "docx",
                    "has_images": False,
                    "has_tables": False,
                    "structure_preserved": True,
                    "heading_level": 2
                }
            )
        ]

    @pytest.fixture
    def image_chunks(self):
        """Sample image chunks from OCR processing."""
        return [
            Document(
                page_content="COMPANY LOGO\n\nAnnual Report 2023",
                metadata={
                    "source": "chart.png",
                    "format_type": "image",
                    "ocr_confidence": 0.95,
                    "image_type": "chart",
                    "has_text": True,
                    "structure_preserved": False
                }
            ),
            Document(
                page_content="Revenue Growth\n\nQ1: $1.2M\nQ2: $1.5M\nQ3: $1.8M\nQ4: $2.1M",
                metadata={
                    "source": "table.png",
                    "format_type": "image",
                    "ocr_confidence": 0.88,
                    "image_type": "table",
                    "has_text": True,
                    "structure_preserved": True
                }
            )
        ]

    @pytest.fixture
    def markdown_chunks(self):
        """Sample Markdown chunks for baseline comparison."""
        return [
            Document(
                page_content="# Main Title\n\nThis is markdown content with proper structure.",
                metadata={
                    "source": "test.md",
                    "format_type": "markdown",
                    "Header 1": "Main Title",
                    "structure_preserved": True
                }
            ),
            Document(
                page_content="## Subsection\n\n- List item 1\n- List item 2\n- List item 3",
                metadata={
                    "source": "test.md",
                    "format_type": "markdown",
                    "Header 2": "Subsection",
                    "structure_preserved": True
                }
            )
        ]

    def test_init_with_base_evaluator(self, base_evaluator):
        """Test MultiFormatQualityEvaluator initialization with base evaluator."""
        evaluator = MultiFormatQualityEvaluator(base_evaluator)
        
        assert evaluator.base_evaluator == base_evaluator
        assert evaluator.performance_tracker == {}
        assert "pdf" in evaluator.format_weights
        assert "docx" in evaluator.format_weights
        assert "image" in evaluator.format_weights

    def test_init_invalid_base_evaluator(self):
        """Test initialization with invalid base evaluator raises error."""
        with pytest.raises(ValueError, match="ChunkQualityEvaluator instance required"):
            MultiFormatQualityEvaluator(None)
        
        with pytest.raises(ValueError, match="ChunkQualityEvaluator instance required"):
            MultiFormatQualityEvaluator("not_evaluator")

    def test_enhanced_metrics_document_structure_pdf(self, multi_format_evaluator, pdf_chunks):
        """Test enhanced metrics assess document structure preservation for PDF."""
        result = multi_format_evaluator.evaluate_multi_format_chunks(pdf_chunks, "pdf")
        
        # Check that PDF-specific structure metrics are included
        assert "format_specific_metrics" in result
        assert "document_structure_score" in result["format_specific_metrics"]
        assert "visual_content_score" in result["format_specific_metrics"]
        assert "format_type" in result
        assert result["format_type"] == "pdf"
        
        # PDF with good structure should score well
        assert result["format_specific_metrics"]["document_structure_score"] > 0.5

    def test_enhanced_metrics_document_structure_docx(self, multi_format_evaluator, docx_chunks):
        """Test enhanced metrics assess document structure preservation for DOCX."""
        result = multi_format_evaluator.evaluate_multi_format_chunks(docx_chunks, "docx")
        
        # Check DOCX-specific structure analysis
        assert result["format_type"] == "docx"
        assert "heading_analysis" in result["format_specific_metrics"]
        assert "format_preservation_score" in result["format_specific_metrics"]
        
        # DOCX with headings should score well
        assert result["format_specific_metrics"]["format_preservation_score"] > 0.5

    def test_visual_content_evaluation_images(self, multi_format_evaluator, image_chunks):
        """Test visual content evaluation analyzes image processing quality."""
        result = multi_format_evaluator.evaluate_multi_format_chunks(image_chunks, "image")
        
        # Check image-specific metrics
        assert result["format_type"] == "image"
        assert "ocr_quality_score" in result["format_specific_metrics"]
        assert "visual_content_score" in result["format_specific_metrics"]
        assert "text_extraction_confidence" in result["format_specific_metrics"]
        
        # High OCR confidence should result in good scores
        assert result["format_specific_metrics"]["ocr_quality_score"] > 0.8

    def test_visual_content_evaluation_mixed_content(self, multi_format_evaluator, pdf_chunks):
        """Test visual content evaluation for documents with mixed content."""
        # Add visual content metadata to PDF chunks
        pdf_chunks[1].metadata["has_images"] = True
        pdf_chunks[1].metadata["has_tables"] = True
        
        result = multi_format_evaluator.evaluate_multi_format_chunks(pdf_chunks, "pdf")
        
        # Should detect and score visual content
        assert "visual_content_score" in result["format_specific_metrics"]
        assert "mixed_content_handling" in result["format_specific_metrics"]

    def test_format_specific_scoring_pdf(self, multi_format_evaluator, pdf_chunks):
        """Test format-specific scoring adapts evaluation criteria for PDF."""
        result = multi_format_evaluator.evaluate_multi_format_chunks(pdf_chunks, "pdf")
        
        # PDF scoring should consider page structure
        assert "page_structure_score" in result["format_specific_metrics"]
        assert "content_extraction_quality" in result["format_specific_metrics"]
        
        # Format-specific overall score should be calculated
        assert "format_adjusted_score" in result
        assert 0 <= result["format_adjusted_score"] <= 100

    def test_format_specific_scoring_docx(self, multi_format_evaluator, docx_chunks):
        """Test format-specific scoring adapts evaluation criteria for DOCX."""
        result = multi_format_evaluator.evaluate_multi_format_chunks(docx_chunks, "docx")
        
        # DOCX scoring should consider document structure
        assert "heading_analysis" in result["format_specific_metrics"]
        assert "structure_consistency" in result["format_specific_metrics"]
        
        # Format-specific score should be adjusted for DOCX characteristics
        assert "format_adjusted_score" in result
        assert result["format_adjusted_score"] != result["base_score"]

    def test_format_specific_scoring_images(self, multi_format_evaluator, image_chunks):
        """Test format-specific scoring adapts evaluation criteria for images."""
        result = multi_format_evaluator.evaluate_multi_format_chunks(image_chunks, "image")
        
        # Image scoring should consider OCR quality
        assert "ocr_quality_score" in result["format_specific_metrics"]
        assert "text_extraction_confidence" in result["format_specific_metrics"]
        
        # Should handle different image types appropriately
        assert "image_type_analysis" in result["format_specific_metrics"]

    def test_comparative_analysis_markdown_baseline(self, multi_format_evaluator, pdf_chunks, markdown_chunks):
        """Test comparative analysis benchmarks multi-format results against Markdown."""
        pdf_result = multi_format_evaluator.evaluate_multi_format_chunks(pdf_chunks, "pdf")
        markdown_result = multi_format_evaluator.evaluate_multi_format_chunks(markdown_chunks, "markdown")
        
        # Comparative analysis should be included
        assert "comparative_analysis" in pdf_result
        assert "markdown_baseline_comparison" in pdf_result["comparative_analysis"]
        
        # Should show relative performance
        comparison = pdf_result["comparative_analysis"]["markdown_baseline_comparison"]
        assert "relative_performance" in comparison
        assert "quality_differential" in comparison

    def test_comparative_analysis_cross_format(self, multi_format_evaluator, pdf_chunks, docx_chunks):
        """Test comparative analysis between different formats."""
        pdf_result = multi_format_evaluator.evaluate_multi_format_chunks(pdf_chunks, "pdf")
        docx_result = multi_format_evaluator.evaluate_multi_format_chunks(docx_chunks, "docx")
        
        # Both should have baseline comparisons
        assert "comparative_analysis" in pdf_result
        assert "comparative_analysis" in docx_result
        
        # Should be able to compare format-adjusted scores
        assert pdf_result["format_adjusted_score"] != docx_result["format_adjusted_score"]

    def test_performance_tracking_overhead(self, multi_format_evaluator, pdf_chunks):
        """Test performance tracking monitors evaluation overhead."""
        start_time = time.time()
        result = multi_format_evaluator.evaluate_multi_format_chunks(pdf_chunks, "pdf")
        end_time = time.time()
        
        # Performance metrics should be tracked
        assert "performance_metrics" in result
        assert "evaluation_time" in result["performance_metrics"]
        assert "chunks_processed" in result["performance_metrics"]
        assert "format_type" in result["performance_metrics"]
        
        # Evaluation time should be reasonable
        assert result["performance_metrics"]["evaluation_time"] > 0
        assert result["performance_metrics"]["evaluation_time"] < (end_time - start_time) + 0.1

    def test_performance_tracking_different_formats(self, multi_format_evaluator, pdf_chunks, image_chunks):
        """Test performance tracking for different document types."""
        pdf_result = multi_format_evaluator.evaluate_multi_format_chunks(pdf_chunks, "pdf")
        image_result = multi_format_evaluator.evaluate_multi_format_chunks(image_chunks, "image")
        
        # Both should have performance tracking
        assert "performance_metrics" in pdf_result
        assert "performance_metrics" in image_result
        
        # Should track format-specific performance
        assert pdf_result["performance_metrics"]["format_type"] == "pdf"
        assert image_result["performance_metrics"]["format_type"] == "image"

    def test_reporting_integration_multi_format(self, multi_format_evaluator, pdf_chunks):
        """Test reporting integration extends existing quality reports."""
        result = multi_format_evaluator.evaluate_multi_format_chunks(pdf_chunks, "pdf")
        
        # Should include all base metrics plus format-specific ones
        assert "total_chunks" in result
        assert "size_distribution" in result
        assert "content_quality" in result
        assert "format_specific_metrics" in result
        assert "comparative_analysis" in result
        assert "performance_metrics" in result

    def test_reporting_integration_format_insights(self, multi_format_evaluator, docx_chunks):
        """Test reporting includes multi-format insights."""
        result = multi_format_evaluator.evaluate_multi_format_chunks(docx_chunks, "docx")
        
        # Should provide format-specific insights
        assert "format_insights" in result
        assert "recommendations" in result["format_insights"]
        assert "quality_highlights" in result["format_insights"]

    def test_assess_document_structure_preservation_pdf(self, multi_format_evaluator, pdf_chunks):
        """Test document structure preservation assessment for PDF."""
        chunk = pdf_chunks[0]
        score = multi_format_evaluator.assess_document_structure_preservation(chunk, "pdf")
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        
        # PDF with good structure should score well
        chunk.metadata["structure_preserved"] = True
        good_score = multi_format_evaluator.assess_document_structure_preservation(chunk, "pdf")
        
        chunk.metadata["structure_preserved"] = False
        poor_score = multi_format_evaluator.assess_document_structure_preservation(chunk, "pdf")
        
        assert good_score > poor_score

    def test_assess_document_structure_preservation_docx(self, multi_format_evaluator, docx_chunks):
        """Test document structure preservation assessment for DOCX."""
        chunk = docx_chunks[0]
        score = multi_format_evaluator.assess_document_structure_preservation(chunk, "docx")
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        
        # DOCX with heading levels should score well
        assert chunk.metadata.get("heading_level") == 1
        assert score > 0.5

    def test_evaluate_visual_content_images(self, multi_format_evaluator, image_chunks):
        """Test visual content evaluation for images."""
        chunk = image_chunks[0]
        result = multi_format_evaluator.evaluate_visual_content(chunk)
        
        assert isinstance(result, dict)
        assert "ocr_confidence" in result
        assert "text_extraction_quality" in result
        assert "visual_content_type" in result
        
        # High OCR confidence should yield good results
        assert result["ocr_confidence"] == 0.95
        assert result["text_extraction_quality"] > 0.8

    def test_evaluate_visual_content_mixed_documents(self, multi_format_evaluator, pdf_chunks):
        """Test visual content evaluation for documents with mixed content."""
        # Modify PDF chunk to have visual content
        chunk = pdf_chunks[1]
        chunk.metadata["has_images"] = True
        chunk.metadata["has_tables"] = True
        
        result = multi_format_evaluator.evaluate_visual_content(chunk)
        
        assert "has_visual_content" in result
        assert "visual_complexity" in result
        assert result["has_visual_content"] is True

    def test_calculate_format_specific_score_pdf(self, multi_format_evaluator, pdf_chunks):
        """Test format-specific score calculation for PDF."""
        chunk = pdf_chunks[0]
        score = multi_format_evaluator.calculate_format_specific_score(chunk, "pdf")
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        
        # PDF score should consider page structure
        assert score > 0

    def test_calculate_format_specific_score_docx(self, multi_format_evaluator, docx_chunks):
        """Test format-specific score calculation for DOCX."""
        chunk = docx_chunks[0]
        score = multi_format_evaluator.calculate_format_specific_score(chunk, "docx")
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        
        # DOCX score should consider document structure
        assert score > 0

    def test_calculate_format_specific_score_images(self, multi_format_evaluator, image_chunks):
        """Test format-specific score calculation for images."""
        chunk = image_chunks[0]
        score = multi_format_evaluator.calculate_format_specific_score(chunk, "image")
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        
        # Image score should consider OCR quality
        assert score > 0.8  # High OCR confidence should yield good score

    def test_benchmark_against_markdown(self, multi_format_evaluator, markdown_chunks):
        """Test benchmarking against Markdown baseline."""
        multi_format_score = 0.75
        content_type = "structured_document"
        
        result = multi_format_evaluator.benchmark_against_markdown(
            multi_format_score, content_type
        )
        
        assert isinstance(result, dict)
        assert "baseline_score" in result
        assert "relative_performance" in result
        assert "quality_differential" in result
        assert "content_type" in result
        
        assert result["content_type"] == content_type

    def test_track_evaluation_performance(self, multi_format_evaluator):
        """Test evaluation performance tracking."""
        format_type = "pdf"
        processing_time = 0.5
        chunk_count = 5
        
        multi_format_evaluator.track_evaluation_performance(
            format_type, processing_time, chunk_count
        )
        
        # Should update performance tracker
        assert format_type in multi_format_evaluator.performance_tracker
        tracker = multi_format_evaluator.performance_tracker[format_type]
        
        assert "total_time" in tracker
        assert "total_chunks" in tracker
        assert "avg_time_per_chunk" in tracker
        assert "evaluation_count" in tracker

    def test_unsupported_format_error(self, multi_format_evaluator, pdf_chunks):
        """Test error handling for unsupported format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            multi_format_evaluator.evaluate_multi_format_chunks(pdf_chunks, "unsupported")

    def test_empty_chunks_error(self, multi_format_evaluator):
        """Test error handling for empty chunks."""
        result = multi_format_evaluator.evaluate_multi_format_chunks([], "pdf")
        
        assert "error" in result
        assert result["error"] == "No chunks to evaluate"

    def test_invalid_chunk_format_handling(self, multi_format_evaluator):
        """Test handling of invalid chunk formats."""
        invalid_chunks = [
            Document(page_content="test", metadata={"format_type": "invalid"})
        ]
        
        # Should handle gracefully and use provided format_type
        result = multi_format_evaluator.evaluate_multi_format_chunks(invalid_chunks, "pdf")
        
        # Should process as PDF despite metadata
        assert result["format_type"] == "pdf"

    def test_backward_compatibility_with_base_evaluator(self, multi_format_evaluator, markdown_chunks):
        """Test backward compatibility with base ChunkQualityEvaluator."""
        # Test with markdown chunks (should use base evaluator)
        result = multi_format_evaluator.evaluate_multi_format_chunks(markdown_chunks, "markdown")
        
        # Should include base evaluation metrics
        assert "total_chunks" in result
        assert "size_distribution" in result
        assert "content_quality" in result
        assert "semantic_coherence" in result
        assert "overlap_analysis" in result
        assert "structural_preservation" in result
        assert "overall_score" in result

    def test_generate_multi_format_report(self, multi_format_evaluator, pdf_chunks, temp_dir):
        """Test generation of multi-format quality report."""
        output_path = temp_dir / "multi_format_report.md"
        
        report = multi_format_evaluator.generate_multi_format_report(
            pdf_chunks, "pdf", str(output_path)
        )
        
        # Should include multi-format sections
        assert "# Multi-Format Quality Evaluation Report" in report
        assert "## Format-Specific Analysis" in report
        assert "## Visual Content Assessment" in report
        assert "## Performance Metrics" in report
        assert "## Comparative Analysis" in report
        
        # File should be created
        assert output_path.exists()

    def test_integration_with_existing_quality_workflow(self, multi_format_evaluator, pdf_chunks):
        """Test integration with existing quality evaluation workflow."""
        # Should work seamlessly with existing patterns
        result = multi_format_evaluator.evaluate_multi_format_chunks(pdf_chunks, "pdf")
        
        # All required metrics should be present
        required_metrics = [
            "total_chunks", "size_distribution", "content_quality", 
            "semantic_coherence", "overlap_analysis", "structural_preservation",
            "overall_score", "format_specific_metrics", "comparative_analysis"
        ]
        
        for metric in required_metrics:
            assert metric in result, f"Missing required metric: {metric}"

    def test_quality_score_consistency(self, multi_format_evaluator, pdf_chunks):
        """Test that quality scores are consistent across evaluations."""
        result1 = multi_format_evaluator.evaluate_multi_format_chunks(pdf_chunks, "pdf")
        result2 = multi_format_evaluator.evaluate_multi_format_chunks(pdf_chunks, "pdf")
        
        # Should produce identical results for same input
        assert result1["overall_score"] == result2["overall_score"]
        assert result1["format_adjusted_score"] == result2["format_adjusted_score"]

    def test_scalability_with_large_chunk_sets(self, multi_format_evaluator):
        """Test scalability with large numbers of chunks."""
        # Create large set of chunks
        large_chunks = [
            Document(
                page_content=f"PDF content chunk {i} with substantial text content for testing scalability.",
                metadata={
                    "source": f"test_{i}.pdf",
                    "format_type": "pdf",
                    "page": i % 10 + 1,
                    "structure_preserved": True
                }
            ) for i in range(100)
        ]
        
        # Should handle large datasets efficiently
        result = multi_format_evaluator.evaluate_multi_format_chunks(large_chunks, "pdf")
        
        assert result["total_chunks"] == 100
        assert "performance_metrics" in result
        assert result["performance_metrics"]["evaluation_time"] > 0