"""
Multi-Format Quality Evaluator for document chunking assessment.

This module extends the existing ChunkQualityEvaluator to provide comprehensive
quality assessment for multi-format documents including PDF, DOCX, PPTX, HTML,
and images while maintaining backward compatibility.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document

from src.chunkers.evaluators import ChunkQualityEvaluator
from src.utils.logger import get_logger


class MultiFormatQualityEvaluator:
    """
    Enhanced quality evaluator for multi-format document processing.
    
    Extends existing ChunkQualityEvaluator functionality to assess quality
    across different document formats while maintaining backward compatibility.
    """
    
    def __init__(self, base_evaluator: ChunkQualityEvaluator):
        """
        Initialize MultiFormatQualityEvaluator.
        
        Args:
            base_evaluator: Instance of ChunkQualityEvaluator for base functionality
            
        Raises:
            ValueError: If base_evaluator is not a ChunkQualityEvaluator instance
        """
        if not isinstance(base_evaluator, ChunkQualityEvaluator):
            raise ValueError("ChunkQualityEvaluator instance required")
            
        self.base_evaluator = base_evaluator
        self.logger = get_logger(__name__)
        
        # Performance tracking
        self.performance_tracker = {}
        
        # Format-specific weights and thresholds
        self.format_weights = {
            'pdf': {
                'structure_preservation': 0.3,
                'content_extraction': 0.25,
                'visual_content': 0.2,
                'page_structure': 0.25
            },
            'docx': {
                'structure_preservation': 0.35,
                'content_extraction': 0.3,
                'heading_analysis': 0.2,
                'format_consistency': 0.15
            },
            'pptx': {
                'structure_preservation': 0.25,
                'content_extraction': 0.2,
                'visual_content': 0.3,
                'slide_structure': 0.25
            },
            'html': {
                'structure_preservation': 0.4,
                'content_extraction': 0.3,
                'markup_quality': 0.2,
                'semantic_structure': 0.1
            },
            'image': {
                'ocr_quality': 0.4,
                'text_extraction': 0.3,
                'visual_content': 0.2,
                'content_confidence': 0.1
            },
            'markdown': {
                'structure_preservation': 0.35,
                'content_quality': 0.35,
                'semantic_coherence': 0.3
            }
        }
        
        # Markdown baseline reference for comparative analysis
        self.markdown_baseline = {
            'structure_score': 0.85,
            'content_score': 0.80,
            'coherence_score': 0.75,
            'overall_score': 0.80
        }
        
        # Supported formats
        self.supported_formats = list(self.format_weights.keys())
    
    def evaluate_multi_format_chunks(self, chunks: List[Document], format_type: str) -> Dict[str, Any]:
        """
        Evaluate chunks with format-specific quality metrics.
        
        Args:
            chunks: List of Document chunks to evaluate
            format_type: Format type ('pdf', 'docx', 'pptx', 'html', 'image', 'markdown')
            
        Returns:
            Dictionary containing comprehensive evaluation results
            
        Raises:
            ValueError: If format_type is not supported
        """
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}. Supported: {self.supported_formats}")
        
        if not chunks:
            return {'error': 'No chunks to evaluate'}
        
        start_time = time.time()
        
        try:
            # Get base evaluation metrics
            base_metrics = self.base_evaluator.evaluate_chunks(chunks)
            
            # Add format-specific metrics
            format_metrics = self._analyze_format_specific_metrics(chunks, format_type)
            
            # Calculate format-adjusted score
            format_adjusted_score = self._calculate_format_adjusted_score(
                base_metrics, format_metrics, format_type
            )
            
            # Perform comparative analysis
            comparative_analysis = self._perform_comparative_analysis(
                base_metrics, format_metrics, format_type
            )
            
            # Generate format insights
            format_insights = self._generate_format_insights(
                base_metrics, format_metrics, format_type
            )
            
            # Track performance
            processing_time = time.time() - start_time
            self.track_evaluation_performance(format_type, processing_time, len(chunks))
            
            # Combine all results
            result = {
                **base_metrics,
                'format_type': format_type,
                'format_specific_metrics': format_metrics,
                'format_adjusted_score': format_adjusted_score,
                'base_score': base_metrics.get('overall_score', 0),
                'comparative_analysis': comparative_analysis,
                'format_insights': format_insights,
                'performance_metrics': {
                    'evaluation_time': processing_time,
                    'chunks_processed': len(chunks),
                    'format_type': format_type,
                    'avg_time_per_chunk': processing_time / len(chunks) if chunks else 0
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Error in multi-format evaluation",
                error=str(e),
                format_type=format_type,
                chunk_count=len(chunks)
            )
            return {
                'error': f'Multi-format evaluation failed: {str(e)}',
                'format_type': format_type,
                'chunks_processed': len(chunks)
            }
    
    def assess_document_structure_preservation(self, chunk: Document, format_type: str) -> float:
        """
        Assess how well document structure is preserved for specific formats.
        
        Args:
            chunk: Document chunk to assess
            format_type: Format type for context-specific assessment
            
        Returns:
            Structure preservation score (0-1)
        """
        metadata = chunk.metadata
        content = chunk.page_content
        
        if format_type == 'pdf':
            return self._assess_pdf_structure(chunk, metadata, content)
        elif format_type == 'docx':
            return self._assess_docx_structure(chunk, metadata, content)
        elif format_type == 'pptx':
            return self._assess_pptx_structure(chunk, metadata, content)
        elif format_type == 'html':
            return self._assess_html_structure(chunk, metadata, content)
        elif format_type == 'image':
            return self._assess_image_structure(chunk, metadata, content)
        elif format_type == 'markdown':
            return self._assess_markdown_structure(chunk, metadata, content)
        else:
            return 0.5  # Default score for unknown formats
    
    def evaluate_visual_content(self, chunk: Document) -> Dict[str, Any]:
        """
        Evaluate visual content processing quality.
        
        Args:
            chunk: Document chunk to evaluate
            
        Returns:
            Dictionary with visual content evaluation metrics
        """
        metadata = chunk.metadata
        content = chunk.page_content
        
        result = {
            'has_visual_content': False,
            'visual_content_type': 'none',
            'ocr_confidence': 0.0,
            'text_extraction_quality': 0.0,
            'visual_complexity': 0.0
        }
        
        # Check for images
        if metadata.get('format_type') == 'image':
            result['has_visual_content'] = True
            result['visual_content_type'] = metadata.get('image_type', 'unknown')
            result['ocr_confidence'] = metadata.get('ocr_confidence', 0.0)
            result['text_extraction_quality'] = min(1.0, result['ocr_confidence'] + 0.1)
            result['visual_complexity'] = self._calculate_visual_complexity(chunk)
        
        # Check for mixed content (PDF, DOCX, etc.)
        elif metadata.get('has_images') or metadata.get('has_tables'):
            result['has_visual_content'] = True
            result['visual_content_type'] = 'mixed'
            result['text_extraction_quality'] = 0.8  # Assume good extraction for mixed content
            result['visual_complexity'] = self._calculate_mixed_content_complexity(chunk)
        
        return result
    
    def calculate_format_specific_score(self, chunk: Document, format_type: str) -> float:
        """
        Calculate format-specific quality score for a chunk.
        
        Args:
            chunk: Document chunk to score
            format_type: Format type for scoring context
            
        Returns:
            Format-specific score (0-1)
        """
        if format_type not in self.format_weights:
            return 0.5
        
        weights = self.format_weights[format_type]
        scores = {}
        
        # Structure preservation
        scores['structure'] = self.assess_document_structure_preservation(chunk, format_type)
        
        # Content extraction quality
        scores['content'] = self._assess_content_extraction_quality(chunk, format_type)
        
        # Format-specific metrics
        if format_type == 'pdf':
            scores['visual'] = self.evaluate_visual_content(chunk)['text_extraction_quality']
            scores['page'] = self._assess_page_structure_quality(chunk)
        elif format_type == 'docx':
            scores['heading'] = self._assess_heading_analysis(chunk)
            scores['format_consistency'] = self._assess_format_consistency(chunk)
        elif format_type == 'image':
            visual_metrics = self.evaluate_visual_content(chunk)
            scores['ocr'] = visual_metrics['ocr_confidence']
            scores['text'] = visual_metrics['text_extraction_quality']
            scores['visual'] = visual_metrics['visual_complexity']
            scores['confidence'] = visual_metrics['ocr_confidence']
        
        # Calculate weighted score
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            score_key = self._map_metric_to_score_key(metric)
            if score_key in scores:
                total_score += scores[score_key] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def benchmark_against_markdown(self, multi_format_score: float, content_type: str) -> Dict[str, Any]:
        """
        Benchmark multi-format results against Markdown quality standards.
        
        Args:
            multi_format_score: Score from multi-format evaluation
            content_type: Type of content being compared
            
        Returns:
            Dictionary with comparative analysis results
        """
        baseline_score = self.markdown_baseline.get('overall_score', 0.80)
        
        relative_performance = multi_format_score / baseline_score if baseline_score > 0 else 0
        quality_differential = multi_format_score - baseline_score
        
        # Performance categorization
        if relative_performance >= 1.1:
            performance_category = 'excellent'
        elif relative_performance >= 0.9:
            performance_category = 'good'
        elif relative_performance >= 0.7:
            performance_category = 'acceptable'
        else:
            performance_category = 'poor'
        
        return {
            'baseline_score': baseline_score,
            'multi_format_score': multi_format_score,
            'relative_performance': relative_performance,
            'quality_differential': quality_differential,
            'performance_category': performance_category,
            'content_type': content_type,
            'recommendations': self._generate_comparison_recommendations(
                relative_performance, quality_differential
            )
        }
    
    def track_evaluation_performance(self, format_type: str, processing_time: float, chunk_count: int):
        """
        Track evaluation performance metrics for different document types.
        
        Args:
            format_type: Format being evaluated
            processing_time: Time taken for evaluation
            chunk_count: Number of chunks processed
        """
        if format_type not in self.performance_tracker:
            self.performance_tracker[format_type] = {
                'total_time': 0.0,
                'total_chunks': 0,
                'evaluation_count': 0,
                'avg_time_per_chunk': 0.0,
                'avg_time_per_evaluation': 0.0
            }
        
        tracker = self.performance_tracker[format_type]
        tracker['total_time'] += processing_time
        tracker['total_chunks'] += chunk_count
        tracker['evaluation_count'] += 1
        
        tracker['avg_time_per_chunk'] = tracker['total_time'] / tracker['total_chunks']
        tracker['avg_time_per_evaluation'] = tracker['total_time'] / tracker['evaluation_count']
    
    def generate_multi_format_report(self, chunks: List[Document], format_type: str, 
                                   output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive multi-format quality report.
        
        Args:
            chunks: List of Document chunks
            format_type: Format type being evaluated
            output_path: Optional path to save report
            
        Returns:
            Generated report as string
        """
        metrics = self.evaluate_multi_format_chunks(chunks, format_type)
        
        report = f"""
# Multi-Format Quality Evaluation Report

## Summary
- **Format Type**: {format_type.upper()}
- **Total Chunks**: {metrics.get('total_chunks', 0)}
- **Base Quality Score**: {metrics.get('base_score', 0):.1f}/100
- **Format-Adjusted Score**: {metrics.get('format_adjusted_score', 0):.1f}/100

## Format-Specific Analysis
{self._generate_format_specific_report_section(metrics, format_type)}

## Visual Content Assessment
{self._generate_visual_content_report_section(metrics)}

## Performance Metrics
- **Evaluation Time**: {metrics.get('performance_metrics', {}).get('evaluation_time', 0):.3f}s
- **Chunks Processed**: {metrics.get('performance_metrics', {}).get('chunks_processed', 0)}
- **Average Time per Chunk**: {metrics.get('performance_metrics', {}).get('avg_time_per_chunk', 0):.3f}s

## Comparative Analysis
{self._generate_comparative_analysis_report_section(metrics)}

## Size Distribution
- **Average Characters**: {metrics.get('size_distribution', {}).get('char_stats', {}).get('mean', 0):.0f}
- **Average Words**: {metrics.get('size_distribution', {}).get('word_stats', {}).get('mean', 0):.0f}
- **Size Consistency**: {metrics.get('size_distribution', {}).get('size_consistency', 0):.2f}

## Content Quality
- **Empty Chunks**: {metrics.get('content_quality', {}).get('empty_chunks', 0)} ({metrics.get('content_quality', {}).get('empty_chunks_pct', 0):.1f}%)
- **Very Short Chunks**: {metrics.get('content_quality', {}).get('very_short_chunks', 0)} ({metrics.get('content_quality', {}).get('very_short_chunks_pct', 0):.1f}%)
- **Incomplete Sentences**: {metrics.get('content_quality', {}).get('incomplete_sentences', 0)} ({metrics.get('content_quality', {}).get('incomplete_sentences_pct', 0):.1f}%)

## Recommendations
{self._generate_multi_format_recommendations(metrics, format_type)}
"""
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(
                "Multi-format evaluation report saved",
                output_path=output_path,
                format_type=format_type
            )
        
        return report
    
    # Private helper methods
    
    def _analyze_format_specific_metrics(self, chunks: List[Document], format_type: str) -> Dict[str, Any]:
        """Analyze format-specific quality metrics."""
        metrics = {
            'document_structure_score': 0.0,
            'visual_content_score': 0.0,
            'format_preservation_score': 0.0
        }
        
        if not chunks:
            return metrics
        
        structure_scores = []
        visual_scores = []
        
        for chunk in chunks:
            # Document structure assessment
            structure_score = self.assess_document_structure_preservation(chunk, format_type)
            structure_scores.append(structure_score)
            
            # Visual content assessment
            visual_result = self.evaluate_visual_content(chunk)
            visual_scores.append(visual_result['text_extraction_quality'])
        
        metrics['document_structure_score'] = np.mean(structure_scores)
        metrics['visual_content_score'] = np.mean(visual_scores)
        metrics['format_preservation_score'] = (
            metrics['document_structure_score'] * 0.6 + 
            metrics['visual_content_score'] * 0.4
        )
        
        # Add format-specific metrics
        if format_type == 'pdf':
            metrics['page_structure_score'] = self._analyze_pdf_page_structure(chunks)
            metrics['content_extraction_quality'] = self._analyze_content_extraction_quality(chunks)
            metrics['mixed_content_handling'] = self._analyze_mixed_content_handling(chunks)
        elif format_type == 'docx':
            metrics['heading_analysis'] = self._analyze_docx_headings(chunks)
            metrics['structure_consistency'] = self._analyze_structure_consistency(chunks)
        elif format_type == 'image':
            metrics['ocr_quality_score'] = self._analyze_ocr_quality(chunks)
            metrics['text_extraction_confidence'] = self._analyze_text_extraction_confidence(chunks)
            metrics['image_type_analysis'] = self._analyze_image_types(chunks)
        elif format_type == 'pptx':
            metrics['slide_structure_score'] = self._analyze_slide_structure(chunks)
            metrics['mixed_content_handling'] = self._analyze_mixed_content_handling(chunks)
        elif format_type == 'html':
            metrics['markup_quality_score'] = self._analyze_markup_quality(chunks)
            metrics['semantic_structure_score'] = self._analyze_semantic_structure(chunks)
        
        return metrics
    
    def _calculate_format_adjusted_score(self, base_metrics: Dict[str, Any], 
                                       format_metrics: Dict[str, Any], 
                                       format_type: str) -> float:
        """Calculate format-adjusted overall score."""
        base_score = base_metrics.get('overall_score', 0)
        
        if format_type not in self.format_weights:
            return base_score
        
        weights = self.format_weights[format_type]
        format_score = 0.0
        
        # Apply format-specific adjustments
        if format_type == 'pdf':
            format_score = (
                format_metrics.get('document_structure_score', 0) * weights['structure_preservation'] +
                format_metrics.get('content_extraction_quality', 0.5) * weights['content_extraction'] +
                format_metrics.get('visual_content_score', 0) * weights['visual_content'] +
                format_metrics.get('page_structure_score', 0.5) * weights['page_structure']
            ) * 100
        elif format_type == 'docx':
            format_score = (
                format_metrics.get('document_structure_score', 0) * weights['structure_preservation'] +
                format_metrics.get('content_extraction_quality', 0.5) * weights['content_extraction'] +
                format_metrics.get('heading_analysis', 0.5) * weights['heading_analysis'] +
                format_metrics.get('structure_consistency', 0.5) * weights['format_consistency']
            ) * 100
        elif format_type == 'image':
            format_score = (
                format_metrics.get('ocr_quality_score', 0) * weights['ocr_quality'] +
                format_metrics.get('text_extraction_confidence', 0) * weights['text_extraction'] +
                format_metrics.get('visual_content_score', 0) * weights['visual_content'] +
                format_metrics.get('text_extraction_confidence', 0) * weights['content_confidence']
            ) * 100
        else:
            # For other formats, use base score with format preservation adjustment
            format_score = base_score * (1 + format_metrics.get('format_preservation_score', 0) * 0.2)
        
        # Combine base score with format-specific adjustments
        final_score = (base_score * 0.7) + (format_score * 0.3)
        return min(100, max(0, final_score))
    
    def _perform_comparative_analysis(self, base_metrics: Dict[str, Any], 
                                    format_metrics: Dict[str, Any], 
                                    format_type: str) -> Dict[str, Any]:
        """Perform comparative analysis against baseline."""
        format_score = format_metrics.get('format_preservation_score', 0)
        
        comparison = self.benchmark_against_markdown(format_score, format_type)
        
        return {
            'markdown_baseline_comparison': comparison,
            'format_specific_insights': self._generate_format_insights(
                base_metrics, format_metrics, format_type
            ),
            'quality_differential': comparison['quality_differential'],
            'relative_score': comparison['relative_performance']
        }
    
    def _generate_format_insights(self, base_metrics: Dict[str, Any], 
                                format_metrics: Dict[str, Any], 
                                format_type: str) -> Dict[str, Any]:
        """Generate format-specific insights and recommendations."""
        insights = {
            'recommendations': [],
            'quality_highlights': [],
            'areas_for_improvement': []
        }
        
        # Format-specific recommendations
        if format_type == 'pdf':
            if format_metrics.get('page_structure_score', 0) < 0.5:
                insights['recommendations'].append(
                    "Consider improving PDF page structure detection"
                )
            if format_metrics.get('visual_content_score', 0) > 0.8:
                insights['quality_highlights'].append(
                    "Excellent visual content extraction"
                )
        elif format_type == 'image':
            ocr_score = format_metrics.get('ocr_quality_score', 0)
            if ocr_score > 0.9:
                insights['quality_highlights'].append(
                    "High OCR quality detected"
                )
            elif ocr_score < 0.7:
                insights['areas_for_improvement'].append(
                    "OCR quality could be improved"
                )
        
        # General quality insights
        if base_metrics.get('overall_score', 0) > 80:
            insights['quality_highlights'].append(
                "High overall quality score achieved"
            )
        
        return insights
    
    # Format-specific assessment methods
    
    def _assess_pdf_structure(self, chunk: Document, metadata: Dict, content: str) -> float:
        """Assess PDF structure preservation."""
        score = 0.5  # Base score
        
        # Check for structure preservation metadata
        if metadata.get('structure_preserved', False):
            score += 0.3
        
        # Check for page information
        if 'page' in metadata:
            score += 0.1
        
        # Check for content organization
        if any(header in content for header in ['#', 'Chapter', 'Section']):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_docx_structure(self, chunk: Document, metadata: Dict, content: str) -> float:
        """Assess DOCX structure preservation."""
        score = 0.5  # Base score
        
        # Check for heading levels
        if 'heading_level' in metadata:
            score += 0.2
        
        # Check for structure preservation
        if metadata.get('structure_preserved', False):
            score += 0.2
        
        # Check for formatted content
        if any(indicator in content for indicator in ['#', '**', '*']):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_pptx_structure(self, chunk: Document, metadata: Dict, content: str) -> float:
        """Assess PPTX structure preservation."""
        score = 0.5  # Base score
        
        # Check for slide information
        if 'slide' in metadata:
            score += 0.2
        
        # Check for structure preservation
        if metadata.get('structure_preserved', False):
            score += 0.2
        
        # Check for bullet points or structured content
        if any(bullet in content for bullet in ['•', '-', '*']):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_html_structure(self, chunk: Document, metadata: Dict, content: str) -> float:
        """Assess HTML structure preservation."""
        score = 0.5  # Base score
        
        # Check for semantic structure
        if any(tag in content for tag in ['<h', '<p', '<div', '<section']):
            score += 0.3
        
        # Check for structure preservation
        if metadata.get('structure_preserved', False):
            score += 0.2
        
        return min(1.0, score)
    
    def _assess_image_structure(self, chunk: Document, metadata: Dict, content: str) -> float:
        """Assess image structure preservation."""
        score = 0.3  # Lower base score for images
        
        # Check OCR confidence
        ocr_confidence = metadata.get('ocr_confidence', 0)
        score += ocr_confidence * 0.4
        
        # Check for text extraction
        if metadata.get('has_text', False):
            score += 0.2
        
        # Check for structure preservation
        if metadata.get('structure_preserved', False):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_markdown_structure(self, chunk: Document, metadata: Dict, content: str) -> float:
        """Assess Markdown structure preservation."""
        score = 0.7  # Higher base score for Markdown
        
        # Check for headers
        if any(header in metadata for header in ['Header 1', 'Header 2', 'Header 3']):
            score += 0.2
        
        # Check for structure preservation
        if metadata.get('structure_preserved', False):
            score += 0.1
        
        return min(1.0, score)
    
    # Additional helper methods for comprehensive analysis
    
    def _calculate_visual_complexity(self, chunk: Document) -> float:
        """Calculate visual complexity score for image chunks."""
        metadata = chunk.metadata
        
        # Base complexity based on image type
        image_type = metadata.get('image_type', 'unknown')
        if image_type == 'chart':
            return 0.8
        elif image_type == 'table':
            return 0.9
        elif image_type == 'diagram':
            return 0.7
        else:
            return 0.5
    
    def _calculate_mixed_content_complexity(self, chunk: Document) -> float:
        """Calculate complexity for mixed content documents."""
        metadata = chunk.metadata
        complexity = 0.5
        
        if metadata.get('has_images'):
            complexity += 0.2
        if metadata.get('has_tables'):
            complexity += 0.3
        
        return min(1.0, complexity)
    
    def _assess_content_extraction_quality(self, chunk: Document, format_type: str) -> float:
        """Assess content extraction quality."""
        content = chunk.page_content
        
        # Basic content quality checks
        if not content or len(content.strip()) == 0:
            return 0.0
        
        # Check for proper sentence structure
        sentences = content.split('.')
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
        sentence_ratio = complete_sentences / len(sentences) if sentences else 0
        
        # Check for reasonable word count
        word_count = len(content.split())
        if word_count < 5:
            return 0.3
        elif word_count < 20:
            return 0.6
        else:
            return min(1.0, 0.8 + sentence_ratio * 0.2)
    
    def _map_metric_to_score_key(self, metric: str) -> str:
        """Map metric names to score dictionary keys."""
        mapping = {
            'structure_preservation': 'structure',
            'content_extraction': 'content',
            'visual_content': 'visual',
            'page_structure': 'page',
            'heading_analysis': 'heading',
            'format_consistency': 'format_consistency',
            'ocr_quality': 'ocr',
            'text_extraction': 'text',
            'content_confidence': 'confidence'
        }
        return mapping.get(metric, metric)
    
    def _generate_comparison_recommendations(self, relative_performance: float, 
                                          quality_differential: float) -> List[str]:
        """Generate recommendations based on comparative analysis."""
        recommendations = []
        
        if relative_performance < 0.7:
            recommendations.append(
                "Consider improving content extraction and structure preservation"
            )
        
        if quality_differential < -0.2:
            recommendations.append(
                "Quality significantly below Markdown baseline - review processing pipeline"
            )
        
        if relative_performance > 1.1:
            recommendations.append(
                "Excellent quality - performance exceeds Markdown baseline"
            )
        
        return recommendations
    
    # Report generation helper methods
    
    def _generate_format_specific_report_section(self, metrics: Dict[str, Any], format_type: str) -> str:
        """Generate format-specific report section."""
        format_metrics = metrics.get('format_specific_metrics', {})
        
        section = f"### {format_type.upper()} Specific Metrics\n"
        section += f"- **Document Structure Score**: {format_metrics.get('document_structure_score', 0):.2f}\n"
        section += f"- **Visual Content Score**: {format_metrics.get('visual_content_score', 0):.2f}\n"
        section += f"- **Format Preservation Score**: {format_metrics.get('format_preservation_score', 0):.2f}\n"
        
        return section
    
    def _generate_visual_content_report_section(self, metrics: Dict[str, Any]) -> str:
        """Generate visual content report section."""
        format_metrics = metrics.get('format_specific_metrics', {})
        
        section = "### Visual Content Analysis\n"
        
        if 'ocr_quality_score' in format_metrics:
            section += f"- **OCR Quality Score**: {format_metrics['ocr_quality_score']:.2f}\n"
        
        if 'text_extraction_confidence' in format_metrics:
            section += f"- **Text Extraction Confidence**: {format_metrics['text_extraction_confidence']:.2f}\n"
        
        if 'visual_content_score' in format_metrics:
            section += f"- **Visual Content Processing**: {format_metrics['visual_content_score']:.2f}\n"
        
        return section
    
    def _generate_comparative_analysis_report_section(self, metrics: Dict[str, Any]) -> str:
        """Generate comparative analysis report section."""
        comparative = metrics.get('comparative_analysis', {})
        baseline = comparative.get('markdown_baseline_comparison', {})
        
        section = "### Comparison with Markdown Baseline\n"
        section += f"- **Relative Performance**: {baseline.get('relative_performance', 0):.2f}\n"
        section += f"- **Quality Differential**: {baseline.get('quality_differential', 0):.2f}\n"
        section += f"- **Performance Category**: {baseline.get('performance_category', 'unknown')}\n"
        
        return section
    
    def _generate_multi_format_recommendations(self, metrics: Dict[str, Any], format_type: str) -> str:
        """Generate multi-format specific recommendations."""
        recommendations = []
        
        # Add format-specific recommendations
        format_insights = metrics.get('format_insights', {})
        recommendations.extend(format_insights.get('recommendations', []))
        
        # Add performance-based recommendations
        if metrics.get('format_adjusted_score', 0) < 60:
            recommendations.append(
                f"⚠️ {format_type.upper()} processing quality below acceptable threshold"
            )
        
        # Add comparative recommendations
        comparative = metrics.get('comparative_analysis', {})
        baseline = comparative.get('markdown_baseline_comparison', {})
        recommendations.extend(baseline.get('recommendations', []))
        
        if not recommendations:
            recommendations.append("✅ Quality meets standards for multi-format processing")
        
        return '\n'.join(f"- {rec}" for rec in recommendations)
    
    # Placeholder methods for format-specific analysis (to be implemented as needed)
    
    def _analyze_pdf_page_structure(self, chunks: List[Document]) -> float:
        """Analyze PDF page structure quality."""
        return 0.7  # Placeholder implementation
    
    def _analyze_content_extraction_quality(self, chunks: List[Document]) -> float:
        """Analyze content extraction quality."""
        return 0.8  # Placeholder implementation
    
    def _analyze_docx_headings(self, chunks: List[Document]) -> float:
        """Analyze DOCX heading structure."""
        return 0.8  # Placeholder implementation
    
    def _analyze_structure_consistency(self, chunks: List[Document]) -> float:
        """Analyze structure consistency."""
        return 0.7  # Placeholder implementation
    
    def _analyze_ocr_quality(self, chunks: List[Document]) -> float:
        """Analyze OCR quality for image chunks."""
        confidences = [chunk.metadata.get('ocr_confidence', 0) for chunk in chunks]
        return np.mean(confidences) if confidences else 0.0
    
    def _analyze_text_extraction_confidence(self, chunks: List[Document]) -> float:
        """Analyze text extraction confidence."""
        confidences = [chunk.metadata.get('ocr_confidence', 0) for chunk in chunks]
        return np.mean(confidences) if confidences else 0.0
    
    def _analyze_image_types(self, chunks: List[Document]) -> Dict[str, int]:
        """Analyze distribution of image types."""
        types = {}
        for chunk in chunks:
            img_type = chunk.metadata.get('image_type', 'unknown')
            types[img_type] = types.get(img_type, 0) + 1
        return types
    
    def _analyze_slide_structure(self, chunks: List[Document]) -> float:
        """Analyze slide structure for PPTX."""
        return 0.8  # Placeholder implementation
    
    def _analyze_mixed_content_handling(self, chunks: List[Document]) -> float:
        """Analyze mixed content handling."""
        return 0.7  # Placeholder implementation
    
    def _analyze_markup_quality(self, chunks: List[Document]) -> float:
        """Analyze HTML markup quality."""
        return 0.8  # Placeholder implementation
    
    def _analyze_semantic_structure(self, chunks: List[Document]) -> float:
        """Analyze semantic structure."""
        return 0.7  # Placeholder implementation
    
    def _assess_page_structure_quality(self, chunk: Document) -> float:
        """Assess page structure quality."""
        return 0.7  # Placeholder implementation
    
    def _assess_heading_analysis(self, chunk: Document) -> float:
        """Assess heading analysis quality."""
        return 0.8  # Placeholder implementation
    
    def _assess_format_consistency(self, chunk: Document) -> float:
        """Assess format consistency."""
        return 0.7  # Placeholder implementation