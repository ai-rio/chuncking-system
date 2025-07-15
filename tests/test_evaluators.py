"""
Unit tests for the ChunkQualityEvaluator class.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from langchain_core.documents import Document

from src.chunkers.evaluators import ChunkQualityEvaluator


class TestChunkQualityEvaluator:
    """Test cases for ChunkQualityEvaluator."""

    @pytest.fixture
    def evaluator(self) -> ChunkQualityEvaluator:
        """Create a ChunkQualityEvaluator instance for testing."""
        return ChunkQualityEvaluator()

    def test_init(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.vectorizer is not None
        assert evaluator.min_words_for_very_short == 10
        assert evaluator.coherence_score_boost_factor == 1.0
        assert evaluator.structure_score_weight_factor == 1.0

    def test_evaluate_chunks_empty_list(self, evaluator):
        """Test evaluation with empty chunk list."""
        result = evaluator.evaluate_chunks([])
        
        assert 'error' in result
        assert result['error'] == 'No chunks to evaluate'

    def test_evaluate_chunks_basic(self, evaluator, sample_chunks):
        """Test basic chunk evaluation."""
        result = evaluator.evaluate_chunks(sample_chunks)
        
        # Check that all expected metrics are present
        assert 'total_chunks' in result
        assert 'size_distribution' in result
        assert 'content_quality' in result
        assert 'semantic_coherence' in result
        assert 'overlap_analysis' in result
        assert 'structural_preservation' in result
        assert 'overall_score' in result
        
        assert result['total_chunks'] == len(sample_chunks)
        assert 0 <= result['overall_score'] <= 100

    def test_analyze_size_distribution_empty(self, evaluator):
        """Test size distribution analysis with empty chunks."""
        result = evaluator._analyze_size_distribution([])
        
        expected_stats = {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
        assert result['char_stats'] == expected_stats
        assert result['word_stats'] == expected_stats
        assert result['size_consistency'] == 0

    def test_analyze_size_distribution_normal(self, evaluator, sample_chunks):
        """Test size distribution analysis with normal chunks."""
        result = evaluator._analyze_size_distribution(sample_chunks)
        
        # Check that stats are calculated
        assert result['char_stats']['mean'] > 0
        assert result['char_stats']['median'] > 0
        assert result['char_stats']['min'] > 0
        assert result['char_stats']['max'] > 0
        
        assert result['word_stats']['mean'] > 0
        assert 0 <= result['size_consistency'] <= 1

    def test_analyze_size_distribution_consistent_sizes(self, evaluator):
        """Test size distribution with very consistent chunk sizes."""
        chunks = [
            Document(page_content="A" * 100, metadata={}),
            Document(page_content="B" * 100, metadata={}),
            Document(page_content="C" * 100, metadata={}),
        ]
        
        result = evaluator._analyze_size_distribution(chunks)
        
        # Should have high consistency score
        assert result['size_consistency'] > 0.9

    def test_analyze_size_distribution_inconsistent_sizes(self, evaluator):
        """Test size distribution with very inconsistent chunk sizes."""
        chunks = [
            Document(page_content="A", metadata={}),  # 1 char
            Document(page_content="B" * 1000, metadata={}),  # 1000 chars
            Document(page_content="C" * 10, metadata={}),  # 10 chars
        ]
        
        result = evaluator._analyze_size_distribution(chunks)
        
        # Should have low consistency score
        assert result['size_consistency'] < 0.5

    def test_analyze_content_quality_good_chunks(self, evaluator):
        """Test content quality analysis with good chunks."""
        chunks = [
            Document(page_content="This is a well-formed sentence.", metadata={}),
            Document(page_content="Another proper sentence with good structure.", metadata={}),
        ]
        
        result = evaluator._analyze_content_quality(chunks)
        
        assert result['empty_chunks'] == 0
        assert result['very_short_chunks'] == 0
        assert result['incomplete_sentences'] == 0
        assert result['empty_chunks_pct'] == 0
        assert result['very_short_chunks_pct'] == 0

    def test_analyze_content_quality_poor_chunks(self, evaluator):
        """Test content quality analysis with poor chunks."""
        chunks = [
            Document(page_content="", metadata={}),  # Empty
            Document(page_content="word", metadata={}),  # Very short
            Document(page_content="Incomplete sentence without ending", metadata={}),  # No punctuation
        ]
        
        result = evaluator._analyze_content_quality(chunks)
        
        assert result['empty_chunks'] == 1
        assert result['very_short_chunks'] == 1
        assert result['incomplete_sentences'] == 1
        assert result['empty_chunks_pct'] == pytest.approx(33.33, abs=0.1)

    def test_analyze_content_quality_special_content(self, evaluator):
        """Test content quality with headers, code, and lists."""
        chunks = [
            Document(page_content="# Header Content", metadata={"Header 1": "Header Content"}),
            Document(page_content="```python\ncode_here\n```", metadata={"content_type": "code"}),
            Document(page_content="- List item\n- Another item", metadata={}),
            Document(page_content="1. Numbered item\n2. Another numbered", metadata={}),
        ]
        
        result = evaluator._analyze_content_quality(chunks)
        
        # Special content types should not be marked as incomplete sentences
        assert result['incomplete_sentences'] == 0

    def test_analyze_semantic_coherence_single_chunk(self, evaluator):
        """Test semantic coherence with single chunk."""
        chunks = [Document(page_content="Single chunk", metadata={})]
        
        result = evaluator._analyze_semantic_coherence(chunks)
        
        assert result['coherence_score'] == 1.0
        assert result['avg_similarity'] == 0.0

    def test_analyze_semantic_coherence_empty_chunks(self, evaluator):
        """Test semantic coherence with empty chunks."""
        chunks = [
            Document(page_content="", metadata={}),
            Document(page_content="", metadata={}),
        ]
        
        result = evaluator._analyze_semantic_coherence(chunks)
        
        assert result['coherence_score'] == 1.0
        assert result['avg_similarity'] == 0.0

    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    def test_analyze_semantic_coherence_normal(self, mock_vectorizer_class, evaluator):
        """Test semantic coherence with normal chunks."""
        # Mock the vectorizer
        mock_vectorizer = Mock()
        mock_vectorizer_class.return_value = mock_vectorizer
        
        # Mock TF-IDF matrix
        mock_matrix = Mock()
        mock_vectorizer.fit_transform.return_value = mock_matrix
        
        # Mock cosine similarity result
        similarity_matrix = np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.7],
            [0.6, 0.7, 1.0]
        ])
        
        with patch('sklearn.metrics.pairwise.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = similarity_matrix
            
            chunks = [
                Document(page_content="First chunk about cats", metadata={}),
                Document(page_content="Second chunk about dogs", metadata={}),
                Document(page_content="Third chunk about pets", metadata={}),
            ]
            
            result = evaluator._analyze_semantic_coherence(chunks)
            
            assert 'coherence_score' in result
            assert 'avg_similarity' in result
            assert 0 <= result['coherence_score'] <= 1
            assert 0 <= result['avg_similarity'] <= 1

    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    def test_analyze_semantic_coherence_error_handling(self, mock_vectorizer_class, evaluator):
        """Test semantic coherence error handling."""
        mock_vectorizer = Mock()
        mock_vectorizer_class.return_value = mock_vectorizer
        mock_vectorizer.fit_transform.side_effect = Exception("TF-IDF failed")
        
        chunks = [
            Document(page_content="First chunk", metadata={}),
            Document(page_content="Second chunk", metadata={}),
        ]
        
        result = evaluator._analyze_semantic_coherence(chunks)
        
        assert result['coherence_score'] == 0.0
        assert result['avg_similarity'] == 0.0
        assert 'error' in result

    def test_analyze_overlap_single_chunk(self, evaluator):
        """Test overlap analysis with single chunk."""
        chunks = [Document(page_content="Single chunk", metadata={})]
        
        result = evaluator._analyze_overlap(chunks)
        
        assert result['avg_overlap'] == 0.0
        assert result['overlap_std'] == 0.0
        assert result['overlap_distribution'] == []

    def test_analyze_overlap_normal_chunks(self, evaluator):
        """Test overlap analysis with normal chunks."""
        chunks = [
            Document(page_content="the quick brown fox", metadata={}),
            Document(page_content="brown fox jumps over", metadata={}),
            Document(page_content="fox jumps over lazy", metadata={}),
        ]
        
        result = evaluator._analyze_overlap(chunks)
        
        assert 0 <= result['avg_overlap'] <= 1
        assert result['overlap_std'] >= 0
        assert len(result['overlap_distribution']) <= 10

    def test_analyze_overlap_no_overlap(self, evaluator):
        """Test overlap analysis with no overlapping content."""
        chunks = [
            Document(page_content="completely different words", metadata={}),
            Document(page_content="totally unique content here", metadata={}),
        ]
        
        result = evaluator._analyze_overlap(chunks)
        
        assert result['avg_overlap'] == 0.0

    def test_analyze_overlap_empty_chunks(self, evaluator):
        """Test overlap analysis with empty chunks."""
        chunks = [
            Document(page_content="", metadata={}),
            Document(page_content="", metadata={}),
        ]
        
        result = evaluator._analyze_overlap(chunks)
        
        assert result['avg_overlap'] == 0.0

    def test_analyze_structure_preservation(self, evaluator):
        """Test structure preservation analysis."""
        chunks = [
            Document(page_content="# Header content", metadata={"Header 1": "Test"}),
            Document(page_content="```python\ncode\n```", metadata={"content_type": "code"}),
            Document(page_content="- List item", metadata={}),
            Document(page_content="[link](http://example.com)", metadata={}),
            Document(page_content="Regular text", metadata={}),
        ]
        
        result = evaluator._analyze_structure_preservation(chunks)
        
        assert result['chunks_with_headers'] >= 1
        assert result['chunks_with_code'] >= 1
        assert result['chunks_with_lists'] >= 1
        assert result['chunks_with_links'] >= 1
        
        # Check percentages
        total = len(chunks)
        assert result['chunks_with_headers_pct'] == (result['chunks_with_headers'] / total) * 100

    def test_analyze_structure_preservation_edge_cases(self, evaluator):
        """Test structure preservation with edge cases."""
        chunks = [
            Document(page_content="# Header in metadata", metadata={"Header 2": "Something"}),
            Document(page_content="def function_name():", metadata={}),  # Code without backticks
            Document(page_content="1. Numbered list\n2. Item two", metadata={}),
            Document(page_content="Normal text without structure", metadata={}),
        ]
        
        result = evaluator._analyze_structure_preservation(chunks)
        
        # Should detect various structure types
        assert result['chunks_with_headers'] >= 1
        assert result['chunks_with_code'] >= 1
        assert result['chunks_with_lists'] >= 1

    def test_calculate_overall_score(self, evaluator):
        """Test overall score calculation."""
        metrics = {
            'size_distribution': {'size_consistency': 0.8},
            'content_quality': {
                'empty_chunks_pct': 0,
                'very_short_chunks_pct': 10,
                'incomplete_sentences_pct': 5
            },
            'semantic_coherence': {'coherence_score': 0.7},
            'structural_preservation': {
                'chunks_with_headers_pct': 50,
                'chunks_with_code_pct': 20,
                'chunks_with_lists_pct': 30,
                'chunks_with_links_pct': 10
            }
        }
        
        score = evaluator._calculate_overall_score(metrics)
        
        assert 0 <= score <= 100
        assert isinstance(score, float)

    def test_calculate_overall_score_error_handling(self, evaluator):
        """Test overall score calculation with missing data."""
        incomplete_metrics = {'size_distribution': {}}
        
        score = evaluator._calculate_overall_score(incomplete_metrics)
        
        assert score == 0.0

    def test_generate_report(self, evaluator, sample_chunks, temp_dir):
        """Test report generation."""
        output_path = temp_dir / "test_report.md"
        
        report = evaluator.generate_report(sample_chunks, str(output_path))
        
        # Check report content
        assert "# Chunk Quality Evaluation Report" in report
        assert "## Summary" in report
        assert "## Size Distribution" in report
        assert "## Content Quality" in report
        assert "## Semantic Coherence" in report
        assert "## Structure Preservation" in report
        assert "## Recommendations" in report
        
        # Check file was created
        assert output_path.exists()
        saved_content = output_path.read_text()
        assert saved_content == report

    def test_generate_report_recommendations(self, evaluator):
        """Test that report includes appropriate recommendations."""
        # Create chunks that should trigger various recommendations
        poor_chunks = [
            Document(page_content="", metadata={}),  # Empty chunk
            Document(page_content="a", metadata={}),  # Very short
            Document(page_content="incomplete", metadata={}),  # No punctuation
        ] * 10  # Multiply to get high percentages
        
        with patch.object(evaluator, 'evaluate_chunks') as mock_evaluate:
            mock_evaluate.return_value = {
                'total_chunks': len(poor_chunks),
                'content_quality': {
                    'empty_chunks': 10, 'empty_chunks_pct': 50,
                    'very_short_chunks': 10, 'very_short_chunks_pct': 50,
                    'incomplete_sentences': 10, 'incomplete_sentences_pct': 50
                },
                'semantic_coherence': {'coherence_score': 0.2, 'avg_similarity': 0.1},
                'size_distribution': {'size_consistency': 0.3, 'char_stats': {'mean': 10}, 'word_stats': {'mean': 2}},
                'structural_preservation': {
                    'chunks_with_headers': 0, 'chunks_with_headers_pct': 0,
                    'chunks_with_code': 0, 'chunks_with_code_pct': 0,
                    'chunks_with_lists': 0, 'chunks_with_lists_pct': 0,
                },
                'overall_score': 30
            }
            
            report = evaluator.generate_report(poor_chunks)
            
            # Should include various warnings
            assert "⚠️" in report
            assert "High number of empty chunks" in report
            assert "Many very short chunks" in report
            assert "Low semantic coherence" in report
            assert "Inconsistent chunk sizes" in report

    def test_generate_report_excellent_quality(self, evaluator):
        """Test report for excellent quality chunks."""
        excellent_chunks = [
            Document(
                page_content="This is an excellent chunk with proper structure and content.",
                metadata={"Header 1": "Test", "chunk_tokens": 12, "word_count": 11}
            )
        ] * 5
        
        with patch.object(evaluator, 'evaluate_chunks') as mock_evaluate:
            mock_evaluate.return_value = {
                'total_chunks': 5,
                'content_quality': {
                    'empty_chunks': 0, 'empty_chunks_pct': 0,
                    'very_short_chunks': 0, 'very_short_chunks_pct': 0,
                    'incomplete_sentences': 0, 'incomplete_sentences_pct': 0
                },
                'semantic_coherence': {'coherence_score': 0.9, 'avg_similarity': 0.8},
                'size_distribution': {'size_consistency': 0.9, 'char_stats': {'mean': 60}, 'word_stats': {'mean': 11}},
                'structural_preservation': {
                    'chunks_with_headers': 5, 'chunks_with_headers_pct': 100,
                    'chunks_with_code': 0, 'chunks_with_code_pct': 0,
                    'chunks_with_lists': 0, 'chunks_with_lists_pct': 0,
                },
                'overall_score': 85
            }
            
            report = evaluator.generate_report(excellent_chunks)
            
            assert "✅ Excellent chunking quality!" in report

    def test_evaluator_with_quality_test_chunks(self, evaluator, quality_test_chunks):
        """Test evaluator with specifically designed test chunks."""
        # Test good chunks
        good_result = evaluator.evaluate_chunks(quality_test_chunks['good_chunks'])
        assert good_result['overall_score'] > 50
        
        # Test poor chunks
        poor_result = evaluator.evaluate_chunks(quality_test_chunks['poor_chunks'])
        assert poor_result['overall_score'] < good_result['overall_score']

    def test_evaluator_consistency(self, evaluator, sample_chunks):
        """Test that evaluator produces consistent results."""
        result1 = evaluator.evaluate_chunks(sample_chunks)
        result2 = evaluator.evaluate_chunks(sample_chunks)
        
        # Results should be identical for same input
        assert result1['overall_score'] == result2['overall_score']
        assert result1['total_chunks'] == result2['total_chunks']

    def test_evaluator_performance_with_large_chunks(self, evaluator):
        """Test evaluator performance with large number of chunks."""
        # Create many chunks to test performance
        large_chunk_list = [
            Document(
                page_content=f"Chunk {i} with some content for testing performance.",
                metadata={"chunk_index": i, "chunk_tokens": 10, "word_count": 9}
            ) for i in range(100)
        ]
        
        # Should complete without timeout or memory issues
        result = evaluator.evaluate_chunks(large_chunk_list)
        
        assert result['total_chunks'] == 100
        assert 'overall_score' in result