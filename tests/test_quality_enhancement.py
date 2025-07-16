"""
Test cases for automated quality enhancement feature.
Following TDD principles - tests written first to define expected behavior.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.utils.path_utils import MarkdownFileManager, QualityEnhancementManager
from src.chunkers.evaluators import ChunkQualityEvaluator


class TestQualityEnhancementManager:
    """Test automated quality enhancement functionality."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        output_dir = tempfile.mkdtemp()
        yield Path(output_dir)
        shutil.rmtree(output_dir)
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks with various quality issues."""
        return [
            # Good chunk
            Document(
                page_content="This is a complete sentence with proper ending. This chunk has good quality and coherence.",
                metadata={"chunk_index": 0, "source": "test"}
            ),
            # Incomplete sentence chunk
            Document(
                page_content="This sentence is incomplete and doesn't end properly so it",
                metadata={"chunk_index": 1, "source": "test"}
            ),
            # Very short chunk
            Document(
                page_content="Short.",
                metadata={"chunk_index": 2, "source": "test"}
            ),
            # Chunk for completion
            Document(
                page_content="continues from previous chunk. This is additional content.",
                metadata={"chunk_index": 3, "source": "test"}
            )
        ]
    
    @pytest.fixture
    def mock_output_paths(self, temp_dirs):
        """Create mock output paths structure."""
        chunks_dir = temp_dirs / "chunks"
        reports_dir = temp_dirs / "reports"
        chunks_dir.mkdir(parents=True)
        reports_dir.mkdir(parents=True)
        
        return {
            'base': temp_dirs,
            'chunks': chunks_dir,
            'reports': reports_dir,
            'chunks_json': chunks_dir / "test_chunks.json"
        }
    
    def test_quality_enhancement_manager_initialization(self):
        """Test that QualityEnhancementManager initializes correctly."""
        markdown_manager = MarkdownFileManager()
        enhancement_manager = QualityEnhancementManager(markdown_manager)
        
        assert enhancement_manager.markdown_manager == markdown_manager
        assert isinstance(enhancement_manager.evaluator, ChunkQualityEvaluator)
    
    def test_auto_enhance_chunks_below_threshold(self, sample_chunks, mock_output_paths):
        """Test auto-enhancement triggers when quality is below threshold."""
        markdown_manager = MarkdownFileManager()
        enhancement_manager = QualityEnhancementManager(markdown_manager)
        
        # Mock quality metrics indicating poor quality
        poor_quality_metrics = {
            'overall_score': 45,  # Below 60 threshold
            'content_quality': {'incomplete_sentences': 0.5},  # 50% incomplete
            'semantic_coherence': {'coherence_score': 0.3}  # Low coherence
        }
        
        # Mock the evaluator to return improved metrics after enhancement
        with patch.object(enhancement_manager.evaluator, 'evaluate_chunks') as mock_evaluate:
            mock_evaluate.return_value = {'overall_score': 75}  # Improved score
            
            result = enhancement_manager.auto_enhance_chunks(
                sample_chunks, 
                poor_quality_metrics, 
                mock_output_paths
            )
        
        # Verify enhancement was attempted
        assert result['original_score'] == 45
        assert result['enhanced_score'] == 75
        assert len(result['improvements_made']) > 0
        assert 'Quality below threshold - applying enhancements' in result['improvements_made']
    
    def test_auto_enhance_chunks_above_threshold(self, sample_chunks, mock_output_paths):
        """Test auto-enhancement skips when quality is above threshold."""
        markdown_manager = MarkdownFileManager()
        enhancement_manager = QualityEnhancementManager(markdown_manager)
        
        # Mock quality metrics indicating good quality
        good_quality_metrics = {
            'overall_score': 85,  # Above 60 threshold
            'content_quality': {'incomplete_sentences': 0.1},
            'semantic_coherence': {'coherence_score': 0.8}
        }
        
        result = enhancement_manager.auto_enhance_chunks(
            sample_chunks, 
            good_quality_metrics, 
            mock_output_paths
        )
        
        # Verify no enhancement was attempted
        assert result['original_score'] == 85
        assert result['enhanced_score'] == 0  # Not enhanced
        assert len(result['improvements_made']) == 0
    
    def test_fix_incomplete_sentences(self, temp_dirs):
        """Test fixing incomplete sentences by merging with next chunk."""
        markdown_manager = MarkdownFileManager()
        enhancement_manager = QualityEnhancementManager(markdown_manager)
        
        # Create chunks with incomplete sentence
        chunks = [
            Document(
                page_content="This sentence is incomplete and doesn't end",
                metadata={"chunk_index": 0}
            ),
            Document(
                page_content="properly. This is the continuation with more content.",
                metadata={"chunk_index": 1}
            )
        ]
        
        enhanced_chunks = enhancement_manager._fix_incomplete_sentences(chunks)
        
        # Verify first chunk now has complete sentence
        assert enhanced_chunks[0].page_content.endswith("properly.")
        assert "This sentence is incomplete and doesn't end properly." in enhanced_chunks[0].page_content
        
        # Verify second chunk has remaining content
        assert "This is the continuation with more content." in enhanced_chunks[1].page_content
        assert not enhanced_chunks[1].page_content.startswith("properly.")
    
    def test_fix_incomplete_sentences_no_next_chunk(self, temp_dirs):
        """Test handling incomplete sentences when no next chunk available."""
        markdown_manager = MarkdownFileManager()
        enhancement_manager = QualityEnhancementManager(markdown_manager)
        
        # Create single chunk with incomplete sentence
        chunks = [
            Document(
                page_content="This sentence is incomplete and doesn't end",
                metadata={"chunk_index": 0}
            )
        ]
        
        enhanced_chunks = enhancement_manager._fix_incomplete_sentences(chunks)
        
        # Verify chunk remains unchanged when no next chunk available
        assert enhanced_chunks[0].page_content == chunks[0].page_content
    
    def test_improve_coherence_short_chunks(self, temp_dirs):
        """Test improving coherence by adding context to short chunks."""
        markdown_manager = MarkdownFileManager()
        enhancement_manager = QualityEnhancementManager(markdown_manager)
        
        chunks = [
            Document(
                page_content="This is a longer chunk with substantial content. It has multiple sentences.",
                metadata={"chunk_index": 0}
            ),
            Document(
                page_content="Short chunk.",  # Very short, should get context
                metadata={"chunk_index": 1}
            )
        ]
        
        enhanced_chunks = enhancement_manager._improve_coherence(chunks)
        
        # Verify short chunk received context from previous chunk
        assert len(enhanced_chunks[1].page_content.split()) > len(chunks[1].page_content.split())
        assert "It has multiple sentences." in enhanced_chunks[1].page_content
        assert "Short chunk." in enhanced_chunks[1].page_content
    
    def test_improve_coherence_no_previous_chunk(self, temp_dirs):
        """Test coherence improvement when no previous chunk available."""
        markdown_manager = MarkdownFileManager()
        enhancement_manager = QualityEnhancementManager(markdown_manager)
        
        chunks = [
            Document(
                page_content="Short.",  # First chunk, no previous context
                metadata={"chunk_index": 0}
            )
        ]
        
        enhanced_chunks = enhancement_manager._improve_coherence(chunks)
        
        # Verify chunk remains unchanged when no previous chunk
        assert enhanced_chunks[0].page_content == chunks[0].page_content
    
    def test_save_enhanced_chunks(self, mock_output_paths, sample_chunks):
        """Test saving enhanced chunks to designated location."""
        markdown_manager = MarkdownFileManager()
        enhancement_manager = QualityEnhancementManager(markdown_manager)
        
        # Mock FileHandler.save_chunks
        with patch('src.utils.file_handler.FileHandler.save_chunks') as mock_save:
            enhancement_manager._save_enhanced_chunks(sample_chunks, mock_output_paths)
        
        # Verify save_chunks was called with correct parameters
        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args[0][0] == sample_chunks  # chunks parameter
        assert 'enhanced_chunks' in call_args[0][1]  # file path contains 'enhanced_chunks'
        assert call_args[0][2] == 'json'  # format parameter
    
    def test_enhancement_with_multiple_issues(self, temp_dirs):
        """Test enhancement handling multiple quality issues simultaneously."""
        markdown_manager = MarkdownFileManager()
        enhancement_manager = QualityEnhancementManager(markdown_manager)
        
        # Create chunks with multiple issues
        chunks = [
            Document(
                page_content="This is good content with proper ending.",
                metadata={"chunk_index": 0}
            ),
            Document(
                page_content="This incomplete",  # Incomplete sentence
                metadata={"chunk_index": 1}
            ),
            Document(
                page_content="sentence continues here. More content follows.",
                metadata={"chunk_index": 2}
            ),
            Document(
                page_content="Short.",  # Very short
                metadata={"chunk_index": 3}
            )
        ]
        
        # Apply both fixes
        fixed_sentences = enhancement_manager._fix_incomplete_sentences(chunks)
        enhanced_chunks = enhancement_manager._improve_coherence(fixed_sentences)
        
        # Verify both issues were addressed
        # Incomplete sentence should be fixed
        assert "This incomplete sentence continues here." in enhanced_chunks[1].page_content
        
        # Short chunk should have additional context from previous chunk
        # After sentence fixing, chunk 2 should contain "More content follows"
        # Short chunk (index 3) should get context from chunk 2 
        original_short_length = len(chunks[3].page_content.split())
        enhanced_short_length = len(enhanced_chunks[3].page_content.split())
        
        # Either it got context (longer) or it stayed the same if no context was available
        assert enhanced_short_length >= original_short_length
        
        # If it got enhanced, it should contain context from previous chunk
        if enhanced_short_length > original_short_length:
            assert 'More content follows' in enhanced_chunks[3].page_content or 'sentence continues here' in enhanced_chunks[3].page_content
    
    def test_enhancement_preserves_metadata(self, temp_dirs):
        """Test that enhancement preserves original chunk metadata."""
        markdown_manager = MarkdownFileManager()
        enhancement_manager = QualityEnhancementManager(markdown_manager)
        
        original_metadata = {
            "chunk_index": 0,
            "source_file": "test.md",
            "chunk_id": "abc123",
            "custom_field": "custom_value"
        }
        
        chunks = [
            Document(
                page_content="This incomplete",
                metadata=original_metadata.copy()
            ),
            Document(
                page_content="sentence continues.",
                metadata={"chunk_index": 1}
            )
        ]
        
        enhanced_chunks = enhancement_manager._fix_incomplete_sentences(chunks)
        
        # Verify metadata is preserved
        for key, value in original_metadata.items():
            assert enhanced_chunks[0].metadata[key] == value
    
    def test_enhancement_performance_with_large_chunks(self, temp_dirs):
        """Test enhancement performance with large number of chunks."""
        markdown_manager = MarkdownFileManager()
        enhancement_manager = QualityEnhancementManager(markdown_manager)
        
        # Create large number of chunks
        chunks = []
        for i in range(1000):
            content = f"This is chunk {i} with some content. "
            if i % 10 == 0:  # Make every 10th chunk incomplete
                content = content.rstrip('. ')
            
            chunks.append(Document(
                page_content=content,
                metadata={"chunk_index": i}
            ))
        
        import time
        start_time = time.time()
        enhanced_chunks = enhancement_manager._fix_incomplete_sentences(chunks)
        end_time = time.time()
        
        # Verify performance is reasonable (should complete within 5 seconds)
        assert end_time - start_time < 5.0
        assert len(enhanced_chunks) == len(chunks)


class TestQualityEnhancementIntegration:
    """Integration tests for quality enhancement with the main system."""
    
    @pytest.fixture
    def temp_setup(self):
        """Setup for integration tests."""
        input_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        
        yield {
            'input_dir': Path(input_dir),
            'output_dir': Path(output_dir)
        }
        
        shutil.rmtree(input_dir)
        shutil.rmtree(output_dir)
    
    def test_end_to_end_quality_enhancement(self, temp_setup):
        """Test complete workflow with quality enhancement."""
        from src.chunkers.hybrid_chunker import HybridMarkdownChunker
        
        # Create test content with quality issues
        test_content = """# Test Book

This is a paragraph that ends properly.

This paragraph has an incomplete sentence that doesn't

finish correctly. This continues the thought.

## Chapter 2

Short.

This is another paragraph with good content and proper structure.
"""
        
        # Create chunks with quality issues
        chunker = HybridMarkdownChunker(chunk_size=50, chunk_overlap=10)  # Small chunks to force issues
        chunks = chunker.chunk_document(test_content, {'source_file': 'test.md'})
        
        # Evaluate quality
        evaluator = ChunkQualityEvaluator()
        quality_metrics = evaluator.evaluate_chunks(chunks)
        
        # Set up enhancement
        markdown_manager = MarkdownFileManager()
        output_paths = markdown_manager.create_output_structure(temp_setup['output_dir'])
        enhancement_manager = QualityEnhancementManager(markdown_manager)
        
        # Apply enhancement if needed
        if quality_metrics.get('overall_score', 0) < 60:
            result = enhancement_manager.auto_enhance_chunks(
                chunks,
                quality_metrics,
                output_paths
            )
            
            # Verify enhancement was applied
            assert result['original_score'] < 60
            assert len(result['improvements_made']) > 0
            assert 'enhanced_chunks' in result
    
    def test_quality_enhancement_with_project_folders(self, temp_setup):
        """Test quality enhancement works with project folder structure."""
        # Create a test markdown file
        test_file = temp_setup['input_dir'] / "test_quality.md"
        test_file.write_text("# Test\n\nIncomplete content that doesn't end\n\nproperly. More content here.")
        
        # Create project structure
        markdown_manager = MarkdownFileManager()
        output_paths = markdown_manager.create_markdown_output_paths(
            test_file,
            temp_setup['output_dir'],
            create_project_folder=True
        )
        
        # Create some test chunks
        from langchain_core.documents import Document
        chunks = [
            Document(page_content="Incomplete content", metadata={"chunk_index": 0}),
            Document(page_content="that continues here.", metadata={"chunk_index": 1})
        ]
        
        # Apply enhancement
        enhancement_manager = QualityEnhancementManager(markdown_manager)
        
        # Mock poor quality to trigger enhancement
        poor_metrics = {
            'overall_score': 40,
            'content_quality': {'incomplete_sentences': 0.6},
            'semantic_coherence': {'coherence_score': 0.2}
        }
        
        with patch.object(enhancement_manager.evaluator, 'evaluate_chunks') as mock_eval:
            mock_eval.return_value = {'overall_score': 80}
            
            result = enhancement_manager.auto_enhance_chunks(
                chunks,
                poor_metrics,
                output_paths
            )
        
        # Verify enhancement worked within project structure
        assert result['enhanced_score'] > result['original_score']
        assert output_paths['project_folder'] is not None
        assert output_paths['project_folder'].exists()


class TestHolisticChunkingStrategy:
    """Test cases for holistic chunking strategy enhancement - TDD approach."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        output_dir = tempfile.mkdtemp()
        yield Path(output_dir)
        shutil.rmtree(output_dir)
    
    def test_content_analysis_framework_exists(self):
        """Test that content analysis framework exists and works."""
        # This test will fail initially - defining expected behavior
        from src.chunkers.strategy_optimizer import StrategyOptimizer
        
        optimizer = StrategyOptimizer()
        
        test_content = """# Technical Document
        
This document contains code:
```python
def example():
    return "hello"
```

And regular text with multiple sentences. Some lists:
- Item 1
- Item 2

## Section 2
More content here.
"""
        
        # Expected behavior: analyze content characteristics
        analysis = optimizer.analyze_content_characteristics(test_content)
        
        # Should return analysis metrics
        assert isinstance(analysis, dict)
        assert 'sentence_length_avg' in analysis
        assert 'paragraph_length_avg' in analysis
        assert 'code_density' in analysis
        assert 'header_frequency' in analysis
        assert 'list_density' in analysis
        assert 'technical_complexity' in analysis
        
        # Verify analysis makes sense
        assert analysis['code_density'] > 0  # Should detect code
        assert analysis['header_frequency'] > 0  # Should detect headers
        assert analysis['list_density'] > 0  # Should detect lists
    
    def test_strategy_recommendation_system(self):
        """Test that strategy recommendation system works based on content analysis."""
        # This test will fail initially - defining expected behavior
        from src.chunkers.strategy_optimizer import StrategyOptimizer
        
        optimizer = StrategyOptimizer()
        
        # Test different content types
        code_heavy_content = """```python
def function1():
    pass

def function2():
    pass
```"""
        
        narrative_content = """This is a long narrative text with multiple paragraphs.
Each paragraph contains several sentences that flow naturally.
The content is primarily prose without technical elements."""
        
        structured_content = """# Chapter 1
## Section 1.1
Content here.
## Section 1.2
More content.
### Subsection 1.2.1
Detailed content."""
        
        # Test code-heavy content
        code_analysis = optimizer.analyze_content_characteristics(code_heavy_content)
        code_recommendation = optimizer.recommend_strategy(code_analysis)
        
        assert isinstance(code_recommendation, dict)
        assert 'primary_strategy' in code_recommendation
        assert 'chunk_size' in code_recommendation
        assert 'overlap_size' in code_recommendation
        assert 'separators' in code_recommendation
        assert 'confidence' in code_recommendation
        
        # Should recommend content-aware strategy for code
        assert code_recommendation['primary_strategy'] == 'content_aware'
        assert code_recommendation['chunk_size'] <= 800  # Smaller chunks for code
        
        # Test narrative content
        narrative_analysis = optimizer.analyze_content_characteristics(narrative_content)
        narrative_recommendation = optimizer.recommend_strategy(narrative_analysis)
        
        # Should recommend sentence-boundary strategy for narrative
        assert narrative_recommendation['primary_strategy'] == 'sentence_boundary'
        
        # Test structured content
        structured_analysis = optimizer.analyze_content_characteristics(structured_content)
        structured_recommendation = optimizer.recommend_strategy(structured_analysis)
        
        # Should recommend semantic-boundary strategy for structured content
        assert structured_recommendation['primary_strategy'] == 'semantic_boundary'
        assert structured_recommendation['chunk_size'] >= 800  # Larger chunks for structured
    
    def test_multi_strategy_comparison_framework(self):
        """Test framework that compares multiple chunking strategies."""
        # This test will fail initially - defining expected behavior
        from src.chunkers.strategy_tester import StrategyTester
        
        tester = StrategyTester()
        
        test_content = """# Test Document

This is test content with various characteristics.

## Code Section
```python
def example():
    return "test"
```

## Regular Text
This is regular text with multiple sentences. It should be chunked appropriately
based on the chosen strategy.

## Lists
- Item 1: Description
- Item 2: More details
- Item 3: Final item

## Conclusion
The document ends here.
"""
        
        # Test multiple strategies
        strategies = ['fixed_size', 'semantic_boundary', 'content_aware', 'sentence_boundary']
        results = tester.test_multiple_strategies(test_content, strategies)
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'best_strategy' in results
        assert 'all_results' in results
        assert 'recommendations' in results
        
        # Verify all strategies were tested
        assert len(results['all_results']) == len(strategies)
        
        # Verify each strategy result has required fields
        for strategy, result in results['all_results'].items():
            if 'error' not in result:
                assert 'chunks' in result
                assert 'quality_metrics' in result
                assert 'parameters_used' in result
                assert 'overall_score' in result
                assert isinstance(result['chunks'], list)
                assert isinstance(result['quality_metrics'], dict)
                assert isinstance(result['overall_score'], (int, float))
        
        # Verify best strategy has highest score
        if results['best_strategy']:
            best_score = results['all_results'][results['best_strategy']]['overall_score']
            for strategy, result in results['all_results'].items():
                if strategy != results['best_strategy'] and 'error' not in result:
                    assert best_score >= result['overall_score']
    
    def test_adaptive_chunker_with_strategy_optimization(self):
        """Test adaptive chunker that selects optimal strategy based on content."""
        # This test will fail initially - defining expected behavior
        from src.chunkers.adaptive_chunker import AdaptiveChunker
        
        chunker = AdaptiveChunker(auto_optimize=True)
        
        # Test with different content types
        test_cases = {
            'code_heavy': '''# API Documentation
```python
class APIClient:
    def __init__(self, api_key):
        self.api_key = api_key
    
    def make_request(self, endpoint):
        pass
```

More code examples follow...''',
            
            'narrative_text': '''This is a long narrative text with multiple paragraphs. 
Each paragraph contains several sentences that flow naturally from one to the next.
The content is primarily prose without technical elements.

Another paragraph continues the narrative with descriptive content.''',
            
            'structured_document': '''# Chapter 1: Introduction

## 1.1 Overview
Content about the overview topic.

## 1.2 Objectives  
Content about objectives.

### 1.2.1 Primary Goals
Detailed content about primary goals.'''
        }
        
        for content_type, content in test_cases.items():
            chunks = chunker.chunk_document_adaptive(content)
            
            # Verify chunks were generated
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            
            # Verify chunk structure
            for chunk in chunks:
                assert hasattr(chunk, 'page_content')
                assert hasattr(chunk, 'metadata')
                assert len(chunk.page_content.strip()) > 0
                assert 'chunk_index' in chunk.metadata
                
                # Verify strategy information is recorded
                assert 'strategy_used' in chunk.metadata
                assert 'optimization_applied' in chunk.metadata
    
    def test_comprehensive_quality_enhancement_with_strategy_selection(self):
        """Test comprehensive enhancement that includes strategy optimization."""
        # This test will fail initially - defining expected behavior
        from src.utils.path_utils import AdvancedQualityEnhancementManager, MarkdownFileManager
        
        # Create problematic content that should trigger strategy optimization
        problematic_content = '''# Test Document

This is content that doesn't end properly and creates quality issues like

Short.

Another incomplete sentence that

continues strangely. More content that should be improved through the enhancement process.

## Section 2

Very short content.

End.'''
        
        # Setup
        markdown_manager = MarkdownFileManager()
        temp_dir = Path(tempfile.mkdtemp())
        output_paths = markdown_manager.create_output_structure(temp_dir)
        
        try:
            # Create initial chunks with quality issues
            from src.chunkers.hybrid_chunker import HybridMarkdownChunker
            chunker = HybridMarkdownChunker(chunk_size=30, chunk_overlap=5)  # Force small chunks
            initial_chunks = chunker.chunk_document(problematic_content)
            
            # Evaluate initial quality
            from src.chunkers.evaluators import ChunkQualityEvaluator
            evaluator = ChunkQualityEvaluator()
            initial_metrics = evaluator.evaluate_chunks(initial_chunks)
            
            # Apply comprehensive enhancement
            enhancement_manager = AdvancedQualityEnhancementManager(markdown_manager)
            results = enhancement_manager.comprehensive_enhancement(
                problematic_content,
                initial_chunks,
                initial_metrics,
                output_paths
            )
            
            # Verify enhancement results structure
            assert isinstance(results, dict)
            assert 'original_score' in results
            assert 'final_score' in results
            assert 'strategy_tested' in results
            assert 'rechunked' in results
            assert 'improvements_made' in results
            assert 'final_chunks' in results
            
            # Verify improvement was made
            assert results['final_score'] >= results['original_score']
            assert isinstance(results['improvements_made'], list)
            
            # If quality was poor, strategy testing should have been attempted
            if results['original_score'] < 60:
                assert results['strategy_tested'] is True
                
            # Verify final chunks are valid
            assert isinstance(results['final_chunks'], list)
            assert len(results['final_chunks']) > 0
            
            for chunk in results['final_chunks']:
                assert hasattr(chunk, 'page_content')
                assert hasattr(chunk, 'metadata')
                assert len(chunk.page_content.strip()) > 0
        
        finally:
            shutil.rmtree(temp_dir)
    
    def test_advanced_quality_metrics_for_strategy_evaluation(self):
        """Test advanced quality metrics that evaluate strategy effectiveness."""
        # This test will fail initially - defining expected behavior
        from src.chunkers.evaluators import AdvancedQualityEvaluator
        from langchain_core.documents import Document
        
        evaluator = AdvancedQualityEvaluator()
        
        # Create test chunks with different characteristics
        test_chunks = [
            Document(
                page_content="This is a complete sentence with proper structure. It contains multiple clauses and flows well.",
                metadata={"chunk_index": 0, "strategy_used": "sentence_boundary"}
            ),
            Document(
                page_content="```python\ndef example():\n    return 'test'\n```",
                metadata={"chunk_index": 1, "strategy_used": "content_aware"}
            ),
            Document(
                page_content="Short fragment without proper ending",
                metadata={"chunk_index": 2, "strategy_used": "fixed_size"}
            ),
            Document(
                page_content="This continues the previous thought and provides context. It demonstrates good chunk continuity.",
                metadata={"chunk_index": 3, "strategy_used": "semantic_boundary"}
            )
        ]
        
        # Test strategy-specific evaluation
        strategy_metrics = evaluator.evaluate_strategy_effectiveness(test_chunks, "mixed_strategy")
        
        # Verify advanced metrics are included
        assert isinstance(strategy_metrics, dict)
        assert 'strategy_used' in strategy_metrics
        assert 'boundary_preservation' in strategy_metrics
        assert 'context_continuity' in strategy_metrics
        assert 'information_density' in strategy_metrics
        assert 'readability_scores' in strategy_metrics
        assert 'topic_coherence' in strategy_metrics
        assert 'chunk_independence' in strategy_metrics
        
        # Verify boundary preservation metrics
        boundary_metrics = strategy_metrics['boundary_preservation']
        assert 'sentence_boundary_score' in boundary_metrics
        assert 'paragraph_boundary_score' in boundary_metrics
        assert 'section_boundary_score' in boundary_metrics
        
        # Verify context continuity metrics
        context_metrics = strategy_metrics['context_continuity']
        assert 'topic_transition_smoothness' in context_metrics
        assert 'reference_completeness' in context_metrics
        assert 'narrative_flow' in context_metrics
        
        # Verify all scores are numeric
        for metric_group in [boundary_metrics, context_metrics]:
            for score in metric_group.values():
                assert isinstance(score, (int, float))
                assert 0 <= score <= 1  # Assuming normalized scores
    
    def test_performance_benchmarking_for_different_strategies(self):
        """Test performance benchmarking across different chunking strategies."""
        # This test will fail initially - defining expected behavior
        from src.chunkers.strategy_tester import StrategyTester
        import time
        
        tester = StrategyTester()
        
        # Test with different content sizes
        base_content = "This is test content with multiple sentences. Each sentence provides meaningful information. "
        content_sizes = [1000, 5000, 10000]  # character counts
        
        performance_results = {}
        
        for size in content_sizes:
            # Create content of specified size
            content = (base_content * (size // len(base_content)))[:size]
            
            # Measure performance
            start_time = time.time()
            results = tester.test_multiple_strategies(content)
            end_time = time.time()
            
            performance_results[size] = {
                'processing_time': end_time - start_time,
                'strategies_tested': len(results['all_results']),
                'best_strategy': results['best_strategy'],
                'best_score': results['all_results'][results['best_strategy']]['overall_score'] if results['best_strategy'] else 0
            }
            
            # Verify performance is reasonable
            assert performance_results[size]['processing_time'] < 30.0  # Should complete within 30 seconds
            assert performance_results[size]['strategies_tested'] > 0
            assert performance_results[size]['best_score'] >= 0
        
        # Verify performance scales reasonably with content size
        times = [perf['processing_time'] for perf in performance_results.values()]
        assert times[1] > times[0]  # Larger content should take more time
        assert times[2] > times[1]  # But not exponentially more
    
    def test_strategy_caching_and_optimization(self):
        """Test that strategy selection results are cached for similar content."""
        # This test will fail initially - defining expected behavior
        from src.chunkers.strategy_optimizer import StrategyOptimizer
        import time
        
        optimizer = StrategyOptimizer(enable_caching=True)
        
        # Similar content should use cached results
        content1 = "This is technical documentation with code examples and structured content."
        content2 = "This is technical documentation with code samples and organized content."
        
        # First analysis
        start_time1 = time.time()
        analysis1 = optimizer.analyze_content_characteristics(content1)
        recommendation1 = optimizer.recommend_strategy(analysis1)
        end_time1 = time.time()
        
        # Second analysis (should use cache)
        start_time2 = time.time()
        analysis2 = optimizer.analyze_content_characteristics(content2)
        recommendation2 = optimizer.recommend_strategy(analysis2)
        end_time2 = time.time()
        
        # Verify caching worked
        assert recommendation1['primary_strategy'] == recommendation2['primary_strategy']
        assert end_time2 - start_time2 < end_time1 - start_time1  # Second should be faster
        
        # Verify cache hit information
        assert 'cache_hit' in recommendation2
        assert recommendation2['cache_hit'] is True
    
    def test_error_handling_in_strategy_selection(self):
        """Test error handling when strategy selection fails."""
        # This test will fail initially - defining expected behavior
        from src.chunkers.strategy_tester import StrategyTester
        
        tester = StrategyTester()
        
        # Test with problematic content
        problematic_cases = [
            "",  # Empty content
            "A",  # Single character
            "   ",  # Only whitespace
            "🚀" * 1000,  # Non-ASCII content
        ]
        
        for content in problematic_cases:
            results = tester.test_multiple_strategies(content)
            
            # Should handle errors gracefully
            assert isinstance(results, dict)
            assert 'best_strategy' in results
            assert 'all_results' in results
            
            # Should either find a working strategy or handle all failures
            if results['best_strategy']:
                assert results['all_results'][results['best_strategy']]['overall_score'] >= 0
            else:
                # All strategies failed, but results should still be structured
                for strategy_result in results['all_results'].values():
                    assert 'error' in strategy_result or 'overall_score' in strategy_result