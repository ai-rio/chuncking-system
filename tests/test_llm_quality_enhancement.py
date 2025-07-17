"""
Test cases for LLM-powered quality enhancement system.
Following TDD principles - tests written first to define expected behavior.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.documents import Document

from src.utils.path_utils import MarkdownFileManager
from src.chunkers.evaluators import ChunkQualityEvaluator


class TestLLMQualityEnhancer:
    """Test LLM-powered quality enhancement functionality."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        output_dir = tempfile.mkdtemp()
        yield Path(output_dir)
        shutil.rmtree(output_dir)
    
    @pytest.fixture
    def sample_poor_chunks(self):
        """Create sample chunks with quality issues for enhancement."""
        return [
            # Incomplete sentence
            Document(
                page_content="This is an incomplete sentence that doesn't properly",
                metadata={"chunk_index": 0, "source": "test"}
            ),
            # Continuation chunk
            Document(
                page_content="end and continues awkwardly into this chunk. More content here.",
                metadata={"chunk_index": 1, "source": "test"}
            ),
            # Very short chunk
            Document(
                page_content="Short.",
                metadata={"chunk_index": 2, "source": "test"}
            ),
            # Context-less chunk
            Document(
                page_content="This refers to something mentioned earlier without context.",
                metadata={"chunk_index": 3, "source": "test"}
            ),
            # Fragmented content
            Document(
                page_content="The concept\n\nwas discussed but\n\nnot fully explained.",
                metadata={"chunk_index": 4, "source": "test"}
            )
        ]
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for testing."""
        return {
            "enhanced_content": "This is an enhanced version of the content with improved clarity and coherence.",
            "improvements_made": [
                "Completed incomplete sentences",
                "Improved sentence structure",
                "Enhanced readability",
                "Added contextual clarity"
            ],
            "quality_score": 85.0,
            "reasoning": "The content has been improved by fixing sentence structure and adding clarity."
        }
    
    def test_llm_quality_enhancer_initialization(self):
        """Test that LLMQualityEnhancer initializes correctly."""
        # This test will fail initially - defining expected behavior
        from src.utils.llm_quality_enhancer import LLMQualityEnhancer
        
        enhancer = LLMQualityEnhancer(
            llm_provider="openai",
            llm_model="gpt-4",
            temperature=0.1
        )
        
        assert enhancer.llm_provider == "openai"
        assert enhancer.llm_model == "gpt-4"
        assert enhancer.temperature == 0.1
        assert hasattr(enhancer, 'llm_client')
        assert hasattr(enhancer, 'enhancement_prompts')
    
    def test_llm_content_rewriting(self, sample_poor_chunks, mock_llm_response):
        """Test LLM-powered content rewriting functionality."""
        # This test will fail initially - defining expected behavior
        from src.utils.llm_quality_enhancer import LLMQualityEnhancer
        
        enhancer = LLMQualityEnhancer()
        
        # Mock LLM response
        with patch.object(enhancer, '_call_llm') as mock_llm:
            mock_llm.return_value = mock_llm_response
            
            # Test rewriting of poor content
            poor_chunk = sample_poor_chunks[0]
            enhanced_chunk = enhancer.enhance_chunk_content(poor_chunk)
            
            # Verify enhancement
            assert enhanced_chunk.page_content != poor_chunk.page_content
            assert len(enhanced_chunk.page_content) > len(poor_chunk.page_content)
            
            # Verify original metadata is preserved
            for key, value in poor_chunk.metadata.items():
                assert enhanced_chunk.metadata[key] == value
            
            # Verify enhancement metadata is added
            assert enhanced_chunk.metadata['llm_enhanced'] is True
            
            # Verify LLM was called with correct parameters
            mock_llm.assert_called_once()
            call_args = mock_llm.call_args[0]
            assert "improve" in call_args[0].lower()
            assert poor_chunk.page_content in call_args[0]
    
    def test_semantic_coherence_enhancement(self, sample_poor_chunks, mock_llm_response):
        """Test semantic coherence enhancement between chunks."""
        # This test will fail initially - defining expected behavior
        from src.utils.llm_quality_enhancer import LLMQualityEnhancer
        
        enhancer = LLMQualityEnhancer()
        
        with patch.object(enhancer, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "enhanced_chunks": [
                    {
                        "content": "This is an enhanced sentence that properly connects to the next chunk.",
                        "metadata": {"chunk_index": 0, "source": "test"}
                    },
                    {
                        "content": "This continues the thought from the previous chunk and provides additional context.",
                        "metadata": {"chunk_index": 1, "source": "test"}
                    }
                ],
                "coherence_score": 92.0,
                "transitions_added": 2
            }
            
            # Test coherence enhancement
            enhanced_chunks = enhancer.enhance_semantic_coherence(sample_poor_chunks[:2])
            
            # Verify enhancement
            assert len(enhanced_chunks) == 2
            assert all(isinstance(chunk, Document) for chunk in enhanced_chunks)
            
            # Verify coherence improvements
            for i, chunk in enumerate(enhanced_chunks):
                assert chunk.page_content != sample_poor_chunks[i].page_content
                # Check that original metadata is preserved
                original_metadata = sample_poor_chunks[i].metadata
                for key, value in original_metadata.items():
                    assert chunk.metadata[key] == value
                # Check that enhancement metadata is added
                assert chunk.metadata.get('llm_enhanced') is True
                assert chunk.metadata.get('coherence_enhanced') is True
                assert 'coherence_score' in chunk.metadata
                assert len(chunk.page_content) > 0
    
    def test_contextual_gap_filling(self, sample_poor_chunks, mock_llm_response):
        """Test contextual gap filling using LLM."""
        # This test will fail initially - defining expected behavior
        from src.utils.llm_quality_enhancer import LLMQualityEnhancer
        
        enhancer = LLMQualityEnhancer()
        
        with patch.object(enhancer, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "gap_analysis": {
                    "gaps_found": 2,
                    "gap_descriptions": [
                        "Missing context between chunks 2 and 3",
                        "Unclear reference in chunk 3"
                    ]
                },
                "enhanced_chunks": [
                    {
                        "content": "Enhanced content with proper context.",
                        "metadata": {"chunk_index": 2, "source": "test"}
                    },
                    {
                        "content": "This refers to the concept mentioned in the previous section, which explains...",
                        "metadata": {"chunk_index": 3, "source": "test"}
                    }
                ],
                "context_score": 88.0
            }
            
            # Test gap filling
            result = enhancer.fill_contextual_gaps(sample_poor_chunks[2:4])
            
            # Verify gap filling
            assert "gap_analysis" in result
            assert "enhanced_chunks" in result
            assert "context_score" in result
            
            # Verify gaps were identified and filled
            assert result["gap_analysis"]["gaps_found"] > 0
            assert len(result["enhanced_chunks"]) == 2
    
    def test_content_completeness_validation(self, sample_poor_chunks):
        """Test content completeness validation using LLM."""
        # This test will fail initially - defining expected behavior
        from src.utils.llm_quality_enhancer import LLMQualityEnhancer
        
        enhancer = LLMQualityEnhancer()
        
        with patch.object(enhancer, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "completeness_analysis": {
                    "complete_chunks": 2,
                    "incomplete_chunks": 3,
                    "missing_elements": [
                        "Incomplete sentences in chunk 0",
                        "Missing context in chunk 3",
                        "Fragmented content in chunk 4"
                    ]
                },
                "completeness_score": 45.0,
                "recommendations": [
                    "Complete sentence fragments",
                    "Add contextual references",
                    "Improve content flow"
                ]
            }
            
            # Test completeness validation
            result = enhancer.validate_content_completeness(sample_poor_chunks)
            
            # Verify completeness analysis
            assert "completeness_analysis" in result
            assert "completeness_score" in result
            assert "recommendations" in result
            
            # Verify analysis results
            assert result["completeness_score"] >= 0
            assert result["completeness_score"] <= 100
            assert len(result["recommendations"]) > 0
    
    def test_llm_quality_metrics_calculation(self, sample_poor_chunks):
        """Test LLM-powered quality metrics calculation."""
        # This test will fail initially - defining expected behavior
        from src.utils.llm_quality_enhancer import LLMQualityEnhancer
        
        enhancer = LLMQualityEnhancer()
        
        with patch.object(enhancer, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "quality_metrics": {
                    "readability_score": 72.5,
                    "coherence_score": 68.0,
                    "completeness_score": 45.0,
                    "context_clarity": 55.0,
                    "overall_quality": 60.1
                },
                "detailed_analysis": {
                    "strengths": ["Clear vocabulary", "Proper grammar"],
                    "weaknesses": ["Incomplete sentences", "Missing context"],
                    "suggestions": ["Complete fragments", "Add transitions"]
                }
            }
            
            # Test quality metrics calculation
            result = enhancer.calculate_llm_quality_metrics(sample_poor_chunks)
            
            # Verify metrics structure
            assert "quality_metrics" in result
            assert "detailed_analysis" in result
            
            # Verify all key metrics are present
            metrics = result["quality_metrics"]
            assert "readability_score" in metrics
            assert "coherence_score" in metrics
            assert "completeness_score" in metrics
            assert "context_clarity" in metrics
            assert "overall_quality" in metrics
            
            # Verify scores are valid ranges
            for score in metrics.values():
                assert 0 <= score <= 100
    
    def test_comprehensive_llm_enhancement(self, sample_poor_chunks, temp_dirs):
        """Test comprehensive LLM enhancement pipeline."""
        # This test will fail initially - defining expected behavior
        from src.utils.llm_quality_enhancer import LLMQualityEnhancer
        
        enhancer = LLMQualityEnhancer()
        
        # Mock comprehensive enhancement response
        with patch.object(enhancer, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "enhanced_chunks": [
                    {
                        "content": "This is a complete sentence that properly connects to the next section.",
                        "metadata": {"chunk_index": 0, "source": "test", "enhanced": True}
                    },
                    {
                        "content": "This continues the thought and provides comprehensive context about the topic.",
                        "metadata": {"chunk_index": 1, "source": "test", "enhanced": True}
                    }
                ],
                "enhancement_summary": {
                    "original_score": 45.0,
                    "enhanced_score": 82.0,
                    "improvement": 37.0,
                    "enhancements_applied": [
                        "Content rewriting",
                        "Semantic coherence",
                        "Contextual gap filling",
                        "Sentence completion"
                    ]
                }
            }
            
            # Test comprehensive enhancement
            result = enhancer.comprehensive_enhance(sample_poor_chunks)
            
            # Verify enhancement results
            assert "enhanced_chunks" in result
            assert "enhancement_summary" in result
            
            # Verify significant improvement
            summary = result["enhancement_summary"]
            assert summary["enhanced_score"] > summary["original_score"]
            assert summary["improvement"] > 30  # Significant improvement expected
            
            # Verify enhanced chunks are valid
            enhanced_chunks = result["enhanced_chunks"]
            assert len(enhanced_chunks) > 0
            
            for chunk_data in enhanced_chunks:
                assert "content" in chunk_data
                assert "metadata" in chunk_data
                assert len(chunk_data["content"]) > 0
    
    def test_llm_enhancement_with_different_content_types(self):
        """Test LLM enhancement with different content types."""
        # This test will fail initially - defining expected behavior
        from src.utils.llm_quality_enhancer import LLMQualityEnhancer
        
        enhancer = LLMQualityEnhancer()
        
        test_cases = {
            'technical': Document(
                page_content="The API endpoint returns JSON data but",
                metadata={"content_type": "technical", "chunk_index": 0}
            ),
            'narrative': Document(
                page_content="The story begins with a character who",
                metadata={"content_type": "narrative", "chunk_index": 0}
            ),
            'code': Document(
                page_content="```python\ndef incomplete_function(\n",
                metadata={"content_type": "code", "chunk_index": 0}
            )
        }
        
        for content_type, chunk in test_cases.items():
            with patch.object(enhancer, '_call_llm') as mock_llm:
                mock_llm.return_value = {
                    "enhanced_content": f"Enhanced {content_type} content with proper completion.",
                    "content_type_specific_improvements": [
                        f"Applied {content_type}-specific enhancements"
                    ]
                }
                
                # Test content-type specific enhancement
                enhanced = enhancer.enhance_chunk_content(chunk)
                
                # Verify enhancement
                assert enhanced.page_content != chunk.page_content
                assert enhanced.metadata["content_type"] == content_type
                
                # Verify LLM was called with content type context
                mock_llm.assert_called_once()
                call_args = mock_llm.call_args[0]
                assert content_type in call_args[0].lower()
    
    def test_llm_error_handling(self, sample_poor_chunks):
        """Test error handling in LLM enhancement."""
        # This test will fail initially - defining expected behavior
        from src.utils.llm_quality_enhancer import LLMQualityEnhancer
        
        enhancer = LLMQualityEnhancer()
        
        # Test with LLM API failure
        with patch.object(enhancer, '_call_llm') as mock_llm:
            mock_llm.side_effect = Exception("API Error")
            
            # Should handle error gracefully
            result = enhancer.enhance_chunk_content(sample_poor_chunks[0])
            
            # Should return original chunk or error indication
            assert result is not None
            assert hasattr(result, 'page_content')
            assert hasattr(result, 'metadata')
    
    def test_llm_enhancement_preserves_metadata(self, sample_poor_chunks):
        """Test that LLM enhancement preserves important metadata."""
        # This test will fail initially - defining expected behavior
        from src.utils.llm_quality_enhancer import LLMQualityEnhancer
        
        enhancer = LLMQualityEnhancer()
        
        # Add rich metadata to test chunk
        rich_chunk = Document(
            page_content="This is content that needs enhancement",
            metadata={
                "chunk_index": 0,
                "source": "test.md",
                "chunk_id": "abc123",
                "custom_field": "important_value",
                "processing_timestamp": "2024-01-01T00:00:00Z"
            }
        )
        
        with patch.object(enhancer, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "enhanced_content": "This is enhanced content with improved clarity.",
                "metadata_preserved": True
            }
            
            # Test enhancement
            enhanced = enhancer.enhance_chunk_content(rich_chunk)
            
            # Verify all metadata is preserved
            for key, value in rich_chunk.metadata.items():
                assert key in enhanced.metadata
                assert enhanced.metadata[key] == value
            
            # Verify enhancement markers are added
            assert "llm_enhanced" in enhanced.metadata
            assert enhanced.metadata["llm_enhanced"] is True
    
    def test_llm_enhancement_performance_tracking(self, sample_poor_chunks):
        """Test performance tracking for LLM enhancement."""
        # This test will fail initially - defining expected behavior
        from src.utils.llm_quality_enhancer import LLMQualityEnhancer
        
        enhancer = LLMQualityEnhancer()
        
        with patch.object(enhancer, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "enhanced_content": "Enhanced content",
                "performance_metrics": {
                    "processing_time": 1.5,
                    "tokens_used": 150,
                    "api_calls": 1
                }
            }
            
            # Test performance tracking
            result = enhancer.enhance_chunk_content(sample_poor_chunks[0])
            
            # Verify performance metrics are tracked
            assert "llm_processing_time" in result.metadata
            assert "llm_tokens_used" in result.metadata
            assert "llm_api_calls" in result.metadata
            
            # Verify metrics are reasonable
            assert result.metadata["llm_processing_time"] > 0
            assert result.metadata["llm_tokens_used"] > 0
            assert result.metadata["llm_api_calls"] > 0


class TestLLMQualityEnhancementIntegration:
    """Integration tests for LLM quality enhancement with existing system."""
    
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
    
    def test_integration_with_existing_quality_evaluator(self, temp_setup):
        """Test integration with existing ChunkQualityEvaluator."""
        # This test will fail initially - defining expected behavior
        from src.utils.llm_quality_enhancer import LLMQualityEnhancer
        from src.utils.path_utils import AdvancedQualityEnhancementManager, MarkdownFileManager
        
        # Create test chunks
        test_chunks = [
            Document(
                page_content="This is incomplete content that",
                metadata={"chunk_index": 0, "source": "test"}
            ),
            Document(
                page_content="continues here with more information.",
                metadata={"chunk_index": 1, "source": "test"}
            )
        ]
        
        # Mock LLM enhancer and evaluator before creating the manager
        with patch('src.utils.llm_quality_enhancer.LLMQualityEnhancer') as mock_enhancer_class, \
             patch('src.chunkers.evaluators.ChunkQualityEvaluator') as mock_evaluator_class:
            
            mock_enhancer = MagicMock()
            mock_enhancer.comprehensive_enhance.return_value = {
                "llm_enhanced": True,
                "enhanced_chunks": [
                    {
                        "content": "This is complete content that connects seamlessly with the next section.",
                        "metadata": {"chunk_index": 0, "source": "test", "llm_enhanced": True}
                    },
                    {
                        "content": "This continues here with comprehensive information and proper context.",
                        "metadata": {"chunk_index": 1, "source": "test", "llm_enhanced": True}
                    }
                ],
                "enhancement_summary": {
                    "original_score": 45.0,
                    "enhanced_score": 85.0,
                    "improvement": 40.0
                }
            }
            mock_enhancer_class.return_value = mock_enhancer
            
            # Mock evaluator to return high score for enhanced chunks
            mock_evaluator = MagicMock()
            mock_evaluator.evaluate_chunks.return_value = {"overall_score": 85.0}
            mock_evaluator_class.return_value = mock_evaluator
            
            # Test integration - create manager after mocking
            markdown_manager = MarkdownFileManager()
            enhancement_manager = AdvancedQualityEnhancementManager(markdown_manager)
            
            # Test enhancement with LLM integration
            output_paths = markdown_manager.create_output_structure(temp_setup['output_dir'])
            
            # This should use LLM enhancement
            result = enhancement_manager.llm_comprehensive_enhancement(
                test_chunks,
                {"overall_score": 45.0},
                output_paths
            )
            
            # Verify LLM enhancement was used
            assert result["enhanced_score"] > 80  # Significant improvement expected
            assert result["llm_enhanced"] is True
            assert len(result["final_chunks"]) == 2
    
    def test_llm_enhancement_with_different_providers(self):
        """Test LLM enhancement with different LLM providers."""
        # This test will fail initially - defining expected behavior
        from src.utils.llm_quality_enhancer import LLMQualityEnhancer
        
        providers = ['openai', 'anthropic', 'google', 'local']
        
        # Mock the LocalProvider validation to avoid network calls
        with patch('src.utils.llm_client.LocalProvider.validate_config', return_value=True):
            for provider in providers:
                enhancer = LLMQualityEnhancer(llm_provider=provider)
                
                # Verify provider-specific initialization
                assert enhancer.llm_provider == provider
                assert hasattr(enhancer, 'llm_client')
                
                # Test that provider-specific methods exist
                if provider == 'openai':
                    assert hasattr(enhancer, '_call_openai')
                elif provider == 'anthropic':
                    assert hasattr(enhancer, '_call_anthropic')
                elif provider == 'google':
                    assert hasattr(enhancer, '_call_google')
                elif provider == 'local':
                    assert hasattr(enhancer, '_call_local')
    
    def test_llm_enhancement_cost_optimization(self, temp_setup):
        """Test cost optimization features for LLM enhancement."""
        # This test will fail initially - defining expected behavior
        from src.utils.llm_quality_enhancer import LLMQualityEnhancer
        
        enhancer = LLMQualityEnhancer(
            cost_optimization=True,
            max_tokens_per_request=500,
            batch_processing=True
        )
        
        # Create large number of chunks
        large_chunk_set = [
            Document(
                page_content=f"Test content {i} that needs enhancement",
                metadata={"chunk_index": i, "source": "test"}
            )
            for i in range(50)
        ]
        
        with patch.object(enhancer, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "enhanced_chunks": [
                    {
                        "content": f"Enhanced content {i}",
                        "metadata": {"chunk_index": i, "source": "test"}
                    }
                    for i in range(50)
                ],
                "enhancement_summary": {
                    "enhanced_score": 85.0,
                    "improvement": 40.0,
                    "enhancements_applied": ["coherence", "clarity"]
                },
                "quality_validation": {},
                "cost_metrics": {
                    "total_tokens": 2500,
                    "api_calls": 5,  # Batched calls
                    "estimated_cost": 0.05
                }
            }
            
            # Test cost-optimized enhancement
            result = enhancer.comprehensive_enhance(large_chunk_set)
            
            # Verify cost optimization was applied
            assert mock_llm.call_count <= 10  # Should be batched
            assert "llm_enhanced" in result
            assert result["llm_enhanced"] is True
            assert len(result["enhanced_chunks"]) == 50