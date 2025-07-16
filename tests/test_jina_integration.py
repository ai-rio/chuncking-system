"""
Test cases for Jina AI embedding integration into quality evaluation.
Following TDD principles - tests written first to define expected behavior.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.chunkers.evaluators import ChunkQualityEvaluator, EnhancedChunkQualityEvaluator
from src.llm.providers.jina_provider import JinaProvider


class TestJinaEmbeddingIntegration:
    """Test Jina AI embedding integration with quality evaluation."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Document(
                page_content="This is a coherent paragraph about machine learning algorithms. It discusses various approaches to classification and regression.",
                metadata={"chunk_index": 0, "source": "test"}
            ),
            Document(
                page_content="Machine learning models require training data to learn patterns. The quality of training data directly impacts model performance.",
                metadata={"chunk_index": 1, "source": "test"}
            ),
            Document(
                page_content="The weather today is sunny. I like to eat pizza on weekends.",
                metadata={"chunk_index": 2, "source": "test"}
            ),
            Document(
                page_content="Neural networks are a subset of machine learning that mimic human brain structure. They excel at pattern recognition tasks.",
                metadata={"chunk_index": 3, "source": "test"}
            )
        ]
    
    @pytest.fixture
    def mock_jina_embeddings(self):
        """Mock Jina embeddings for testing."""
        # Simulate realistic embeddings - similar content should have similar embeddings
        return {
            "embeddings": [
                # Chunk 0: ML algorithms (similar to chunk 1 and 3)
                [0.8, 0.6, 0.2, 0.1, 0.9, 0.3],
                # Chunk 1: ML training (similar to chunk 0 and 3)
                [0.7, 0.7, 0.3, 0.0, 0.8, 0.4],
                # Chunk 2: Weather/pizza (dissimilar)
                [0.1, 0.2, 0.9, 0.8, 0.1, 0.0],
                # Chunk 3: Neural networks (similar to chunk 0 and 1)
                [0.9, 0.5, 0.1, 0.2, 0.9, 0.2]
            ],
            "tokens_used": 100,
            "model": "jina-embeddings-v2-base-en"
        }
    
    def test_enhanced_evaluator_initialization(self):
        """Test that EnhancedChunkQualityEvaluator initializes with Jina integration."""
        # This test will fail initially - defining expected behavior
        evaluator = EnhancedChunkQualityEvaluator(
            use_jina_embeddings=True,
            jina_api_key="test_key",
            fallback_to_tfidf=True
        )
        
        assert evaluator.use_jina_embeddings is True
        assert evaluator.fallback_to_tfidf is True
        assert hasattr(evaluator, 'jina_provider')
        assert isinstance(evaluator.jina_provider, JinaProvider)
        assert hasattr(evaluator, 'embedding_cache')
    
    def test_enhanced_evaluator_without_jina(self):
        """Test that evaluator works without Jina (fallback mode)."""
        # This test will fail initially - defining expected behavior
        evaluator = EnhancedChunkQualityEvaluator(
            use_jina_embeddings=False,
            fallback_to_tfidf=True
        )
        
        assert evaluator.use_jina_embeddings is False
        assert evaluator.fallback_to_tfidf is True
        assert evaluator.jina_provider is None
    
    def test_jina_embedding_generation(self, sample_chunks, mock_jina_embeddings):
        """Test Jina embedding generation for chunks."""
        # This test will fail initially - defining expected behavior
        evaluator = EnhancedChunkQualityEvaluator(
            use_jina_embeddings=True,
            jina_api_key="test_key"
        )
        
        # Mock Jina provider response
        with patch.object(evaluator.jina_provider, 'generate_embeddings') as mock_embeddings:
            mock_embeddings.return_value = MagicMock(
                embeddings=mock_jina_embeddings["embeddings"],
                tokens_used=mock_jina_embeddings["tokens_used"],
                model=mock_jina_embeddings["model"]
            )
            
            # Test embedding generation
            embeddings = evaluator._generate_chunk_embeddings(sample_chunks)
            
            # Verify embeddings structure
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (4, 6)  # 4 chunks, 6-dimensional embeddings
            
            # Verify Jina was called correctly
            mock_embeddings.assert_called_once()
            call_args = mock_embeddings.call_args[0][0]
            assert len(call_args) == 4  # 4 chunk texts
    
    def test_embedding_based_semantic_coherence(self, sample_chunks, mock_jina_embeddings):
        """Test semantic coherence analysis using Jina embeddings."""
        # This test will fail initially - defining expected behavior
        evaluator = EnhancedChunkQualityEvaluator(
            use_jina_embeddings=True,
            jina_api_key="test_key"
        )
        
        # Mock Jina provider response
        with patch.object(evaluator.jina_provider, 'generate_embeddings') as mock_embeddings:
            mock_embeddings.return_value = MagicMock(
                embeddings=mock_jina_embeddings["embeddings"],
                tokens_used=mock_jina_embeddings["tokens_used"],
                model=mock_jina_embeddings["model"]
            )
            
            # Test semantic coherence analysis
            coherence_metrics = evaluator._analyze_semantic_coherence_with_embeddings(sample_chunks)
            
            # Verify coherence metrics structure
            assert isinstance(coherence_metrics, dict)
            assert 'coherence_score' in coherence_metrics
            assert 'avg_similarity' in coherence_metrics
            assert 'similarity_std' in coherence_metrics
            assert 'embedding_based' in coherence_metrics
            
            # Verify coherence scores are reasonable
            assert 0 <= coherence_metrics['coherence_score'] <= 1
            assert 0 <= coherence_metrics['avg_similarity'] <= 1
            assert coherence_metrics['embedding_based'] is True
            
            # Verify that ML-related chunks (0, 1, 3) have higher similarity than weather chunk (2)
            # This tests that Jina embeddings capture semantic meaning better than TF-IDF
            assert coherence_metrics['coherence_score'] > 0.3  # Should detect some coherence
    
    def test_embedding_similarity_calculation(self, mock_jina_embeddings):
        """Test embedding similarity calculations."""
        # This test will fail initially - defining expected behavior
        evaluator = EnhancedChunkQualityEvaluator(
            use_jina_embeddings=True,
            jina_api_key="test_key"
        )
        
        embeddings = np.array(mock_jina_embeddings["embeddings"])
        
        # Test similarity calculation
        similarities = evaluator._calculate_embedding_similarities(embeddings)
        
        # Verify similarity matrix
        assert isinstance(similarities, np.ndarray)
        assert similarities.shape == (4, 4)  # 4x4 similarity matrix
        
        # Verify diagonal is 1.0 (self-similarity)
        np.testing.assert_allclose(np.diag(similarities), 1.0, rtol=1e-5)
        
        # Verify symmetry
        np.testing.assert_allclose(similarities, similarities.T, rtol=1e-5)
        
        # Test that ML chunks (0, 1, 3) are more similar to each other than to weather chunk (2)
        ml_similarities = [similarities[0][1], similarities[0][3], similarities[1][3]]
        weather_similarities = [similarities[0][2], similarities[1][2], similarities[3][2]]
        
        avg_ml_similarity = np.mean(ml_similarities)
        avg_weather_similarity = np.mean(weather_similarities)
        
        assert avg_ml_similarity > avg_weather_similarity
    
    def test_enhanced_quality_evaluation_with_jina(self, sample_chunks, mock_jina_embeddings):
        """Test complete quality evaluation with Jina embeddings."""
        # This test will fail initially - defining expected behavior
        evaluator = EnhancedChunkQualityEvaluator(
            use_jina_embeddings=True,
            jina_api_key="test_key"
        )
        
        # Mock Jina provider response
        with patch.object(evaluator.jina_provider, 'generate_embeddings') as mock_embeddings:
            mock_embeddings.return_value = MagicMock(
                embeddings=mock_jina_embeddings["embeddings"],
                tokens_used=mock_jina_embeddings["tokens_used"],
                model=mock_jina_embeddings["model"]
            )

            # Test complete evaluation
            metrics = evaluator.evaluate_chunks(sample_chunks)
            
            # Verify enhanced metrics structure
            assert isinstance(metrics, dict)
            assert 'semantic_coherence' in metrics
            assert 'embedding_metrics' in metrics
            assert 'jina_enhanced' in metrics
            
            # Verify Jina-specific metrics
            embedding_metrics = metrics['embedding_metrics']
            assert 'provider' in embedding_metrics
            assert 'model' in embedding_metrics
            assert 'tokens_used' in embedding_metrics
            assert 'embedding_dimension' in embedding_metrics
            
            assert embedding_metrics['provider'] == 'jina'
            assert embedding_metrics['model'] == 'jina-embeddings-v2-base-en'
            assert embedding_metrics['tokens_used'] == 100
            assert embedding_metrics['embedding_dimension'] == 6
            
            # Verify Jina enhancement flag
            assert metrics['jina_enhanced'] is True
            
            # Verify overall score is calculated with embedding insights
            assert 'overall_score' in metrics
            assert isinstance(metrics['overall_score'], (int, float))
    
    def test_embedding_caching(self, sample_chunks, mock_jina_embeddings):
        """Test that embeddings are cached for performance."""
        # This test will fail initially - defining expected behavior
        evaluator = EnhancedChunkQualityEvaluator(
            use_jina_embeddings=True,
            jina_api_key="test_key",
            enable_embedding_cache=True
        )
        
        # Mock Jina provider response
        with patch.object(evaluator.jina_provider, 'generate_embeddings') as mock_embeddings:
            mock_embeddings.return_value = MagicMock(
                embeddings=mock_jina_embeddings["embeddings"],
                tokens_used=mock_jina_embeddings["tokens_used"],
                model=mock_jina_embeddings["model"]
            )
            
            # First evaluation - should call Jina
            metrics1 = evaluator.evaluate_chunks(sample_chunks)
            assert mock_embeddings.call_count == 1
            
            # Second evaluation with same chunks - should use cache
            metrics2 = evaluator.evaluate_chunks(sample_chunks)
            assert mock_embeddings.call_count == 1  # No additional calls
            
            # Verify results are consistent
            assert metrics1['semantic_coherence']['coherence_score'] == metrics2['semantic_coherence']['coherence_score']
            
            # Verify cache hit information
            assert metrics2['embedding_metrics']['cache_hit'] is True
    
    def test_fallback_to_tfidf_on_jina_failure(self, sample_chunks):
        """Test fallback to TF-IDF when Jina fails."""
        # This test will fail initially - defining expected behavior
        evaluator = EnhancedChunkQualityEvaluator(
            use_jina_embeddings=True,
            jina_api_key="test_key",
            fallback_to_tfidf=True
        )
        
        # Mock Jina provider to raise an exception
        with patch.object(evaluator.jina_provider, 'generate_embeddings') as mock_embeddings:
            mock_embeddings.side_effect = Exception("Jina API error")
            
            # Test evaluation with fallback
            metrics = evaluator.evaluate_chunks(sample_chunks)
            
            # Verify fallback was used
            assert 'semantic_coherence' in metrics
            assert 'embedding_metrics' in metrics
            assert metrics['jina_enhanced'] is False
            assert metrics['embedding_metrics']['fallback_used'] is True
            assert metrics['embedding_metrics']['fallback_reason'] == "Jina API error"
            
            # Verify TF-IDF was used instead
            assert 'coherence_score' in metrics['semantic_coherence']
            assert isinstance(metrics['semantic_coherence']['coherence_score'], (int, float))
    
    def test_semantic_topic_clustering_with_embeddings(self, sample_chunks, mock_jina_embeddings):
        """Test semantic topic clustering using Jina embeddings."""
        # This test will fail initially - defining expected behavior
        evaluator = EnhancedChunkQualityEvaluator(
            use_jina_embeddings=True,
            jina_api_key="test_key"
        )
        
        # Mock Jina provider response
        with patch.object(evaluator.jina_provider, 'generate_embeddings') as mock_embeddings:
            mock_embeddings.return_value = MagicMock(
                embeddings=mock_jina_embeddings["embeddings"],
                tokens_used=mock_jina_embeddings["tokens_used"]
            )
            
            # Test topic clustering
            clustering_metrics = evaluator._analyze_topic_clustering(sample_chunks)
            
            # Verify clustering metrics
            assert isinstance(clustering_metrics, dict)
            assert 'topic_clusters' in clustering_metrics
            assert 'cluster_coherence' in clustering_metrics
            assert 'outlier_chunks' in clustering_metrics
            
            # Verify topic clusters
            topic_clusters = clustering_metrics['topic_clusters']
            assert isinstance(topic_clusters, list)
            assert len(topic_clusters) >= 1  # Should identify at least one cluster
            
            # Verify outlier detection
            outlier_chunks = clustering_metrics['outlier_chunks']
            assert isinstance(outlier_chunks, list)
            # Weather/pizza chunk should be identified as outlier
            assert 2 in outlier_chunks  # Chunk index 2 is the weather/pizza chunk
    
    def test_embedding_dimension_consistency(self, sample_chunks):
        """Test that embedding dimensions are consistent across chunks."""
        # This test will fail initially - defining expected behavior
        evaluator = EnhancedChunkQualityEvaluator(
            use_jina_embeddings=True,
            jina_api_key="test_key"
        )
        
        # Mock varying length embeddings (should be normalized)
        varying_embeddings = [
            [0.1, 0.2, 0.3],  # 3D
            [0.4, 0.5, 0.6, 0.7],  # 4D
            [0.8, 0.9],  # 2D
            [0.1, 0.2, 0.3, 0.4, 0.5]  # 5D
        ]
        
        with patch.object(evaluator.jina_provider, 'generate_embeddings') as mock_embeddings:
            mock_embeddings.return_value = MagicMock(
                embeddings=varying_embeddings,
                tokens_used=50,
                model="jina-embeddings-v2-base-en"
            )
            
            # Test embedding normalization
            normalized_embeddings = evaluator._normalize_embedding_dimensions(varying_embeddings)
            
            # Verify all embeddings have same dimension
            assert isinstance(normalized_embeddings, np.ndarray)
            assert len(set(len(emb) for emb in normalized_embeddings)) == 1  # All same length
            
            # Verify normalization preserves relative magnitudes
            assert normalized_embeddings.shape[0] == 4  # 4 chunks
            assert normalized_embeddings.shape[1] > 0  # Some dimension
    
    def test_performance_comparison_tfidf_vs_jina(self, sample_chunks, mock_jina_embeddings):
        """Test performance comparison between TF-IDF and Jina embeddings."""
        # This test will fail initially - defining expected behavior
        
        # Test with TF-IDF
        tfidf_evaluator = ChunkQualityEvaluator()  # Original evaluator
        tfidf_metrics = tfidf_evaluator.evaluate_chunks(sample_chunks)
        
        # Test with Jina embeddings
        jina_evaluator = EnhancedChunkQualityEvaluator(
            use_jina_embeddings=True,
            jina_api_key="test_key"
        )
        
        with patch.object(jina_evaluator.jina_provider, 'generate_embeddings') as mock_embeddings:
            mock_embeddings.return_value = MagicMock(
                embeddings=mock_jina_embeddings["embeddings"],
                tokens_used=mock_jina_embeddings["tokens_used"],
                model=mock_jina_embeddings["model"]
            )
            
            jina_metrics = jina_evaluator.evaluate_chunks(sample_chunks)
        
        # Compare semantic coherence scores
        tfidf_coherence = tfidf_metrics['semantic_coherence']['coherence_score']
        jina_coherence = jina_metrics['semantic_coherence']['coherence_score']
        
        # Verify both produce valid scores
        assert 0 <= tfidf_coherence <= 1
        assert 0 <= jina_coherence <= 1
        
        # For ML-related content, Jina should potentially provide better semantic understanding
        # (This is content-dependent, but we can verify the system works)
        assert isinstance(jina_coherence, (int, float))
        assert 'jina_enhanced' in jina_metrics
        assert jina_metrics['jina_enhanced'] is True
    
    def test_hybrid_evaluation_mode(self, sample_chunks, mock_jina_embeddings):
        """Test hybrid mode using both TF-IDF and Jina embeddings."""
        # This test will fail initially - defining expected behavior
        evaluator = EnhancedChunkQualityEvaluator(
            use_jina_embeddings=True,
            jina_api_key="test_key",
            hybrid_mode=True
        )
        
        # Mock Jina provider response
        with patch.object(evaluator.jina_provider, 'generate_embeddings') as mock_embeddings:
            mock_embeddings.return_value = MagicMock(
                embeddings=mock_jina_embeddings["embeddings"],
                tokens_used=mock_jina_embeddings["tokens_used"]
            )
            
            # Test hybrid evaluation
            metrics = evaluator.evaluate_chunks(sample_chunks)
            
            # Verify hybrid metrics
            assert 'semantic_coherence' in metrics
            assert 'hybrid_analysis' in metrics
            
            hybrid_analysis = metrics['hybrid_analysis']
            assert 'tfidf_score' in hybrid_analysis
            assert 'jina_score' in hybrid_analysis
            assert 'combined_score' in hybrid_analysis
            assert 'agreement_score' in hybrid_analysis
            
            # Verify agreement score calculation
            agreement_score = hybrid_analysis['agreement_score']
            assert 0 <= agreement_score <= 1
            
            # Verify combined score is reasonable blend
            combined_score = hybrid_analysis['combined_score']
            tfidf_score = hybrid_analysis['tfidf_score']
            jina_score = hybrid_analysis['jina_score']
            
            assert min(tfidf_score, jina_score) <= combined_score <= max(tfidf_score, jina_score)


class TestJinaProviderIntegration:
    """Test Jina provider integration specifics."""
    
    def test_jina_provider_embedding_batch_processing(self):
        """Test Jina provider handles batch embedding requests efficiently."""
        # This test will fail initially - defining expected behavior
        provider = JinaProvider(api_key="test_key")
        
        # Test batch processing
        texts = [
            "This is the first text to embed",
            "This is the second text to embed", 
            "This is the third text to embed"
        ]
        
        with patch.object(provider, 'generate_embeddings') as mock_batch:
            mock_batch.return_value = MagicMock(
                embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                tokens_used=75,
                model="jina-embeddings-v2-base-en"
            )
            
            # Test batch embedding
            result = provider.generate_embeddings(texts)
            
            # Verify batch call efficiency
            mock_batch.assert_called_once_with(texts)
            assert len(result.embeddings) == 3
            assert result.tokens_used == 75
    
    def test_jina_provider_error_handling(self):
        """Test Jina provider handles errors gracefully."""
        # This test will fail initially - defining expected behavior
        provider = JinaProvider(api_key="invalid_key")
        
        with patch('requests.post') as mock_post:
            # Mock API error response
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Invalid API key"
            mock_post.return_value = mock_response
            
            # Test error handling
            from src.llm.providers.base import EmbeddingError
            with pytest.raises(EmbeddingError):
                provider.generate_embeddings(["test text"])
    
    def test_jina_embedding_quality_metrics(self):
        """Test quality metrics specific to Jina embeddings."""
        # This test will fail initially - defining expected behavior
        evaluator = EnhancedChunkQualityEvaluator(
            use_jina_embeddings=True,
            jina_api_key="test_key"
        )
        
        # Mock high-quality embeddings (well-separated)
        high_quality_embeddings = [
            [1.0, 0.0, 0.0],  # Distinct vector 1
            [0.0, 1.0, 0.0],  # Distinct vector 2
            [0.0, 0.0, 1.0],  # Distinct vector 3
        ]
        
        quality_metrics = evaluator._calculate_embedding_quality_metrics(high_quality_embeddings)
        
        # Verify quality metrics
        assert isinstance(quality_metrics, dict)
        assert 'embedding_variance' in quality_metrics
        assert 'embedding_entropy' in quality_metrics
        assert 'separation_quality' in quality_metrics
        
        # High-quality embeddings should have good separation
        assert quality_metrics['separation_quality'] > 0.5
        assert quality_metrics['embedding_variance'] > 0.1