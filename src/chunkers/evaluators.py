import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import re
import hashlib
import time
from src.utils.logger import get_logger

class ChunkQualityEvaluator:
    """Evaluate the quality of document chunks"""
    
    def __init__(self): # Removed document_type='general' from init for now, as it's not used elsewhere.
        # Initialize logger
        self.logger = get_logger(__name__)
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Adaptive thresholds (removed document_type from init, so these will be fixed values for now)
        # These can be made dynamic if a document_type parameter is reintroduced and used.
        self.min_words_for_very_short = 10 # Default from settings in previous turns
        self.coherence_score_boost_factor = 1.0 # Default, no boost
        self.structure_score_weight_factor = 1.0 # Default, no change

    def evaluate_chunks(self, chunks: List[Document]) -> Dict[str, Any]:
        """Comprehensive chunk quality evaluation"""
        
        if not chunks:
            return {'error': 'No chunks to evaluate'}
        
        metrics = {
            'total_chunks': len(chunks),
            'size_distribution': self._analyze_size_distribution(chunks),
            'content_quality': self._analyze_content_quality(chunks),
            'semantic_coherence': self._analyze_semantic_coherence(chunks),
            'overlap_analysis': self._analyze_overlap(chunks), 
            'structural_preservation': self._analyze_structure_preservation(chunks)
        }
        
        # Overall quality score (0-100)
        metrics['overall_score'] = self._calculate_overall_score(metrics)
        
        return metrics
    
    def _analyze_size_distribution(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze chunk size distribution"""
        sizes = [len(chunk.page_content) for chunk in chunks]
        word_counts = [len(chunk.page_content.split()) for chunk in chunks]
        
        if not sizes:
            return {
                'char_stats': {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0},
                'word_stats': {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0},
                'size_consistency': 0
            }

        mean_size = np.mean(sizes)
        std_dev_size = np.std(sizes)
        
        size_consistency = 1 - (std_dev_size / mean_size) if mean_size > 0 else 0
        size_consistency = max(0, size_consistency) # Ensure it's not negative

        return {
            'char_stats': {
                'mean': mean_size,
                'median': np.median(sizes),
                'std': std_dev_size,
                'min': np.min(sizes),
                'max': np.max(sizes)
            },
            'word_stats': {
                'mean': np.mean(word_counts),
                'median': np.median(word_counts),
                'std': np.std(word_counts),
                'min': np.min(word_counts),
                'max': np.max(word_counts)
            },
            'size_consistency': size_consistency
        }
    
    def _analyze_content_quality(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze content quality metrics"""
        
        quality_metrics = {
            'empty_chunks': 0,
            'very_short_chunks': 0,
            'incomplete_sentences': 0,
        }
        
        for chunk in chunks:
            content = chunk.page_content.strip()
            
            if not content:
                quality_metrics['empty_chunks'] += 1
                continue
            
            if len(content.split()) < self.min_words_for_very_short: # Use class attribute
                quality_metrics['very_short_chunks'] += 1
            
            is_header = re.match(r'^#+\s', content) or any(h in chunk.metadata for h in ["Header 1", "Header 2", "Header 3", "Header 4"])
            is_code = '```' in content or chunk.metadata.get('content_type') == 'code' or re.search(r'\b(def|class|import|function|const|var)\b', content)
            is_list = re.search(r'^\s*[-*+]\s+.*|\s*\d+\.\s+.*', content, re.MULTILINE) # Re-added list detection
            
            if not (is_header or is_code or is_list) and not re.search(r'[.!?:]$', content): # Check for ending punctuation
                quality_metrics['incomplete_sentences'] += 1
            
        total_chunks = len(chunks)
        quality_percentages = {
            f"{key}_pct": (value / total_chunks * 100) if total_chunks > 0 else 0
            for key, value in quality_metrics.items()
        }
        
        return {**quality_metrics, **quality_percentages}
    
    def _analyze_semantic_coherence(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze semantic coherence between chunks"""
        
        if len(chunks) < 2:
            return {'coherence_score': 1.0, 'avg_similarity': 0.0, 'similarity_std': 0.0}
        
        try:
            texts = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
            
            if len(texts) < 2:
                 return {'coherence_score': 1.0, 'avg_similarity': 0.0, 'similarity_std': 0.0}

            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            similarities = cosine_similarity(tfidf_matrix)
            
            adjacent_similarities = []
            for i in range(len(similarities) - 1):
                adjacent_similarities.append(similarities[i][i + 1])
            
            avg_adjacent_similarity = np.mean(adjacent_similarities) if adjacent_similarities else 0
            
            # Apply boost factor (self.coherence_score_boost_factor)
            coherence_score = min(1.0, avg_adjacent_similarity * self.coherence_score_boost_factor) 

            overall_avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            
            return {
                'coherence_score': coherence_score,
                'avg_similarity': overall_avg_similarity,
                'similarity_std': np.std(adjacent_similarities) if adjacent_similarities else 0
            }
            
        except Exception as e:
            self.logger.error(
                "Error in semantic analysis during chunk evaluation",
                error=str(e),
                error_type=type(e).__name__,
                num_chunks=len(chunks)
            )
            return {'coherence_score': 0.0, 'avg_similarity': 0.0, 'error': str(e), 'similarity_std': 0.0}
    
    def _analyze_overlap(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze content overlap between chunks"""
        
        if len(chunks) < 2:
            return {'avg_overlap': 0.0, 'overlap_std': 0.0, 'overlap_distribution': []}
        
        overlap_scores = []
        
        for i in range(len(chunks) - 1):
            current_words = set(chunks[i].page_content.lower().split())
            next_words = set(chunks[i + 1].page_content.lower().split())
            
            if len(current_words) == 0 or len(next_words) == 0:
                continue
            
            intersection = len(current_words.intersection(next_words))
            union = len(current_words.union(next_words))
            
            if union > 0:
                jaccard_similarity = intersection / union
                overlap_scores.append(jaccard_similarity)
        
        return {
            'avg_overlap': np.mean(overlap_scores) if overlap_scores else 0,
            'overlap_std': np.std(overlap_scores) if overlap_scores else 0,
            'overlap_distribution': overlap_scores[:10]
        }
    
    def _analyze_structure_preservation(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze how well document structure is preserved"""
        
        structure_metrics = {
            'chunks_with_headers': 0,
            'chunks_with_code': 0,
            'chunks_with_lists': 0,
            'chunks_with_links': 0
        }
        
        for chunk in chunks:
            content = chunk.page_content
            
            if any(h in chunk.metadata for h in ["Header 1", "Header 2", "Header 3", "Header 4"]):
                structure_metrics['chunks_with_headers'] += 1
            elif re.search(r'^#+\s', content, re.MULTILINE):
                structure_metrics['chunks_with_headers'] += 1
            
            if chunk.metadata.get('has_code', False):
                structure_metrics['chunks_with_code'] += 1
            
            # This regex is robust for both bullet and numbered lists starting a line.
            if re.search(r'^\s*[-*+]\s+.*|\s*\d+\.\s+.*', content, re.MULTILINE):
                structure_metrics['chunks_with_lists'] += 1
            
            if '[' in content and '](' in content:
                structure_metrics['chunks_with_links'] += 1
        
        total_chunks = len(chunks)
        structure_percentages = {
            f"{key}_pct": (value / total_chunks * 100) if total_chunks > 0 else 0
            for key, value in structure_metrics.items()
        }
        
        return {**structure_metrics, **structure_percentages}
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)"""
        
        try:
            size_score = metrics['size_distribution']['size_consistency'] * 20 # Adjusted weight
            
            content_metrics = metrics['content_quality']
            content_score_raw = (
                (100 - content_metrics['empty_chunks_pct']) * 0.5 + 
                (100 - content_metrics['very_short_chunks_pct']) * 0.3 +
                (100 - content_metrics['incomplete_sentences_pct']) * 0.2
            )
            content_score = (content_score_raw / 100) * 30 # Adjusted weight
            
            coherence_score = metrics['semantic_coherence']['coherence_score'] * 25
            
            structure_metrics = metrics['structural_preservation']
            structure_score_raw = (
                structure_metrics['chunks_with_headers_pct'] * 0.4 +
                structure_metrics['chunks_with_code_pct'] * 0.3 +
                structure_metrics['chunks_with_lists_pct'] * 0.2 +
                structure_metrics['chunks_with_links_pct'] * 0.1 # New bonus for links
            )
            structure_score = (structure_score_raw / 100) * 25 # Adjusted weight
            
            overall_score = size_score + content_score + coherence_score + structure_score
            return min(100, max(0, overall_score))
            
        except Exception as e:
            self.logger.error(
                "Error calculating overall quality score",
                error=str(e),
                error_type=type(e).__name__
            )
            return 0.0
    
    def generate_report(self, chunks: List[Document], output_path: str = None) -> str:
        """Generate a detailed evaluation report"""
        
        metrics = self.evaluate_chunks(chunks)
        
        report = f"""
# Chunk Quality Evaluation Report

## Summary
- **Total Chunks**: {metrics['total_chunks']}
- **Overall Quality Score**: {metrics['overall_score']:.1f}/100

## Size Distribution
- **Average Characters**: {metrics['size_distribution']['char_stats']['mean']:.0f}
- **Average Words**: {metrics['size_distribution']['word_stats']['mean']:.0f}
- **Size Consistency**: {metrics['size_distribution']['size_consistency']:.2f}

## Content Quality
- **Empty Chunks**: {metrics['content_quality']['empty_chunks']} ({metrics['content_quality']['empty_chunks_pct']:.1f}%)
- **Very Short Chunks**: {metrics['content_quality']['very_short_chunks']} ({metrics['content_quality']['very_short_chunks_pct']:.1f}%)
- **Incomplete Sentences**: {metrics['content_quality']['incomplete_sentences']} ({metrics['content_quality']['incomplete_sentences_pct']:.1f}%)

## Semantic Coherence
- **Coherence Score**: {metrics['semantic_coherence']['coherence_score']:.3f}
- **Average Similarity**: {metrics['semantic_coherence']['avg_similarity']:.3f}

## Structure Preservation
- **Chunks with Headers**: {metrics['structural_preservation']['chunks_with_headers']} ({metrics['structural_preservation']['chunks_with_headers_pct']:.1f}%)
- **Chunks with Code**: {metrics['structural_preservation']['chunks_with_code']} ({metrics['structural_preservation']['chunks_with_code_pct']:.1f}%)
- **Chunks with Lists**: {metrics['structural_preservation']['chunks_with_lists']} ({metrics['structural_preservation']['chunks_with_lists_pct']:.1f}%)

## Recommendations
"""
        
        # Add recommendations based on metrics
        if metrics['content_quality']['empty_chunks_pct'] > 5:
            report += "- ⚠️  High number of empty chunks detected. Consider improving preprocessing.\n"
        
        if metrics['content_quality']['very_short_chunks_pct'] > 20:
            report += "- ⚠️  Many very short chunks. Consider increasing minimum chunk size.\n"
        
        if metrics['semantic_coherence']['coherence_score'] < 0.3:
            report += "- ⚠️  Low semantic coherence. Consider adjusting chunk overlap or using semantic chunking.\n"
        
        if metrics['size_distribution']['size_consistency'] < 0.5:
            report += "- ⚠️  Inconsistent chunk sizes. Consider using fixed-size chunking for more consistency.\n"
        
        if metrics['overall_score'] >= 80:
            report += "- ✅ Excellent chunking quality!\n"
        elif metrics['overall_score'] >= 60:
            report += "- ✅ Good chunking quality with room for improvement.\n"
        else:
            report += "- ❌ Chunking quality needs significant improvement.\n"
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(
                "Evaluation report saved successfully",
                output_path=output_path,
                report_size_chars=len(report)
            )
        
        return report


class AdvancedQualityEvaluator(ChunkQualityEvaluator):
    """Enhanced evaluator with strategy-specific metrics."""
    
    def __init__(self):
        super().__init__()
    
    def evaluate_strategy_effectiveness(self, chunks: List[Document], 
                                      strategy_used: str) -> Dict[str, Any]:
        """Evaluate how well the chunking strategy performed."""
        
        base_metrics = self.evaluate_chunks(chunks)
        
        strategy_metrics = {
            'strategy_used': strategy_used,
            'boundary_preservation': self._analyze_boundary_preservation(chunks),
            'context_continuity': self._analyze_context_continuity(chunks),
            'information_density': self._analyze_information_density(chunks),
            'readability_scores': self._calculate_readability_scores(chunks),
            'topic_coherence': self._analyze_topic_coherence(chunks),
            'chunk_independence': self._analyze_chunk_independence(chunks)
        }
        
        return {**base_metrics, **strategy_metrics}
    
    def _analyze_boundary_preservation(self, chunks: List[Document]) -> Dict[str, float]:
        """Analyze how well semantic boundaries are preserved."""
        sentence_score = self._score_sentence_boundaries(chunks)
        paragraph_score = self._score_paragraph_boundaries(chunks)
        section_score = self._score_section_boundaries(chunks)
        
        return {
            'sentence_boundary_score': sentence_score,
            'paragraph_boundary_score': paragraph_score,
            'section_boundary_score': section_score
        }
    
    def _analyze_context_continuity(self, chunks: List[Document]) -> Dict[str, float]:
        """Analyze context flow between chunks."""
        topic_smoothness = self._score_topic_transitions(chunks)
        reference_completeness = self._score_reference_completeness(chunks)
        narrative_flow = self._score_narrative_flow(chunks)
        
        return {
            'topic_transition_smoothness': topic_smoothness,
            'reference_completeness': reference_completeness,
            'narrative_flow': narrative_flow
        }
    
    def _analyze_information_density(self, chunks: List[Document]) -> float:
        """Analyze information density of chunks."""
        if not chunks:
            return 0.0
        
        # Simple heuristic: ratio of unique words to total words
        total_words = 0
        unique_words = set()
        
        for chunk in chunks:
            words = chunk.page_content.lower().split()
            total_words += len(words)
            unique_words.update(words)
        
        return len(unique_words) / total_words if total_words > 0 else 0.0
    
    def _calculate_readability_scores(self, chunks: List[Document]) -> Dict[str, float]:
        """Calculate readability scores for chunks."""
        if not chunks:
            return {'avg_sentence_length': 0.0, 'avg_word_length': 0.0}
        
        total_sentences = 0
        total_words = 0
        total_chars = 0
        
        for chunk in chunks:
            content = chunk.page_content
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            words = content.split()
            
            total_sentences += len(sentences)
            total_words += len(words)
            total_chars += sum(len(word) for word in words)
        
        avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
        avg_word_length = total_chars / total_words if total_words > 0 else 0
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length
        }
    
    def _analyze_topic_coherence(self, chunks: List[Document]) -> float:
        """Analyze topic coherence within chunks."""
        if not chunks:
            return 0.0
        
        # Simple heuristic: average cosine similarity between adjacent chunks
        if len(chunks) < 2:
            return 1.0
        
        try:
            chunk_texts = [chunk.page_content for chunk in chunks]
            vectors = self.vectorizer.fit_transform(chunk_texts)
            
            similarities = []
            for i in range(len(chunks) - 1):
                sim = cosine_similarity(vectors[i:i+1], vectors[i+1:i+2])[0][0]
                similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.0
        except Exception:
            return 0.0
    
    def _analyze_chunk_independence(self, chunks: List[Document]) -> float:
        """Analyze how independently each chunk can be understood."""
        if not chunks:
            return 0.0
        
        # Simple heuristic: chunks that start with complete sentences
        independent_chunks = 0
        
        for chunk in chunks:
            content = chunk.page_content.strip()
            if content and content[0].isupper():
                independent_chunks += 1
        
        return independent_chunks / len(chunks) if chunks else 0.0
    
    def _score_sentence_boundaries(self, chunks: List[Document]) -> float:
        """Score how well sentence boundaries are preserved."""
        if not chunks:
            return 0.0
        
        # Count chunks that end with proper sentence endings
        proper_endings = 0
        
        for chunk in chunks:
            content = chunk.page_content.strip()
            if content and content[-1] in '.!?':
                proper_endings += 1
        
        return proper_endings / len(chunks) if chunks else 0.0
    
    def _score_paragraph_boundaries(self, chunks: List[Document]) -> float:
        """Score how well paragraph boundaries are preserved."""
        if not chunks:
            return 0.0
        
        # Count chunks that start with paragraph-like patterns
        paragraph_starts = 0
        
        for chunk in chunks:
            content = chunk.page_content.strip()
            if content and (content.startswith('#') or content[0].isupper()):
                paragraph_starts += 1
        
        return paragraph_starts / len(chunks) if chunks else 0.0
    
    def _score_section_boundaries(self, chunks: List[Document]) -> float:
        """Score how well section boundaries are preserved."""
        if not chunks:
            return 0.0
        
        # Count chunks that contain headers
        section_chunks = 0
        
        for chunk in chunks:
            content = chunk.page_content.strip()
            if re.search(r'^#+\s+', content, re.MULTILINE):
                section_chunks += 1
        
        return section_chunks / len(chunks) if chunks else 0.0
    
    def _score_topic_transitions(self, chunks: List[Document]) -> float:
        """Score how smoothly topics transition between chunks."""
        if len(chunks) < 2:
            return 1.0
        
        # Simple heuristic: semantic similarity between consecutive chunks
        try:
            chunk_texts = [chunk.page_content for chunk in chunks]
            vectors = self.vectorizer.fit_transform(chunk_texts)
            
            transitions = []
            for i in range(len(chunks) - 1):
                sim = cosine_similarity(vectors[i:i+1], vectors[i+1:i+2])[0][0]
                transitions.append(sim)
            
            return np.mean(transitions) if transitions else 0.0
        except Exception:
            return 0.0
    
    def _score_reference_completeness(self, chunks: List[Document]) -> float:
        """Score how complete references are within chunks."""
        if not chunks:
            return 0.0
        
        # Simple heuristic: chunks that don't start with pronouns or incomplete references
        complete_chunks = 0
        incomplete_starters = ['this', 'that', 'these', 'those', 'it', 'they', 'he', 'she']
        
        for chunk in chunks:
            content = chunk.page_content.strip().lower()
            if content:
                first_word = content.split()[0] if content.split() else ''
                if first_word not in incomplete_starters:
                    complete_chunks += 1
        
        return complete_chunks / len(chunks) if chunks else 0.0
    
    def _score_narrative_flow(self, chunks: List[Document]) -> float:
        """Score how well narrative flow is maintained."""
        if not chunks:
            return 0.0
        
        # Simple heuristic: chunks that maintain logical flow
        flow_score = 0
        
        for i, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            
            # Check for transition words/phrases
            transition_words = ['however', 'therefore', 'moreover', 'furthermore', 'additionally', 'consequently']
            has_transition = any(word in content.lower() for word in transition_words)
            
            # Check for proper sentence structure
            has_complete_sentences = '.' in content or '!' in content or '?' in content
            
            if has_complete_sentences:
                flow_score += 0.5
            if has_transition and i > 0:
                flow_score += 0.5
        
        return flow_score / len(chunks) if chunks else 0.0


class EnhancedChunkQualityEvaluator(ChunkQualityEvaluator):
    """Enhanced evaluator with Jina AI embedding integration for superior semantic analysis."""
    
    def __init__(self, 
                 use_jina_embeddings: bool = True,
                 jina_api_key: Optional[str] = None,
                 jina_model: str = "jina-embeddings-v2-base-en",
                 fallback_to_tfidf: bool = True,
                 enable_embedding_cache: bool = True,
                 hybrid_mode: bool = False,
                 **kwargs):
        """
        Initialize Enhanced Chunk Quality Evaluator with Jina integration.
        
        Args:
            use_jina_embeddings: Whether to use Jina AI embeddings
            jina_api_key: Jina AI API key
            jina_model: Jina embedding model to use
            fallback_to_tfidf: Whether to fallback to TF-IDF on Jina failure
            enable_embedding_cache: Whether to cache embeddings for performance
            hybrid_mode: Whether to use both TF-IDF and Jina for comparison
        """
        super().__init__(**kwargs)
        
        self.use_jina_embeddings = use_jina_embeddings
        self.fallback_to_tfidf = fallback_to_tfidf
        self.hybrid_mode = hybrid_mode
        self.enable_embedding_cache = enable_embedding_cache
        
        # Initialize Jina provider if enabled
        self.jina_provider = None
        if use_jina_embeddings and jina_api_key:
            try:
                from src.llm.providers.jina_provider import JinaProvider
                self.jina_provider = JinaProvider(
                    api_key=jina_api_key,
                    model=jina_model,
                    embedding_model=jina_model
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize Jina provider: {e}")
                if not fallback_to_tfidf:
                    raise
                self.use_jina_embeddings = False
        
        # Initialize embedding cache
        self.embedding_cache = {} if enable_embedding_cache else None
        
    def evaluate_chunks(self, chunks: List[Document]) -> Dict[str, Any]:
        """Enhanced chunk quality evaluation with Jina embeddings."""
        
        if not chunks:
            return {'error': 'No chunks to evaluate'}
        
        # Get base metrics from parent class
        base_metrics = super().evaluate_chunks(chunks)
        
        # Add enhanced semantic analysis if Jina is available
        if self.use_jina_embeddings and self.jina_provider:
            try:
                # Replace semantic coherence with embedding-based analysis
                embedding_coherence = self._analyze_semantic_coherence_with_embeddings(chunks)
                base_metrics['semantic_coherence'] = embedding_coherence
                
                # Add embedding-specific metrics
                base_metrics['embedding_metrics'] = self._get_embedding_metrics(chunks)
                base_metrics['jina_enhanced'] = True
                
                # Add topic clustering analysis
                base_metrics['topic_clustering'] = self._analyze_topic_clustering(chunks)
                
                # Hybrid mode: compare with TF-IDF
                if self.hybrid_mode:
                    tfidf_coherence = super()._analyze_semantic_coherence(chunks)
                    base_metrics['hybrid_analysis'] = self._create_hybrid_analysis(
                        embedding_coherence, tfidf_coherence
                    )
                
                # Recalculate overall score with enhanced metrics
                base_metrics['overall_score'] = self._calculate_enhanced_overall_score(base_metrics)
                
            except Exception as e:
                self.logger.error(f"Jina embedding analysis failed: {e}")
                if self.fallback_to_tfidf:
                    base_metrics['jina_enhanced'] = False
                    base_metrics['embedding_metrics'] = {
                        'fallback_used': True,
                        'fallback_reason': str(e)
                    }
                else:
                    raise
        else:
            base_metrics['jina_enhanced'] = False
            base_metrics['embedding_metrics'] = {'provider': 'none'}
        
        return base_metrics
    
    def _analyze_semantic_coherence_with_embeddings(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze semantic coherence using Jina embeddings."""
        
        if len(chunks) < 2:
            return {'coherence_score': 1.0, 'avg_similarity': 0.0, 'similarity_std': 0.0, 'embedding_based': True}
        
        try:
            # Generate embeddings for all chunks
            embeddings = self._generate_chunk_embeddings(chunks)
            
            # Calculate similarity matrix
            similarities = self._calculate_embedding_similarities(embeddings)
            
            # Calculate adjacent chunk similarities (coherence)
            adjacent_similarities = []
            for i in range(len(similarities) - 1):
                adjacent_similarities.append(similarities[i][i + 1])
            
            avg_adjacent_similarity = np.mean(adjacent_similarities) if adjacent_similarities else 0
            
            # Apply boost factor and normalize
            coherence_score = min(1.0, avg_adjacent_similarity * self.coherence_score_boost_factor)
            
            # Calculate overall similarity statistics
            upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
            overall_avg_similarity = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0
            
            return {
                'coherence_score': coherence_score,
                'avg_similarity': overall_avg_similarity,
                'similarity_std': np.std(adjacent_similarities) if adjacent_similarities else 0,
                'embedding_based': True,
                'adjacent_similarities': adjacent_similarities,
                'similarity_matrix_shape': similarities.shape
            }
            
        except Exception as e:
            self.logger.error(f"Embedding-based coherence analysis failed: {e}")
            # Fallback to TF-IDF if enabled
            if self.fallback_to_tfidf:
                fallback_result = super()._analyze_semantic_coherence(chunks)
                fallback_result['embedding_based'] = False
                fallback_result['fallback_used'] = True
                return fallback_result
            else:
                raise
    
    def _generate_chunk_embeddings(self, chunks: List[Document]) -> np.ndarray:
        """Generate embeddings for chunks using Jina."""
        
        # Extract text content
        texts = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
        
        if not texts:
            return np.array([])
        
        # Check cache first
        cache_key = None
        if self.enable_embedding_cache:
            cache_key = self._generate_cache_key(texts)
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
        
        # Generate embeddings via Jina
        embedding_response = self.jina_provider.generate_embeddings(texts)
        embeddings = np.array(embedding_response.embeddings)
        
        # Normalize dimensions if needed
        embeddings = self._normalize_embedding_dimensions(embeddings)
        
        # Cache the result
        if self.enable_embedding_cache and cache_key:
            self.embedding_cache[cache_key] = embeddings
        
        return embeddings
    
    def _calculate_embedding_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between embeddings."""
        if embeddings.size == 0:
            return np.array([])
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_embeddings = embeddings / norms
        
        # Calculate cosine similarity matrix
        similarities = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Ensure values are in valid range [-1, 1]
        similarities = np.clip(similarities, -1, 1)
        
        return similarities
    
    def _normalize_embedding_dimensions(self, embeddings) -> np.ndarray:
        """Normalize embedding dimensions to ensure consistency."""
        if embeddings is None:
            return np.array([])
        
        # Handle numpy array input
        if isinstance(embeddings, np.ndarray):
            return embeddings
        
        # Handle list input
        if isinstance(embeddings, list) and len(embeddings) == 0:
            return np.array([])
        
        embeddings_array = np.array(embeddings)
        
        # If all embeddings have the same dimension, return as-is
        if len(embeddings_array.shape) == 2:
            return embeddings_array
        
        # Handle variable-length embeddings by padding/truncating
        max_dim = max(len(emb) for emb in embeddings)
        normalized = []
        
        for emb in embeddings:
            if len(emb) < max_dim:
                # Pad with zeros
                padded = list(emb) + [0.0] * (max_dim - len(emb))
                normalized.append(padded)
            else:
                # Truncate to max_dim
                normalized.append(emb[:max_dim])
        
        return np.array(normalized)
    
    def _get_embedding_metrics(self, chunks: List[Document]) -> Dict[str, Any]:
        """Get metrics about the embedding process."""
        
        try:
            texts = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
            
            # Generate a sample embedding to get metadata
            if texts:
                sample_response = self.jina_provider.generate_embeddings(texts[:1])
                
                return {
                    'provider': 'jina',
                    'model': sample_response.model,
                    'tokens_used': sample_response.tokens_used * len(texts),  # Estimate total
                    'embedding_dimension': len(sample_response.embeddings[0]) if sample_response.embeddings else 0,
                    'cache_hit': False,  # Updated during actual generation
                    'chunk_count': len(texts)
                }
            else:
                return {
                    'provider': 'jina',
                    'model': self.jina_provider.embedding_model,
                    'tokens_used': 0,
                    'embedding_dimension': 0,
                    'chunk_count': 0
                }
                
        except Exception as e:
            return {
                'provider': 'jina',
                'error': str(e),
                'chunk_count': len(chunks)
            }
    
    def _analyze_topic_clustering(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze topic clustering using Jina embeddings."""
        
        if len(chunks) < 3:
            return {
                'topic_clusters': [],
                'cluster_coherence': 1.0,
                'outlier_chunks': []
            }
        
        try:
            # Generate embeddings
            embeddings = self._generate_chunk_embeddings(chunks)
            
            if embeddings.size == 0:
                return {
                    'topic_clusters': [],
                    'cluster_coherence': 0.0,
                    'outlier_chunks': []
                }
            
            # Determine optimal number of clusters (2 to sqrt(n))
            n_chunks = len(embeddings)
            max_clusters = min(int(np.sqrt(n_chunks)), 5)
            n_clusters = max(2, max_clusters)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Analyze clusters
            topic_clusters = []
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_indices) > 0:
                    topic_clusters.append({
                        'cluster_id': int(cluster_id),
                        'chunk_indices': cluster_indices.tolist(),
                        'size': len(cluster_indices)
                    })
            
            # Calculate cluster coherence
            silhouette_scores = []
            for i, embedding in enumerate(embeddings):
                # Calculate average distance to same cluster
                same_cluster = embeddings[cluster_labels == cluster_labels[i]]
                if len(same_cluster) > 1:
                    same_cluster_dist = np.mean([
                        np.linalg.norm(embedding - other) 
                        for other in same_cluster 
                        if not np.array_equal(embedding, other)
                    ])
                else:
                    same_cluster_dist = 0
                
                # Simple coherence approximation
                silhouette_scores.append(1.0 / (1.0 + same_cluster_dist))
            
            cluster_coherence = np.mean(silhouette_scores) if silhouette_scores else 0
            
            # Identify outliers (chunks far from their cluster center)
            outlier_chunks = []
            for i, embedding in enumerate(embeddings):
                cluster_center = kmeans.cluster_centers_[cluster_labels[i]]
                distance = np.linalg.norm(embedding - cluster_center)
                # Consider outlier if distance > 75th percentile
                if distance > np.percentile([
                    np.linalg.norm(emb - kmeans.cluster_centers_[cluster_labels[j]]) 
                    for j, emb in enumerate(embeddings)
                ], 75):
                    outlier_chunks.append(i)
            
            return {
                'topic_clusters': topic_clusters,
                'cluster_coherence': float(cluster_coherence),
                'outlier_chunks': outlier_chunks,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            self.logger.error(f"Topic clustering analysis failed: {e}")
            return {
                'topic_clusters': [],
                'cluster_coherence': 0.0,
                'outlier_chunks': [],
                'error': str(e)
            }
    
    def _create_hybrid_analysis(self, embedding_coherence: Dict, tfidf_coherence: Dict) -> Dict[str, Any]:
        """Create hybrid analysis comparing TF-IDF and Jina embeddings."""
        
        jina_score = embedding_coherence.get('coherence_score', 0)
        tfidf_score = tfidf_coherence.get('coherence_score', 0)
        
        # Calculate agreement between methods
        agreement_score = 1.0 - abs(jina_score - tfidf_score)
        
        # Create combined score (weighted average)
        # Jina typically gets higher weight due to superior semantic understanding
        jina_weight = 0.7
        tfidf_weight = 0.3
        combined_score = (jina_score * jina_weight) + (tfidf_score * tfidf_weight)
        
        return {
            'jina_score': jina_score,
            'tfidf_score': tfidf_score,
            'combined_score': combined_score,
            'agreement_score': agreement_score,
            'method_weights': {'jina': jina_weight, 'tfidf': tfidf_weight}
        }
    
    def _calculate_enhanced_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate enhanced overall score incorporating embedding insights."""
        
        try:
            # Base scoring from parent class
            base_score = super()._calculate_overall_score(metrics)
            
            # Enhancement bonus from embedding analysis
            enhancement_bonus = 0
            
            if metrics.get('jina_enhanced', False):
                # Bonus for high-quality semantic coherence
                coherence_score = metrics['semantic_coherence'].get('coherence_score', 0)
                if coherence_score > 0.8:
                    enhancement_bonus += 5  # Up to 5 bonus points for excellent coherence
                elif coherence_score > 0.6:
                    enhancement_bonus += 2  # Small bonus for good coherence
                
                # Bonus for good topic clustering
                clustering = metrics.get('topic_clustering', {})
                cluster_coherence = clustering.get('cluster_coherence', 0)
                if cluster_coherence > 0.7:
                    enhancement_bonus += 3  # Bonus for well-clustered content
                
                # Penalty for too many outliers
                outlier_ratio = len(clustering.get('outlier_chunks', [])) / len(metrics.get('chunks', [1]))
                if outlier_ratio > 0.3:
                    enhancement_bonus -= 2  # Penalty for fragmented content
            
            # Hybrid mode bonus
            if metrics.get('hybrid_analysis'):
                agreement_score = metrics['hybrid_analysis'].get('agreement_score', 0)
                if agreement_score > 0.8:
                    enhancement_bonus += 2  # Bonus for method agreement
            
            enhanced_score = base_score + enhancement_bonus
            return min(100, max(0, enhanced_score))
            
        except Exception as e:
            self.logger.error(f"Enhanced scoring calculation failed: {e}")
            return super()._calculate_overall_score(metrics)
    
    def _calculate_embedding_quality_metrics(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """Calculate quality metrics for embeddings themselves."""
        
        if not embeddings:
            return {}
        
        embeddings_array = np.array(embeddings)
        
        # Calculate embedding variance (higher = more diverse)
        embedding_variance = float(np.var(embeddings_array))
        
        # Calculate embedding entropy (approximation)
        flattened = embeddings_array.flatten()
        hist, _ = np.histogram(flattened, bins=50)
        hist = hist / np.sum(hist)  # Normalize
        embedding_entropy = float(-np.sum(hist * np.log(hist + 1e-10)))
        
        # Calculate separation quality
        if len(embeddings) > 1:
            pairwise_distances = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    dist = np.linalg.norm(np.array(embeddings[i]) - np.array(embeddings[j]))
                    pairwise_distances.append(dist)
            
            separation_quality = float(np.mean(pairwise_distances)) if pairwise_distances else 0
        else:
            separation_quality = 0
        
        return {
            'embedding_variance': embedding_variance,
            'embedding_entropy': embedding_entropy,
            'separation_quality': separation_quality
        }
    
    def _generate_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for embedding results."""
        combined_text = "|".join(texts)
        return hashlib.md5(combined_text.encode()).hexdigest()
    
    def generate_enhanced_report(self, chunks: List[Document], output_path: str = None) -> str:
        """Generate enhanced evaluation report with Jina insights."""
        
        metrics = self.evaluate_chunks(chunks)
        
        # Start with base report
        report = super().generate_report(chunks, output_path=None)
        
        # Add enhanced analysis section
        if metrics.get('jina_enhanced', False):
            report += f"""

## 🚀 Enhanced Analysis with Jina AI Embeddings

### Embedding Metrics
- **Provider**: {metrics['embedding_metrics'].get('provider', 'N/A')}
- **Model**: {metrics['embedding_metrics'].get('model', 'N/A')}
- **Tokens Used**: {metrics['embedding_metrics'].get('tokens_used', 0)}
- **Embedding Dimension**: {metrics['embedding_metrics'].get('embedding_dimension', 0)}

### Advanced Semantic Coherence
- **Embedding-based Coherence**: {metrics['semantic_coherence'].get('coherence_score', 0):.3f}
- **Average Similarity**: {metrics['semantic_coherence'].get('avg_similarity', 0):.3f}
- **Similarity Variance**: {metrics['semantic_coherence'].get('similarity_std', 0):.3f}

### Topic Clustering Analysis
- **Number of Topics**: {metrics['topic_clustering'].get('n_clusters', 0)}
- **Cluster Coherence**: {metrics['topic_clustering'].get('cluster_coherence', 0):.3f}
- **Outlier Chunks**: {len(metrics['topic_clustering'].get('outlier_chunks', []))}

"""
            
            # Add hybrid analysis if available
            if metrics.get('hybrid_analysis'):
                hybrid = metrics['hybrid_analysis']
                report += f"""### Hybrid Analysis (TF-IDF vs Jina)
- **Jina Score**: {hybrid['jina_score']:.3f}
- **TF-IDF Score**: {hybrid['tfidf_score']:.3f}
- **Combined Score**: {hybrid['combined_score']:.3f}
- **Method Agreement**: {hybrid['agreement_score']:.3f}

"""
        
        # Enhanced recommendations
        report += "## 🎯 Enhanced Recommendations\n"
        
        if metrics.get('jina_enhanced', False):
            coherence = metrics['semantic_coherence'].get('coherence_score', 0)
            if coherence > 0.8:
                report += "- ✅ Excellent semantic coherence detected with Jina embeddings!\n"
            elif coherence < 0.4:
                report += "- ⚠️  Low semantic coherence. Consider reorganizing content flow.\n"
            
            clustering = metrics.get('topic_clustering', {})
            outlier_ratio = len(clustering.get('outlier_chunks', [])) / max(1, len(chunks))
            if outlier_ratio > 0.3:
                report += "- ⚠️  High number of outlier chunks detected. Content may be fragmented.\n"
            
            if metrics.get('hybrid_analysis', {}).get('agreement_score', 1) < 0.5:
                report += "- ⚠️  Low agreement between TF-IDF and Jina analysis. Content may need review.\n"
        else:
            report += "- 💡 Consider enabling Jina AI embeddings for superior semantic analysis.\n"
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Enhanced evaluation report saved to {output_path}")
        
        return report
