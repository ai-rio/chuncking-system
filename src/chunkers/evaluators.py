import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer # Keep for other metrics, but not main semantic
from sklearn.metrics.pairwise import cosine_similarity
import re
import os # Import os for path handling
import sys # Import sys for path manipulation

# Add the parent directory to the sys.path to allow imports from src.config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import src.config.settings as config # Import the settings module

# Import SentenceTransformer for semantic embeddings
from sentence_transformers import SentenceTransformer, util

class ChunkQualityEvaluator:
    """Evaluate the quality of document chunks"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.min_words_for_very_short = 10
        self.coherence_score_boost_factor = 1.0 # This might need adjustment after true semantic scores appear
        self.structure_score_weight_factor = 1.0

        # Initialize embedding model for semantic coherence evaluation
        self.embedding_model = None
        if config.config.EMBEDDING_MODEL:
            try:
                self.embedding_model = SentenceTransformer(config.config.EMBEDDING_MODEL)
                print(f"Evaluator loaded embedding model: {config.config.EMBEDDING_MODEL}")
            except Exception as e:
                print(f"Evaluator Error: Could not load embedding model {config.config.EMBEDDING_MODEL}: {e}. Semantic coherence evaluation will use TF-IDF fallback.")
        else:
            print("Evaluator Warning: EMBEDDING_MODEL not set in settings. Semantic coherence evaluation will use TF-IDF fallback.")


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
        """
        Analyze content quality metrics, now smarter about incomplete sentences
        for different content types.
        """
        
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
            
            if len(content.split()) < self.min_words_for_very_short:
                quality_metrics['very_short_chunks'] += 1
            
            # Determine if the chunk is structural (header, code, table, list, image description)
            is_structural = False
            # Check for chunk_type 'structural' or 'visual' which are set by HybridChunker
            if chunk.metadata.get('chunk_type') in ['structural', 'visual']:
                is_structural = True
            elif re.match(r'^#+\s', content) or any(h in chunk.metadata for h in ["Part", "Chapter", "Section", "Sub-section"]):
                is_structural = True
            elif re.search(r'^\s*[-*+]\s+.*|\s*\d+\.\s+.*', content, re.MULTILINE):
                is_structural = True


            # Only check for incomplete sentences if it's not a structural chunk
            if not is_structural and not re.search(r'[.!?:]$', content):
                quality_metrics['incomplete_sentences'] += 1
            
        total_chunks = len(chunks)
        quality_percentages = {
            f"{key}_pct": (value / total_chunks * 100) if total_chunks > 0 else 0
            for key, value in quality_metrics.items()
        }
        
        return {**quality_metrics, **quality_percentages}
    
    def _analyze_semantic_coherence(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Analyze semantic coherence between chunks using SentenceTransformer if available,
        otherwise fall back to TF-IDF.
        
        Returns:
            Dict with the following metrics:
            - boundary_score: Measure of semantic separation between adjacent chunks (higher is better for RAG)
            - internal_cohesion_score: Measure of semantic uniformity within each chunk (higher is better)
            - avg_similarity: Average similarity between all chunks (for reference)
            - similarity_std: Standard deviation of similarities (for reference)
        """
        
        # Filter for prose chunks only for semantic coherence
        # Now explicitly looking for 'prose' chunk_type set by HybridChunker
        prose_chunks = [
            chunk for chunk in chunks
            if chunk.metadata.get('chunk_type') == 'prose'
        ]

        if len(prose_chunks) < 2:
            return {
                'boundary_score': 1.0,  # Perfect boundary score when no comparison needed
                'internal_cohesion_score': 1.0,  # Perfect cohesion when only one chunk
                'avg_similarity': 0.0, 
                'similarity_std': 0.0
            }
        
        texts = [chunk.page_content for chunk in prose_chunks if chunk.page_content.strip()]
        if len(texts) < 2:
            return {
                'boundary_score': 1.0,  # Perfect boundary score when no comparison needed
                'internal_cohesion_score': 1.0,  # Perfect cohesion when only one chunk
                'avg_similarity': 0.0, 
                'similarity_std': 0.0
            }

        try:
            if self.embedding_model:
                # Use SentenceTransformer for semantic coherence
                chunk_embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
                # Calculate cosine similarity between adjacent chunk embeddings
                adjacent_similarities = util.cos_sim(chunk_embeddings[:-1], chunk_embeddings[1:]).diag().cpu().numpy()
                
                avg_adjacent_similarity = np.mean(adjacent_similarities) if adjacent_similarities.size > 0 else 0
                
                # Calculate boundary score: 1 - adjacent_similarity
                # Lower adjacent similarity means better semantic boundaries between chunks
                boundary_score = 1.0 - avg_adjacent_similarity
                
                # Calculate internal cohesion for each chunk
                internal_cohesion_scores = []
                for i, text in enumerate(texts):
                    # Split text into sentences or paragraphs for internal analysis
                    segments = [s.strip() for s in text.split('.') if s.strip()]
                    if len(segments) > 1:
                        # Encode segments and calculate average similarity within the chunk
                        segment_embeddings = self.embedding_model.encode(segments, convert_to_tensor=True)
                        segment_similarities = util.cos_sim(segment_embeddings, segment_embeddings).cpu().numpy()
                        # Get average of upper triangle (excluding diagonal)
                        internal_similarity = np.mean(segment_similarities[np.triu_indices_from(segment_similarities, k=1)])
                        internal_cohesion_scores.append(internal_similarity)
                
                # Average internal cohesion across all chunks
                internal_cohesion_score = np.mean(internal_cohesion_scores) if internal_cohesion_scores else 0.5
                
                # Calculate overall average similarity between all chunks (for reference)
                overall_avg_similarity = np.mean(util.cos_sim(chunk_embeddings, chunk_embeddings).cpu().numpy()[np.triu_indices_from(util.cos_sim(chunk_embeddings, chunk_embeddings).cpu().numpy(), k=1)])
                
                return {
                    'boundary_score': boundary_score,  # Higher is better (indicates distinct chunks)
                    'internal_cohesion_score': internal_cohesion_score,  # Higher is better (indicates coherent chunks)
                    'avg_similarity': overall_avg_similarity,
                    'similarity_std': np.std(adjacent_similarities) if adjacent_similarities.size > 0 else 0
                }
            else:
                # Fallback to TF-IDF if embedding model is not loaded
                print("Semantic coherence: Falling back to TF-IDF for evaluation (embedding model not loaded).")
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                similarities = cosine_similarity(tfidf_matrix)
                
                adjacent_similarities = []
                for i in range(len(similarities) - 1):
                    adjacent_similarities.append(similarities[i][i + 1])
                
                avg_adjacent_similarity = np.mean(adjacent_similarities) if adjacent_similarities else 0
                
                # Calculate boundary score: 1 - adjacent_similarity
                boundary_score = 1.0 - avg_adjacent_similarity
                
                # For TF-IDF fallback, use a simplified internal cohesion calculation
                internal_cohesion_scores = []
                for i, text in enumerate(texts):
                    segments = [s.strip() for s in text.split('.') if s.strip()]
                    if len(segments) > 1:
                        segment_matrix = self.vectorizer.fit_transform(segments)
                        segment_similarities = cosine_similarity(segment_matrix)
                        internal_similarity = np.mean(segment_similarities[np.triu_indices_from(segment_similarities, k=1)])
                        internal_cohesion_scores.append(internal_similarity)
                
                internal_cohesion_score = np.mean(internal_cohesion_scores) if internal_cohesion_scores else 0.5
                
                # Calculate overall average similarity between all chunks (for reference)
                overall_avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
                
                return {
                    'boundary_score': boundary_score,  # Higher is better (indicates distinct chunks)
                    'internal_cohesion_score': internal_cohesion_score,  # Higher is better (indicates coherent chunks)
                    'avg_similarity': overall_avg_similarity,
                    'similarity_std': np.std(adjacent_similarities) if adjacent_similarities else 0
                }
            
        except Exception as e:
            print(f"Error in semantic analysis: {e}")
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
            'chunks_with_links': 0,
            'chunks_with_tables': 0,
            'chunks_with_images': 0 # Added new metric for images
        }
        
        for chunk in chunks:
            content = chunk.page_content
            
            if any(h in chunk.metadata for h in ["Part", "Chapter", "Section", "Sub-section"]):
                structure_metrics['chunks_with_headers'] += 1
            elif re.search(r'^#+\s', content, re.MULTILINE):
                structure_metrics['chunks_with_headers'] += 1
            
            if '```' in content or (chunk.metadata.get('content_type') == 'code') or \
               re.search(r'\b(def|class|import|function|const|var)\b', content):
                structure_metrics['chunks_with_code'] += 1
            
            if re.search(r'^\s*[-*+]\s+.*|\s*\d+\.\s+.*', content, re.MULTILINE):
                structure_metrics['chunks_with_lists'] += 1
            
            if '[' in content and '](' in content:
                structure_metrics['chunks_with_links'] += 1

            if chunk.metadata.get('content_type') == 'table':
                structure_metrics['chunks_with_tables'] += 1

            # Check for image content type in metadata OR the new has_images flag
            if chunk.metadata.get('has_images', False): # Modified to directly check the has_images flag
                structure_metrics['chunks_with_images'] += 1

        total_chunks = len(chunks)
        structure_percentages = {
            f"{key}_pct": (value / total_chunks * 100) if total_chunks > 0 else 0
            for key, value in structure_metrics.items()
        }
        
        return {**structure_metrics, **structure_percentages}
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)
        
        The score now properly rewards:
        1. Good semantic boundaries between chunks (higher boundary_score)
        2. High internal cohesion within each chunk (higher internal_cohesion_score)
        3. Consistent chunk sizes
        4. Good content quality (no empty/short chunks, complete sentences)
        5. Good structure preservation
        """
        
        try:
            # Size consistency score (0-20 points)
            size_score = metrics['size_distribution']['size_consistency'] * 20
            
            # Content quality score (0-30 points)
            content_metrics = metrics['content_quality']
            content_score_raw = (
                (100 - content_metrics['empty_chunks_pct']) * 0.6 +
                (100 - content_metrics['very_short_chunks_pct']) * 0.3 +
                (100 - content_metrics['incomplete_sentences_pct']) * 0.1
            )
            content_score = (content_score_raw / 100) * 30
            
            # Semantic quality score (0-25 points)
            # Now using boundary_score (higher is better) and internal_cohesion_score (higher is better)
            semantic_metrics = metrics['semantic_coherence']
            boundary_score = semantic_metrics.get('boundary_score', 0.5)  # Default if not present
            internal_cohesion = semantic_metrics.get('internal_cohesion_score', 0.5)  # Default if not present
            
            # Combine boundary and cohesion scores with appropriate weights
            # Boundary score (distinct chunks) is weighted more heavily for RAG applications
            semantic_score = (boundary_score * 0.6 + internal_cohesion * 0.4) * 25
            
            # Structure preservation score (0-25 points)
            structure_metrics = metrics['structural_preservation']
            structure_score_raw = (
                structure_metrics['chunks_with_headers_pct'] * 0.25 + # Slightly reduced weight to balance
                structure_metrics['chunks_with_code_pct'] * 0.15 +
                structure_metrics['chunks_with_lists_pct'] * 0.1 +
                structure_metrics['chunks_with_links_pct'] * 0.1 +
                structure_metrics['chunks_with_tables_pct'] * 0.2 +
                structure_metrics['chunks_with_images_pct'] * 0.2 # New weight for images
            )
            structure_score = (structure_score_raw / 100) * 25
            
            # Calculate overall score (0-100 points)
            overall_score = size_score + content_score + semantic_score + structure_score
            return min(100, max(0, overall_score))
            
        except Exception as e:
            print(f"Error calculating overall score: {e}")
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

## Semantic Quality
- **Boundary Score**: {metrics['semantic_coherence'].get('boundary_score', 0.0):.3f} (higher is better - indicates distinct chunks)
- **Internal Cohesion**: {metrics['semantic_coherence'].get('internal_cohesion_score', 0.0):.3f} (higher is better - indicates coherent chunks)
- **Average Similarity**: {metrics['semantic_coherence']['avg_similarity']:.3f} (for reference)

## Structure Preservation
- **Chunks with Headers**: {metrics['structural_preservation']['chunks_with_headers']} ({metrics['structural_preservation']['chunks_with_headers_pct']:.1f}%)
- **Chunks with Code**: {metrics['structural_preservation']['chunks_with_code']} ({metrics['structural_preservation']['chunks_with_code_pct']:.1f}%)
- **Chunks with Lists**: {metrics['structural_preservation']['chunks_with_lists']} ({metrics['structural_preservation']['chunks_with_lists_pct']:.1f}%)
- **Chunks with Tables**: {metrics['structural_preservation']['chunks_with_tables']} ({metrics['structural_preservation']['chunks_with_tables_pct']:.1f}%)
- **Chunks with Images**: {metrics['structural_preservation']['chunks_with_images']} ({metrics['structural_preservation']['chunks_with_images_pct']:.1f}%)

## Recommendations
"""
        
        # Add recommendations based on metrics
        if metrics['content_quality']['empty_chunks_pct'] > 5:
            report += "- ⚠️  High number of empty chunks detected. Consider improving preprocessing.\n"
        
        if metrics['content_quality']['very_short_chunks_pct'] > 20:
            report += "- ⚠️  Many very short chunks. Consider increasing minimum chunk size.\n"
        
        if metrics['semantic_coherence'].get('boundary_score', 0.0) < 0.5:
            report += "- ⚠️  Low semantic boundary score. Consider reducing chunk overlap to create more distinct chunks.\n"
        
        if metrics['semantic_coherence'].get('internal_cohesion_score', 0.0) < 0.4:
            report += "- ⚠️  Low internal cohesion within chunks. Consider using semantic chunking to improve chunk coherence.\n"
        
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
        
        return report

