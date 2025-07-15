import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
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
            
            if '```' in content or (chunk.metadata.get('content_type') == 'code') or \
               re.search(r'\b(def|class|import|function|const|var)\b', content):
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
