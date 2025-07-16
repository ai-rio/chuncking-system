"""
Strategy optimizer for holistic chunking strategy enhancement.
Created following TDD principles - minimal implementation to pass tests.
"""

import re
import time
from typing import Dict, Any, Optional
from hashlib import md5


class StrategyOptimizer:
    """Optimizer that analyzes content and recommends optimal chunking strategies."""
    
    def __init__(self, enable_caching: bool = False):
        self.enable_caching = enable_caching
        self._cache = {} if enable_caching else None
    
    def analyze_content_characteristics(self, content: str) -> Dict[str, Any]:
        """Analyze content characteristics to determine optimal chunking strategy."""
        # Basic content analysis
        sentences = re.split(r'[.!?]+', content)
        paragraphs = content.split('\n\n')
        
        # Calculate code density
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        code_density = len(''.join(code_blocks)) / len(content) if content else 0
        
        # Calculate header frequency
        headers = re.findall(r'^#+\s+', content, re.MULTILINE)
        header_frequency = len(headers) / len(content.split('\n')) if content else 0
        
        # Calculate list density
        lists = re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE)
        list_density = len(lists) / len(content.split('\n')) if content else 0
        
        # Calculate sentence and paragraph lengths
        sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
        paragraph_lengths = [len(p.strip()) for p in paragraphs if p.strip()]
        
        return {
            'sentence_length_avg': sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0,
            'paragraph_length_avg': sum(paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0,
            'code_density': code_density,
            'header_frequency': header_frequency,
            'list_density': list_density,
            'technical_complexity': self._assess_technical_complexity(content)
        }
    
    def recommend_strategy(self, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal chunking strategy based on content analysis."""
        # Check cache first
        if self.enable_caching and self._cache is not None:
            cache_key = self._generate_cache_key(content_analysis)
            if cache_key in self._cache:
                cached_result = self._cache[cache_key].copy()
                cached_result['cache_hit'] = True
                return cached_result
        
        # Strategy recommendation logic
        recommendation = {
            'primary_strategy': 'sentence_boundary',  # Default
            'chunk_size': 800,
            'overlap_size': 150,
            'separators': ['\n\n', '\n', '. ', '? ', '! ', ' ', ''],
            'confidence': 0.8,
            'cache_hit': False
        }
        
        # Adjust based on content characteristics
        if content_analysis['code_density'] > 0.3:
            recommendation['primary_strategy'] = 'content_aware'
            recommendation['chunk_size'] = 600
            recommendation['overlap_size'] = 50
        elif content_analysis['header_frequency'] > 0.1:
            recommendation['primary_strategy'] = 'semantic_boundary'
            recommendation['chunk_size'] = 1000
            recommendation['overlap_size'] = 100
        elif content_analysis['list_density'] > 0.2:
            recommendation['primary_strategy'] = 'sentence_boundary'
            recommendation['chunk_size'] = 800
            recommendation['overlap_size'] = 150
        
        # Cache the result
        if self.enable_caching and self._cache is not None:
            cache_key = self._generate_cache_key(content_analysis)
            self._cache[cache_key] = recommendation.copy()
        
        return recommendation
    
    def _assess_technical_complexity(self, content: str) -> float:
        """Assess technical complexity of content."""
        # Simple heuristic: count technical terms
        technical_patterns = [
            r'\b(function|class|method|variable|parameter|return|import|export)\b',
            r'\b(API|HTTP|JSON|XML|SQL|database|server|client)\b',
            r'\b(algorithm|optimization|performance|scalability)\b'
        ]
        
        total_matches = 0
        for pattern in technical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            total_matches += len(matches)
        
        return min(total_matches / len(content.split()) if content else 0, 1.0)
    
    def _generate_cache_key(self, content_analysis: Dict[str, Any]) -> str:
        """Generate cache key from content analysis."""
        # Simple cache key based on analysis characteristics, rounded to be more lenient
        key_data = f"{content_analysis['code_density']:.1f}_{content_analysis['header_frequency']:.1f}_{content_analysis['list_density']:.1f}_{content_analysis['technical_complexity']:.1f}"
        return md5(key_data.encode()).hexdigest()