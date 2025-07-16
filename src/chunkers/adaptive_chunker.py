"""
Adaptive chunker that selects optimal strategy based on content analysis.
Created following TDD principles - minimal implementation to pass tests.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from .hybrid_chunker import HybridMarkdownChunker
from .strategy_tester import StrategyTester
from .strategy_optimizer import StrategyOptimizer


class AdaptiveChunker(HybridMarkdownChunker):
    """Chunker that adapts parameters based on content analysis."""
    
    def __init__(self, auto_optimize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.auto_optimize = auto_optimize
        self.strategy_tester = StrategyTester() if auto_optimize else None
        self.optimizer = StrategyOptimizer() if auto_optimize else None
    
    def chunk_document_adaptive(self, content: str, 
                               metadata: Dict[str, Any] = None) -> List[Document]:
        """Chunk document using adaptive strategy selection."""
        
        if metadata is None:
            metadata = {}
        
        if self.auto_optimize and len(content) > 1000:  # Only optimize for substantial content
            # Analyze content characteristics
            content_analysis = self.optimizer.analyze_content_characteristics(content)
            
            # Test multiple strategies if content is complex enough
            if len(content) > 5000:  # Only for substantial content
                strategy_results = self.strategy_tester.test_multiple_strategies(content)
                best_strategy = strategy_results['best_strategy']
                
                # Use best strategy's parameters
                if best_strategy and best_strategy in strategy_results['all_results']:
                    optimal_params = strategy_results['all_results'][best_strategy]['parameters_used']
                    return self._chunk_with_parameters(content, optimal_params, metadata, best_strategy)
            
            # For smaller content, use optimizer recommendations
            else:
                recommendation = self.optimizer.recommend_strategy(content_analysis)
                return self._chunk_with_parameters(content, recommendation, metadata, recommendation['primary_strategy'])
        
        # Fallback to current hybrid approach
        chunks = self.chunk_document(content, metadata)
        
        # Add default metadata
        for chunk in chunks:
            chunk.metadata['strategy_used'] = 'hybrid_default'
            chunk.metadata['optimization_applied'] = False
        
        return chunks
    
    def _chunk_with_parameters(self, content: str, params: Dict[str, Any], 
                              metadata: Dict[str, Any], strategy_used: str) -> List[Document]:
        """Chunk content with specific parameters."""
        
        # Temporarily adjust chunker parameters
        original_size = self.chunk_size
        original_overlap = self.chunk_overlap
        
        try:
            self.chunk_size = params.get('chunk_size', self.chunk_size)
            self.chunk_overlap = params.get('overlap_size', self.chunk_overlap)
            
            # Reinitialize splitters with new parameters
            self._init_splitters()
            
            # Chunk the content
            chunks = self.chunk_document(content, metadata)
            
            # Add strategy metadata
            for chunk in chunks:
                chunk.metadata['strategy_used'] = strategy_used
                chunk.metadata['optimization_applied'] = True
                chunk.metadata['parameters_used'] = params
            
            return chunks
            
        finally:
            # Restore original parameters
            self.chunk_size = original_size
            self.chunk_overlap = original_overlap
            self._init_splitters()