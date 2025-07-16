"""
Strategy tester for comparing multiple chunking strategies.
Created following TDD principles - minimal implementation to pass tests.
"""

from typing import Dict, Any, List, Optional
from .strategy_optimizer import StrategyOptimizer
from .evaluators import ChunkQualityEvaluator
from .hybrid_chunker import HybridMarkdownChunker


class StrategyTester:
    """Test multiple chunking strategies and compare their effectiveness."""
    
    def __init__(self):
        self.evaluator = ChunkQualityEvaluator()
        self.optimizer = StrategyOptimizer()
    
    def test_multiple_strategies(self, content: str, original_metadata: Dict[str, Any], strategies: Optional[List[str]] = None) -> Dict[str, Any]:
        """Test multiple strategies and return comparative results."""
        
        if strategies is None:
            strategies = ['fixed_size', 'semantic_boundary', 'content_aware', 'sentence_boundary']
        
        # Handle edge cases
        if not content or not content.strip():
            return self._handle_empty_content(strategies)
        
        results = {}
        
        for strategy in strategies:
            try:
                # Get strategy-specific parameters
                params = self._get_strategy_parameters(content, strategy)
                
                # Create chunker with strategy-specific configuration
                chunker = self._create_chunker(strategy, params)
                
                # Generate chunks
                chunks = chunker.chunk_document(content, metadata=original_metadata)
                
                # Evaluate quality
                quality_metrics = self.evaluator.evaluate_chunks(chunks)
                
                results[strategy] = {
                    'chunks': chunks,
                    'quality_metrics': quality_metrics,
                    'parameters_used': params,
                    'chunk_count': len(chunks),
                    'overall_score': quality_metrics.get('overall_score', 0)
                }
                
            except Exception as e:
                results[strategy] = {
                    'error': str(e),
                    'overall_score': 0
                }
        
        return self._rank_strategies(results)
    
    def _handle_empty_content(self, strategies: List[str]) -> Dict[str, Any]:
        """Handle empty or problematic content."""
        results = {}
        
        for strategy in strategies:
            results[strategy] = {
                'error': 'Empty or invalid content',
                'overall_score': 0
            }
        
        return {
            'best_strategy': None,
            'all_results': results,
            'recommendations': []
        }
    
    def _get_strategy_parameters(self, content: str, strategy: str) -> Dict[str, Any]:
        """Get parameters for a specific strategy."""
        # Analyze content for strategy-specific parameters
        analysis = self.optimizer.analyze_content_characteristics(content)
        
        base_params = {
            'chunk_size': 800,
            'chunk_overlap': 150,
            'strategy': strategy
        }
        
        # Strategy-specific parameter adjustments
        if strategy == 'fixed_size':
            base_params['chunk_size'] = 800
            base_params['chunk_overlap'] = 150
        elif strategy == 'semantic_boundary':
            base_params['chunk_size'] = 1000
            base_params['chunk_overlap'] = 100
        elif strategy == 'content_aware':
            base_params['chunk_size'] = 600
            base_params['chunk_overlap'] = 50
        elif strategy == 'sentence_boundary':
            base_params['chunk_size'] = 800
            base_params['chunk_overlap'] = 150
        
        return base_params
    
    def _create_chunker(self, strategy: str, params: Dict[str, Any]) -> HybridMarkdownChunker:
        """Create a chunker with strategy-specific configuration."""
        # For now, use the existing HybridMarkdownChunker with different parameters
        return HybridMarkdownChunker(
            chunk_size=params['chunk_size'],
            chunk_overlap=params['chunk_overlap']
        )
    
    def _rank_strategies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Rank strategies by performance."""
        
        # Sort by overall score
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return {
                'best_strategy': None,
                'all_results': results,
                'recommendations': ['All strategies failed - check content validity']
            }
        
        ranked = sorted(
            valid_results.items(),
            key=lambda x: x[1].get('overall_score', 0),
            reverse=True
        )
        
        best_strategy = ranked[0][0] if ranked else None
        
        recommendations = []
        if best_strategy:
            best_score = ranked[0][1]['overall_score']
            recommendations.append(f"Best strategy: {best_strategy} (score: {best_score:.1f})")
            
            # Add specific recommendations based on strategy
            if best_strategy == 'content_aware':
                recommendations.append("Content has mixed formats - use content-aware chunking")
            elif best_strategy == 'semantic_boundary':
                recommendations.append("Content is well-structured - use semantic boundaries")
            elif best_strategy == 'sentence_boundary':
                recommendations.append("Content is narrative - use sentence boundaries")
            elif best_strategy == 'fixed_size':
                recommendations.append("Content is uniform - fixed size chunking works well")
        
        return {
            'best_strategy': best_strategy,
            'all_results': results,
            'recommendations': recommendations
        }