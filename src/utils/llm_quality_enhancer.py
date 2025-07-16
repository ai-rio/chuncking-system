"""
LLM-powered quality enhancement system for chunk content.

This module provides intelligent content enhancement using Large Language Models
to improve readability, coherence, and completeness of chunked content.
"""

import json
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from langchain_core.documents import Document

from src.utils.logger import get_logger
from src.utils.llm_client import LLMClient
from src.chunkers.evaluators import ChunkQualityEvaluator


class LLMQualityEnhancer:
    """
    LLM-powered quality enhancement for document chunks.
    
    This class uses Large Language Models to intelligently improve
    chunk quality through content rewriting, semantic enhancement,
    and contextual gap filling.
    """
    
    def __init__(self, 
                 llm_provider: str = "google",
                 llm_model: str = "gemini-2.0-flash-exp",
                 temperature: float = 0.1,
                 max_tokens: int = 1000,
                 cost_optimization: bool = True,
                 batch_processing: bool = True,
                 max_tokens_per_request: int = 4000):
        """
        Initialize LLM Quality Enhancer.
        
        Args:
            llm_provider: LLM provider (google, openai, anthropic, local)
            llm_model: Specific model to use
            temperature: Model temperature for generation
            max_tokens: Maximum tokens per response
            cost_optimization: Whether to optimize for cost
            batch_processing: Whether to batch requests
            max_tokens_per_request: Maximum tokens per batch request
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_optimization = cost_optimization
        self.batch_processing = batch_processing
        self.max_tokens_per_request = max_tokens_per_request
        
        self.logger = get_logger(__name__)
        self.llm_client = LLMClient(provider=llm_provider, model=llm_model)
        self.evaluator = ChunkQualityEvaluator()
        
        # Initialize enhancement prompts
        self.enhancement_prompts = self._load_enhancement_prompts()
        
        # Performance tracking
        self.total_api_calls = 0
        self.total_tokens_used = 0
        self.total_processing_time = 0.0
    
    def _load_enhancement_prompts(self) -> Dict[str, str]:
        """Load enhancement prompts for different tasks."""
        return {
            "content_rewriting": """
You are an expert content editor. Your task is to improve the following text chunk while preserving its original meaning and metadata accuracy.

Focus on:
1. Completing incomplete sentences
2. Improving readability and flow
3. Enhancing clarity without changing core meaning
4. Maintaining technical accuracy
5. Preserving any code, links, or formatting

Original chunk:
{content}

Content type: {content_type}
Context: {context}

Provide your response in JSON format:
{{
    "enhanced_content": "improved version of the content",
    "improvements_made": ["list of specific improvements"],
    "confidence_score": 0.95,
    "reasoning": "explanation of changes made"
}}
""",
            
            "semantic_coherence": """
You are an expert at improving semantic coherence between text chunks. Analyze the following chunks and enhance them to flow better together.

Focus on:
1. Adding appropriate transitions between chunks
2. Ensuring logical flow and continuity
3. Maintaining context across chunk boundaries
4. Preserving individual chunk integrity
5. Improving overall narrative coherence

Chunks to enhance:
{chunks}

Provide your response in JSON format:
{{
    "enhanced_chunks": [
        {{"content": "enhanced chunk 1", "metadata": {{}}}},
        {{"content": "enhanced chunk 2", "metadata": {{}}}}
    ],
    "coherence_score": 0.92,
    "transitions_added": 2,
    "improvements_made": ["list of coherence improvements"]
}}
""",
            
            "contextual_gaps": """
You are an expert at identifying and filling contextual gaps in text. Analyze the following chunks and identify missing context or unclear references.

Focus on:
1. Identifying unclear references or pronouns
2. Finding missing contextual information
3. Detecting logical gaps between chunks
4. Providing contextual bridging where needed
5. Maintaining accuracy and coherence

Chunks to analyze:
{chunks}

Provide your response in JSON format:
{{
    "gap_analysis": {{
        "gaps_found": 2,
        "gap_descriptions": ["description of each gap found"]
    }},
    "enhanced_chunks": [
        {{"content": "enhanced chunk with filled gaps", "metadata": {{}}}}
    ],
    "context_score": 0.88,
    "improvements_made": ["list of contextual improvements"]
}}
""",
            
            "completeness_validation": """
You are an expert content validator. Analyze the following chunks for completeness and identify missing elements.

Focus on:
1. Incomplete sentences or thoughts
2. Missing explanations or details
3. Unclear or ambiguous content
4. Fragmented information
5. Overall content completeness

Chunks to validate:
{chunks}

Provide your response in JSON format:
{{
    "completeness_analysis": {{
        "complete_chunks": 2,
        "incomplete_chunks": 3,
        "missing_elements": ["list of missing elements"]
    }},
    "completeness_score": 0.65,
    "recommendations": ["list of specific recommendations"],
    "priority_fixes": ["most important fixes needed"]
}}
""",
            
            "quality_metrics": """
You are an expert content quality analyst. Analyze the following chunks and provide comprehensive quality metrics.

Focus on:
1. Readability and clarity
2. Semantic coherence
3. Content completeness
4. Context clarity
5. Overall quality assessment

Chunks to analyze:
{chunks}

Provide your response in JSON format:
{{
    "quality_metrics": {{
        "readability_score": 0.75,
        "coherence_score": 0.68,
        "completeness_score": 0.45,
        "context_clarity": 0.55,
        "overall_quality": 0.60
    }},
    "detailed_analysis": {{
        "strengths": ["list of content strengths"],
        "weaknesses": ["list of content weaknesses"],
        "suggestions": ["list of improvement suggestions"]
    }},
    "confidence": 0.92
}}
""",
            
            "comprehensive_enhancement": """
You are an expert content enhancer. Perform comprehensive enhancement on the following chunks to significantly improve their quality.

Apply all relevant improvements:
1. Content rewriting for clarity
2. Semantic coherence enhancement
3. Contextual gap filling
4. Sentence completion
5. Flow and readability improvements

Original chunks:
{chunks}

Additional context:
- Content type: {content_type}
- Target quality score: 80+
- Preserve metadata accuracy
- Maintain technical precision

Provide your response in JSON format:
{{
    "enhanced_chunks": [
        {{"content": "comprehensively enhanced chunk", "metadata": {{"enhanced": true}}}}
    ],
    "enhancement_summary": {{
        "original_score": 45.0,
        "enhanced_score": 82.0,
        "improvement": 37.0,
        "enhancements_applied": ["list of enhancements applied"]
    }},
    "quality_validation": {{
        "meets_threshold": true,
        "confidence": 0.90
    }}
}}
"""
        }
    
    def enhance_chunk_content(self, chunk: Document) -> Document:
        """
        Enhance individual chunk content using LLM.
        
        Args:
            chunk: Document chunk to enhance
            
        Returns:
            Enhanced document chunk
        """
        try:
            start_time = time.time()
            
            # Prepare context for LLM
            content_type = chunk.metadata.get('content_type', 'general')
            context = self._extract_context(chunk)
            
            # Format prompt
            prompt = self.enhancement_prompts["content_rewriting"].format(
                content=chunk.page_content,
                content_type=content_type,
                context=context
            )
            
            # Call LLM
            response = self._call_llm(prompt)
            
            # Process response
            if isinstance(response, dict) and 'enhanced_content' in response:
                enhanced_content = response['enhanced_content']
                
                # Create enhanced chunk
                enhanced_chunk = Document(
                    page_content=enhanced_content,
                    metadata={
                        **chunk.metadata,
                        'llm_enhanced': True,
                        'llm_provider': self.llm_provider,
                        'llm_model': self.llm_model,
                        'enhancement_confidence': response.get('confidence_score', 0.8),
                        'improvements_made': response.get('improvements_made', []),
                        'llm_processing_time': time.time() - start_time,
                        'llm_tokens_used': self._estimate_tokens(prompt + enhanced_content),
                        'llm_api_calls': 1
                    }
                )
                
                return enhanced_chunk
            else:
                # Return original chunk if enhancement failed
                return chunk
                
        except Exception as e:
            self.logger.error(f"Error enhancing chunk content: {str(e)}")
            return chunk
    
    def enhance_semantic_coherence(self, chunks: List[Document]) -> List[Document]:
        """
        Enhance semantic coherence between chunks.
        
        Args:
            chunks: List of chunks to enhance coherence
            
        Returns:
            List of enhanced chunks with improved coherence
        """
        try:
            if len(chunks) < 2:
                return chunks
            
            # Prepare chunks for LLM
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    'content': chunk.page_content,
                    'metadata': chunk.metadata,
                    'index': chunk.metadata.get('chunk_index', 0)
                })
            
            # Format prompt
            prompt = self.enhancement_prompts["semantic_coherence"].format(
                chunks=json.dumps(chunks_data, indent=2)
            )
            
            # Call LLM
            response = self._call_llm(prompt)
            
            # Process response
            if isinstance(response, dict) and 'enhanced_chunks' in response:
                enhanced_chunks = []
                
                for i, chunk_data in enumerate(response['enhanced_chunks']):
                    if i < len(chunks):
                        original_chunk = chunks[i]
                        enhanced_chunk = Document(
                            page_content=chunk_data['content'],
                            metadata={
                                **original_chunk.metadata,
                                **chunk_data.get('metadata', {}),
                                'llm_enhanced': True,
                                'coherence_enhanced': True,
                                'coherence_score': response.get('coherence_score', 0.8)
                            }
                        )
                        enhanced_chunks.append(enhanced_chunk)
                
                return enhanced_chunks
            else:
                return chunks
                
        except Exception as e:
            self.logger.error(f"Error enhancing semantic coherence: {str(e)}")
            return chunks
    
    def fill_contextual_gaps(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Fill contextual gaps in chunks using LLM.
        
        Args:
            chunks: List of chunks to analyze for gaps
            
        Returns:
            Dictionary with gap analysis and enhanced chunks
        """
        try:
            # Prepare chunks for analysis
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    'content': chunk.page_content,
                    'metadata': chunk.metadata,
                    'index': chunk.metadata.get('chunk_index', 0)
                })
            
            # Format prompt
            prompt = self.enhancement_prompts["contextual_gaps"].format(
                chunks=json.dumps(chunks_data, indent=2)
            )
            
            # Call LLM
            response = self._call_llm(prompt)
            
            # Process response
            if isinstance(response, dict):
                return response
            else:
                return {
                    "gap_analysis": {"gaps_found": 0, "gap_descriptions": []},
                    "enhanced_chunks": [],
                    "context_score": 0.5
                }
                
        except Exception as e:
            self.logger.error(f"Error filling contextual gaps: {str(e)}")
            return {
                "gap_analysis": {"gaps_found": 0, "gap_descriptions": []},
                "enhanced_chunks": [],
                "context_score": 0.5
            }
    
    def validate_content_completeness(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Validate content completeness using LLM.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            Dictionary with completeness analysis
        """
        try:
            # Prepare chunks for validation
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    'content': chunk.page_content,
                    'metadata': chunk.metadata,
                    'index': chunk.metadata.get('chunk_index', 0)
                })
            
            # Format prompt
            prompt = self.enhancement_prompts["completeness_validation"].format(
                chunks=json.dumps(chunks_data, indent=2)
            )
            
            # Call LLM
            response = self._call_llm(prompt)
            
            # Process response
            if isinstance(response, dict):
                return response
            else:
                return {
                    "completeness_analysis": {
                        "complete_chunks": 0,
                        "incomplete_chunks": len(chunks),
                        "missing_elements": []
                    },
                    "completeness_score": 0.5,
                    "recommendations": []
                }
                
        except Exception as e:
            self.logger.error(f"Error validating content completeness: {str(e)}")
            return {
                "completeness_analysis": {
                    "complete_chunks": 0,
                    "incomplete_chunks": len(chunks),
                    "missing_elements": []
                },
                "completeness_score": 0.5,
                "recommendations": []
            }
    
    def calculate_llm_quality_metrics(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Calculate quality metrics using LLM analysis.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with quality metrics and analysis
        """
        try:
            # Prepare chunks for analysis
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    'content': chunk.page_content,
                    'metadata': chunk.metadata,
                    'index': chunk.metadata.get('chunk_index', 0)
                })
            
            # Format prompt
            prompt = self.enhancement_prompts["quality_metrics"].format(
                chunks=json.dumps(chunks_data, indent=2)
            )
            
            # Call LLM
            response = self._call_llm(prompt)
            
            # Process response
            if isinstance(response, dict):
                return response
            else:
                return {
                    "quality_metrics": {
                        "readability_score": 50.0,
                        "coherence_score": 50.0,
                        "completeness_score": 50.0,
                        "context_clarity": 50.0,
                        "overall_quality": 50.0
                    },
                    "detailed_analysis": {
                        "strengths": [],
                        "weaknesses": [],
                        "suggestions": []
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating LLM quality metrics: {str(e)}")
            return {
                "quality_metrics": {
                    "readability_score": 50.0,
                    "coherence_score": 50.0,
                    "completeness_score": 50.0,
                    "context_clarity": 50.0,
                    "overall_quality": 50.0
                },
                "detailed_analysis": {
                    "strengths": [],
                    "weaknesses": [],
                    "suggestions": []
                }
            }
    
    def comprehensive_enhance(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Perform comprehensive enhancement on chunks.
        
        Args:
            chunks: List of chunks to enhance
            
        Returns:
            Dictionary with enhanced chunks and summary
        """
        try:
            # Get original quality metrics
            original_metrics = self.evaluator.evaluate_chunks(chunks)
            original_score = original_metrics.get('overall_score', 0)
            
            # Prepare chunks for comprehensive enhancement
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    'content': chunk.page_content,
                    'metadata': chunk.metadata,
                    'index': chunk.metadata.get('chunk_index', 0)
                })
            
            # Determine content type
            content_type = self._determine_content_type(chunks)
            
            # Format prompt
            prompt = self.enhancement_prompts["comprehensive_enhancement"].format(
                chunks=json.dumps(chunks_data, indent=2),
                content_type=content_type
            )
            
            # Call LLM
            response = self._call_llm(prompt)
            
            # Process response
            if isinstance(response, dict) and 'enhanced_chunks' in response:
                enhanced_chunks = []
                
                for chunk_data in response['enhanced_chunks']:
                    enhanced_chunk = Document(
                        page_content=chunk_data['content'],
                        metadata={
                            **chunk_data.get('metadata', {}),
                            'llm_enhanced': True,
                            'llm_provider': self.llm_provider,
                            'llm_model': self.llm_model,
                            'comprehensive_enhancement': True
                        }
                    )
                    enhanced_chunks.append(enhanced_chunk)
                
                # Update enhancement summary
                enhancement_summary = response.get('enhancement_summary', {})
                enhancement_summary['original_score'] = original_score
                
                return {
                    "enhanced_chunks": response['enhanced_chunks'],
                    "enhancement_summary": enhancement_summary,
                    "quality_validation": response.get('quality_validation', {}),
                    "llm_enhanced": True
                }
            else:
                return {
                    "enhanced_chunks": chunks,
                    "enhancement_summary": {
                        "original_score": original_score,
                        "enhanced_score": original_score,
                        "improvement": 0,
                        "enhancements_applied": []
                    },
                    "llm_enhanced": False
                }
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive enhancement: {str(e)}")
            return {
                "enhanced_chunks": chunks,
                "enhancement_summary": {
                    "original_score": 0,
                    "enhanced_score": 0,
                    "improvement": 0,
                    "enhancements_applied": []
                },
                "llm_enhanced": False
            }
    
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM with the given prompt.
        
        Args:
            prompt: Prompt to send to LLM
            
        Returns:
            LLM response as dictionary
        """
        try:
            start_time = time.time()
            
            response = self.llm_client.complete(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Update performance tracking
            self.total_api_calls += 1
            self.total_processing_time += time.time() - start_time
            
            # Try to parse JSON response
            if isinstance(response, str):
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse LLM response as JSON")
                    return {"error": "Invalid JSON response"}
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error calling LLM: {str(e)}")
            return {"error": str(e)}
    
    def _extract_context(self, chunk: Document) -> str:
        """Extract relevant context from chunk metadata."""
        context_parts = []
        
        if 'source' in chunk.metadata:
            context_parts.append(f"Source: {chunk.metadata['source']}")
        
        if 'chunk_index' in chunk.metadata:
            context_parts.append(f"Chunk position: {chunk.metadata['chunk_index']}")
        
        if 'content_type' in chunk.metadata:
            context_parts.append(f"Content type: {chunk.metadata['content_type']}")
        
        return "; ".join(context_parts) if context_parts else "No additional context"
    
    def _determine_content_type(self, chunks: List[Document]) -> str:
        """Determine the primary content type from chunks."""
        content_types = []
        
        for chunk in chunks:
            if 'content_type' in chunk.metadata:
                content_types.append(chunk.metadata['content_type'])
        
        if content_types:
            # Return most common content type
            return max(set(content_types), key=content_types.count)
        
        # Analyze content to determine type
        combined_content = " ".join([chunk.page_content for chunk in chunks])
        
        if '```' in combined_content or 'def ' in combined_content:
            return 'technical'
        elif any(word in combined_content.lower() for word in ['story', 'character', 'plot']):
            return 'narrative'
        else:
            return 'general'
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for LLM usage."""
        return {
            'total_api_calls': self.total_api_calls,
            'total_tokens_used': self.total_tokens_used,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': self.total_processing_time / max(1, self.total_api_calls)
        }
    
    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API."""
        # This would be implemented with actual OpenAI API calls
        # For now, return a mock response
        return {
            "enhanced_content": "Mock enhanced content from OpenAI",
            "confidence_score": 0.85
        }
    
    def _call_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic API."""
        # This would be implemented with actual Anthropic API calls
        # For now, return a mock response
        return {
            "enhanced_content": "Mock enhanced content from Anthropic",
            "confidence_score": 0.85
        }
    
    def _call_google(self, prompt: str) -> Dict[str, Any]:
        """Call Google API."""
        # This would be implemented with actual Google API calls
        # For now, return a mock response
        return {
            "enhanced_content": "Mock enhanced content from Google",
            "confidence_score": 0.85
        }
    
    def _call_local(self, prompt: str) -> Dict[str, Any]:
        """Call local LLM."""
        # This would be implemented with local LLM calls
        # For now, return a mock response
        return {
            "enhanced_content": "Mock enhanced content from local LLM",
            "confidence_score": 0.85
        }