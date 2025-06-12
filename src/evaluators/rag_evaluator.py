# src/evaluators/rag_evaluator.py
"""
RAG Evaluation Framework using Ragas

This module implements automated evaluation of our RAG pipeline using the Ragas framework.
It generates synthetic question-answer pairs from existing chunks and evaluates them using
key metrics: context_precision, context_recall, faithfulness, and answer_relevance.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from datetime import datetime

try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall, 
        faithfulness,
        answer_relevancy
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
except ImportError as e:
    print(f"Warning: Ragas dependencies not installed. Please run: pip install ragas datasets")
    print(f"Import error: {e}")
    raise

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except ImportError as e:
    print(f"Warning: Google Generative AI dependencies not installed. Please run: pip install langchain-google-genai")
    print(f"Import error: {e}")
    raise

from langchain_core.documents import Document
from ..utils.metadata_enricher import MetadataEnricher
from ..config.settings import config


@dataclass
class EvaluationSample:
    """Data structure for a single evaluation sample."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str
    chunk_index: int
    main_topic: str


class RAGEvaluator:
    """
    RAG Evaluation Framework using Ragas.
    
    This class provides functionality to:
    1. Generate synthetic evaluation datasets from existing chunks
    2. Evaluate RAG pipeline performance using Ragas metrics
    3. Generate comprehensive evaluation reports
    """
    
    def __init__(self, metadata_enricher: Optional[MetadataEnricher] = None):
        """
        Initialize the RAG evaluator.
        
        Args:
            metadata_enricher: Optional MetadataEnricher instance for LLM calls
        """
        self.metadata_enricher = metadata_enricher or MetadataEnricher()
        self.evaluation_samples: List[EvaluationSample] = []
        
        # Configure Gemini LLM and embeddings for Ragas
        self._setup_ragas_models()
        
        # Ragas metrics to evaluate
        self.metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ]
    
    def _setup_ragas_models(self) -> None:
        """
        Configure Gemini LLM and embeddings for Ragas evaluation.
        """
        try:
            # Initialize Gemini LLM for Ragas evaluation with minimal config
            gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.1,  # Low temperature for consistent evaluation
                google_api_key=config.GEMINI_API_KEY,
                max_output_tokens=1024  # Limit output tokens to avoid issues
            )
            
            # Initialize Gemini embeddings for Ragas
            gemini_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=config.GEMINI_API_KEY
            )
            
            # Wrap for Ragas compatibility
            self.ragas_llm = LangchainLLMWrapper(gemini_llm)
            self.ragas_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)
            
            print("✅ Configured Ragas to use Gemini models")
            
        except Exception as e:
            print(f"❌ Error setting up Ragas models: {e}")
            print("⚠️ Falling back to default Ragas models (may require OpenAI API key)")
            # Set to None to use default Ragas models
            self.ragas_llm = None
            self.ragas_embeddings = None
    
    async def generate_synthetic_qa_pairs(self, chunks: List[Document], max_samples: int = 5) -> List[EvaluationSample]:
        """
        Generate synthetic question-answer pairs from document chunks.
        
        Args:
            chunks: List of document chunks to generate QA pairs from
            max_samples: Maximum number of samples to generate
            
        Returns:
            List of EvaluationSample objects
        """
        print(f"🔄 Generating synthetic QA pairs from {len(chunks)} chunks...")
        
        evaluation_samples = []
        
        # Select diverse chunks for evaluation (skip very short ones)
        selected_chunks = []
        for i, chunk in enumerate(chunks):
            # Handle both Document objects and string inputs
            if isinstance(chunk, str):
                # Convert string to Document object
                chunk = Document(page_content=chunk, metadata={'chunk_index': i})
            elif not hasattr(chunk, 'page_content'):
                print(f"  ⚠️ Warning: Chunk {i} is not a Document object or string, skipping...")
                continue
                
            content = chunk.page_content.strip()
            word_count = len(content.split())
            
            # Debug: Print chunk info
            print(f"  Chunk {i}: {word_count} words, starts with: '{content[:50]}...'")
            
            # Skip very short chunks (less than 10 words) unless they're substantial headers (20+ words)
            if word_count >= 10:
                selected_chunks.append(chunk)
                print(f"  ✅ Selected chunk {i} ({word_count} words)")
                if len(selected_chunks) >= max_samples:
                    break
            else:
                print(f"  ❌ Skipped chunk {i} (too short: {word_count} words)")
        
        print(f"📝 Selected {len(selected_chunks)} chunks for QA generation")
        
        for i, chunk in enumerate(selected_chunks):
            try:
                print(f"  Generating QA pair {i+1}/{len(selected_chunks)}...")
                
                # Generate question from chunk content
                question = await self._generate_question_from_chunk(chunk)
                
                # Generate answer from chunk content  
                answer = await self._generate_answer_from_chunk(chunk, question)
                
                # Use the answer as ground truth (since we're generating synthetically)
                ground_truth = answer
                
                # Context is the chunk content
                contexts = [chunk.page_content]
                
                evaluation_sample = EvaluationSample(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth,
                    chunk_index=chunk.metadata.get('chunk_index', i),
                    main_topic=chunk.metadata.get('main_topic', 'Unknown')
                )
                
                evaluation_samples.append(evaluation_sample)
                
            except Exception as e:
                print(f"  ⚠️ Error generating QA pair for chunk {i}: {e}")
                continue
        
        self.evaluation_samples = evaluation_samples
        print(f"✅ Generated {len(evaluation_samples)} synthetic QA pairs")
        return evaluation_samples
    
    async def _generate_question_from_chunk(self, chunk: Document) -> str:
        """
        Generate a question that can be answered from the given chunk.
        
        Args:
            chunk: Document chunk to generate question from
            
        Returns:
            Generated question string
        """
        prompt = f"""
        Based on the following text chunk, generate a specific, answerable question that can be fully answered using ONLY the information provided in this text.
        
        The question should:
        - Be specific and focused
        - Require information from the text to answer
        - Not ask for information not present in the text
        - Be clear and well-formed
        
        Text chunk:
        {chunk.page_content}
        
        Generate only the question, no additional text:
        """
        
        try:
            response = await self.metadata_enricher._call_llm_async(prompt)
            # Clean up the response
            question = response.strip().strip('"').strip("'")
            if not question.endswith('?'):
                question += '?'
            return question
        except Exception as e:
            print(f"Error generating question: {e}")
            # Fallback question
            main_topic = chunk.metadata.get('main_topic', 'this topic')
            return f"What is discussed about {main_topic} in this text?"
    
    async def _generate_answer_from_chunk(self, chunk: Document, question: str) -> str:
        """
        Generate an answer to the question based on the chunk content.
        
        Args:
            chunk: Document chunk containing the information
            question: Question to answer
            
        Returns:
            Generated answer string
        """
        prompt = f"""
        Answer the following question based ONLY on the information provided in the text chunk. 
        Be concise but complete. If the text doesn't contain enough information to fully answer the question, say so.
        
        Question: {question}
        
        Text chunk:
        {chunk.page_content}
        
        Answer:
        """
        
        try:
            response = await self.metadata_enricher._call_llm_async(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            # Fallback answer
            return f"Based on the provided text: {chunk.page_content[:100]}..."
    
    def evaluate_with_ragas(self, evaluation_samples: Optional[List[EvaluationSample]] = None) -> Dict[str, Any]:
        """
        Evaluate the RAG pipeline using Ragas metrics.
        
        Args:
            evaluation_samples: Optional list of evaluation samples. Uses self.evaluation_samples if None.
            
        Returns:
            Dictionary containing evaluation results
        """
        if evaluation_samples is None:
            evaluation_samples = self.evaluation_samples
            
        if not evaluation_samples:
            raise ValueError("No evaluation samples available. Generate samples first.")
        
        print(f"🔍 Evaluating {len(evaluation_samples)} samples with Ragas...")
        
        # Prepare data for Ragas
        data = {
            'question': [sample.question for sample in evaluation_samples],
            'answer': [sample.answer for sample in evaluation_samples], 
            'contexts': [sample.contexts for sample in evaluation_samples],
            'ground_truth': [sample.ground_truth for sample in evaluation_samples]
        }
        
        # Create Hugging Face dataset
        dataset = Dataset.from_dict(data)
        
        try:
            # Run Ragas evaluation with Gemini models (if available)
            if self.ragas_llm and self.ragas_embeddings:
                print("🔍 Using Gemini models for Ragas evaluation...")
                result = evaluate(
                    dataset=dataset,
                    metrics=self.metrics,
                    llm=self.ragas_llm,
                    embeddings=self.ragas_embeddings
                )
            else:
                print("🔍 Using default Ragas models (requires OpenAI API key)...")
                result = evaluate(
                    dataset=dataset,
                    metrics=self.metrics
                )
            
            print("✅ Ragas evaluation completed")
            return result
            
        except Exception as e:
            print(f"❌ Error during Ragas evaluation: {e}")
            raise
    
    def generate_evaluation_report(self, results: Dict[str, Any], output_path: str) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Ragas evaluation results
            output_path: Path to save the report
            
        Returns:
            Report content as string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract metrics from results
        metrics_summary = {}
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            for metric in ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']:
                if metric in df.columns:
                    metrics_summary[metric] = {
                        'mean': df[metric].mean(),
                        'std': df[metric].std(),
                        'min': df[metric].min(),
                        'max': df[metric].max()
                    }
        
        report = f"""
# RAG Evaluation Report

**Generated:** {timestamp}  
**Framework:** Ragas  
**Samples Evaluated:** {len(self.evaluation_samples)}

## Executive Summary

This report presents the evaluation results of our RAG (Retrieval-Augmented Generation) pipeline using the Ragas framework <mcreference link="https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a/" index="1">1</mcreference>. The evaluation covers four key metrics that assess both retrieval and generation quality.

## Metrics Overview

### Core Ragas Metrics <mcreference link="https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a/" index="1">1</mcreference>

1. **Context Precision**: Measures the signal-to-noise ratio of retrieved context
2. **Context Recall**: Evaluates if all relevant information was retrieved  
3. **Faithfulness**: Assesses factual accuracy of generated answers
4. **Answer Relevancy**: Measures how relevant answers are to questions

## Results Summary

"""
        
        # Add metrics results
        for metric_name, stats in metrics_summary.items():
            report += f"""
### {metric_name.replace('_', ' ').title()}
- **Mean Score**: {stats['mean']:.3f}
- **Standard Deviation**: {stats['std']:.3f}  
- **Range**: {stats['min']:.3f} - {stats['max']:.3f}

"""
        
        # Add sample details
        report += """
## Sample Details

| Sample | Topic | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
|--------|-------|------------------|----------------|--------------|------------------|
"""
        
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            for i, sample in enumerate(self.evaluation_samples):
                if i < len(df):
                    row = df.iloc[i]
                    report += f"| {i+1} | {sample.main_topic[:20]}... | {row.get('context_precision', 'N/A'):.3f} | {row.get('context_recall', 'N/A'):.3f} | {row.get('faithfulness', 'N/A'):.3f} | {row.get('answer_relevancy', 'N/A'):.3f} |\n"
        
        report += """

## Recommendations

Based on the evaluation results:

1. **Context Precision**: If scores are low, consider improving chunk relevance filtering
2. **Context Recall**: Low scores suggest the need for better retrieval strategies  
3. **Faithfulness**: Poor scores indicate the LLM is hallucinating or adding information
4. **Answer Relevancy**: Low scores suggest prompt engineering improvements needed

## Technical Details

- **Evaluation Framework**: Ragas (Retrieval-Augmented Generation Assessment)
- **LLM Model**: {config.LLM_METADATA_MODEL}
- **Chunk Strategy**: Hybrid (Table-aware, Header-Recursive, Semantic, LLM-Enriched)
- **Synthetic Data**: Generated from existing document chunks

---
*Report generated by RAG Evaluation Framework*
"""
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📊 Evaluation report saved to {output_path}")
        return report
    
    def save_evaluation_data(self, output_path: str) -> None:
        """
        Save evaluation samples and results to JSON file.
        
        Args:
            output_path: Path to save the evaluation data
        """
        data = {
            'timestamp': datetime.now().isoformat(),
            'samples': [
                {
                    'question': sample.question,
                    'answer': sample.answer,
                    'contexts': sample.contexts,
                    'ground_truth': sample.ground_truth,
                    'chunk_index': sample.chunk_index,
                    'main_topic': sample.main_topic
                }
                for sample in self.evaluation_samples
            ]
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Evaluation data saved to {output_path}")