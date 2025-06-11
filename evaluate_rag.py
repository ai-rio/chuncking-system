#!/usr/bin/env python3
# evaluate_rag.py
"""
RAG Evaluation Script

This script demonstrates the integration of Ragas evaluation framework with our
Hybrid Document Chunking System. It loads existing chunks, generates synthetic
evaluation data, and produces comprehensive evaluation reports.

Usage:
    python evaluate_rag.py
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from langchain_core.documents import Document
from src.evaluators.rag_evaluator import RAGEvaluator
from src.utils.metadata_enricher import MetadataEnricher
from src.config.settings import config


def load_chunks_from_json(file_path: str) -> List[Document]:
    """
    Load document chunks from JSON file and convert to Document objects.
    
    Args:
        file_path: Path to the JSON file containing chunks
        
    Returns:
        List of Document objects
    """
    print(f"📂 Loading chunks from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        documents = []
        for chunk_data in chunks_data:
            # Create Document object from chunk data
            doc = Document(
                page_content=chunk_data['content'],
                metadata=chunk_data.get('metadata', {})
            )
            documents.append(doc)
        
        print(f"✅ Loaded {len(documents)} chunks")
        return documents
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return []
    except Exception as e:
        print(f"❌ Error loading chunks: {e}")
        return []


def print_evaluation_summary(results: Dict[str, Any], samples_count: int) -> None:
    """
    Print a summary of evaluation results to console with proper NaN handling.
    
    Args:
        results: Ragas evaluation results
        samples_count: Number of samples evaluated
    """
    print("\n" + "="*60)
    print("🎯 RAG EVALUATION RESULTS SUMMARY")
    print("="*60)
    print(f"📊 Samples Evaluated: {samples_count}")
    print(f"🔧 Framework: Ragas")
    print(f"🤖 LLM Model: {config.LLM_METADATA_MODEL}")
    
    # Extract and display metrics
    if hasattr(results, 'to_pandas'):
        df = results.to_pandas()
        print("\n📈 METRICS OVERVIEW:")
        print("-" * 40)
        
        metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
        for metric in metrics:
            if metric in df.columns:
                mean_score = df[metric].mean()
                std_score = df[metric].std()
                if np.isnan(mean_score) or np.isnan(std_score):
                    print(f"{metric.replace('_', ' ').title():20}: N/A (could not calculate)")
                else:
                    print(f"{metric.replace('_', ' ').title():20}: {mean_score:.3f} (±{std_score:.3f})")
        
        print("\n📋 DETAILED SCORES:")
        print("-" * 40)
        for i, row in df.iterrows():
            print(f"Sample {i+1:2d}: ", end="")
            for metric in metrics:
                if metric in row:
                    if pd.isna(row[metric]):
                        print(f"{metric[:8]:8s}=N/A     ", end="")
                    else:
                        print(f"{metric[:8]:8s}={row[metric]:.3f} ", end="")
            print()
        
        # Check if all values are NaN and provide helpful message
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and df[numeric_cols].isna().all().all():
            print("\n⚠️  Note: All metric values are NaN. This typically indicates:")
            print("   • Model outputs are not in the expected JSON format")
            print("   • Evaluation samples may not be suitable for scoring")
            print("   • There may be issues with the Ragas metric calculations")
    
    print("\n💡 INTERPRETATION GUIDE:")
    print("-" * 40)
    print("• Context Precision: Higher = Better context relevance (0-1)")
    print("• Context Recall:    Higher = Better information retrieval (0-1)")
    print("• Faithfulness:      Higher = More factually accurate answers (0-1)")
    print("• Answer Relevancy:  Higher = More relevant answers to questions (0-1)")
    print("\n🎯 Target: All metrics should ideally be > 0.7 for production use")
    print("="*60)


async def main_async() -> None:
    """
    Main asynchronous function to run RAG evaluation.
    """
    print("🚀 Starting RAG Evaluation with Ragas Framework")
    print("=" * 50)
    
    # Configuration
    chunks_file = "data/output/chunks/sample_document_chunks.json"
    output_dir = "data/evaluation_results"
    max_samples = 5  # Start small for testing
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Load existing chunks
        print("\n📋 STEP 1: Loading Document Chunks")
        chunks = load_chunks_from_json(chunks_file)
        
        if not chunks:
            print("❌ No chunks found. Please ensure chunks are generated first.")
            print("💡 Run: python main.py")
            return
        
        print(f"✅ Found {len(chunks)} chunks to evaluate")
        
        # Step 2: Initialize evaluator
        print("\n🔧 STEP 2: Initializing RAG Evaluator")
        metadata_enricher = MetadataEnricher()
        evaluator = RAGEvaluator(metadata_enricher=metadata_enricher)
        
        # Step 3: Generate synthetic evaluation data
        print("\n🎲 STEP 3: Generating Synthetic QA Pairs")
        evaluation_samples = await evaluator.generate_synthetic_qa_pairs(
            chunks=chunks, 
            max_samples=max_samples
        )
        
        if not evaluation_samples:
            print("❌ Failed to generate evaluation samples")
            return
        
        # Display sample questions for verification
        print("\n📝 Generated Questions (Preview):")
        for i, sample in enumerate(evaluation_samples[:3]):
            print(f"  {i+1}. {sample.question}")
            print(f"     Topic: {sample.main_topic}")
            print(f"     Answer: {sample.answer[:100]}...\n")
        
        # Step 4: Run Ragas evaluation
        print("\n🔍 STEP 4: Running Ragas Evaluation")
        results = evaluator.evaluate_with_ragas(evaluation_samples)
        
        # Step 5: Display results
        print_evaluation_summary(results, len(evaluation_samples))
        
        # Step 6: Generate reports
        print("\n📊 STEP 5: Generating Reports")
        
        # Save evaluation data
        eval_data_path = os.path.join(output_dir, "evaluation_data.json")
        evaluator.save_evaluation_data(eval_data_path)
        
        # Generate markdown report
        report_path = os.path.join(output_dir, "rag_evaluation_report.md")
        report_content = evaluator.generate_evaluation_report(results, report_path)
        
        # Save detailed results as JSON with proper NaN handling
        results_path = os.path.join(output_dir, "ragas_results.json")
        try:
            if hasattr(results, 'to_pandas'):
                df = results.to_pandas()
                
                # Handle NaN values gracefully
                import numpy as np
                
                # Convert NaN to None for JSON serialization
                def convert_nan_to_none(obj):
                    if isinstance(obj, dict):
                        return {k: convert_nan_to_none(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_nan_to_none(item) for item in obj]
                    elif isinstance(obj, float) and np.isnan(obj):
                        return None
                    else:
                        return obj
                
                # Calculate summary with NaN handling
                summary_dict = {}
                for col in df.columns:
                    if df[col].dtype in ['float64', 'int64']:
                        mean_val = df[col].mean()
                        summary_dict[col] = None if np.isnan(mean_val) else mean_val
                    else:
                        summary_dict[col] = None
                
                # Convert detailed scores with NaN handling
                detailed_scores = convert_nan_to_none(df.to_dict('records'))
                
                results_dict = {
                    'summary': summary_dict,
                    'detailed_scores': detailed_scores,
                    'metadata': {
                        'samples_count': len(evaluation_samples),
                        'model': config.LLM_METADATA_MODEL,
                        'framework': 'Ragas',
                        'note': 'NaN values indicate metrics that could not be calculated due to data format or model output issues'
                    }
                }
                
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(results_dict, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Detailed results saved to {results_path}")
            else:
                # Fallback: save raw results as string representation
                fallback_dict = {
                    'raw_results': str(results),
                    'metadata': {
                        'samples_count': len(evaluation_samples),
                        'model': config.LLM_METADATA_MODEL,
                        'framework': 'Ragas',
                        'note': 'Raw results saved due to unexpected format'
                    }
                }
                
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(fallback_dict, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Fallback results saved to {results_path}")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not save detailed results as JSON: {e}")
            # Create a minimal results file with error information
            error_dict = {
                'error': str(e),
                'metadata': {
                    'samples_count': len(evaluation_samples),
                    'model': config.LLM_METADATA_MODEL,
                    'framework': 'Ragas',
                    'note': 'Error occurred during results processing'
                }
            }
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(error_dict, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Error information saved to {results_path}")
        
        print("\n🎉 RAG Evaluation Complete!")
        print(f"📁 Results saved in: {output_dir}/")
        print(f"📋 Report available at: {report_path}")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Please install required packages:")
        print("   pip install ragas datasets langchain-openai")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    """
    Main function to run the evaluation.
    """
    # Check if we're in Colab and handle accordingly
    try:
        import google.colab
        print("🔍 Detected Google Colab environment")
    except ImportError:
        pass
    
    # Run the async main function
    asyncio.run(main_async())


if __name__ == "__main__":
    main()