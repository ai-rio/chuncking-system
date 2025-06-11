#!/usr/bin/env python3
"""
Test script to verify Ragas configuration with Gemini models.

This script tests the RAG evaluator's ability to use Gemini models
for evaluation instead of OpenAI models.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.evaluators.rag_evaluator import RAGEvaluator, EvaluationSample
from src.config.settings import config


def test_ragas_gemini_setup():
    """
    Test that Ragas can be configured with Gemini models.
    """
    print("🧪 Testing Ragas configuration with Gemini models...")
    
    try:
        # Initialize RAG evaluator
        evaluator = RAGEvaluator()
        
        # Check if Gemini models were configured successfully
        if evaluator.ragas_llm and evaluator.ragas_embeddings:
            print("✅ Gemini models configured successfully for Ragas")
            print(f"   LLM: {type(evaluator.ragas_llm).__name__}")
            print(f"   Embeddings: {type(evaluator.ragas_embeddings).__name__}")
            return True
        else:
            print("⚠️ Gemini models not configured, will use default Ragas models")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Ragas configuration: {e}")
        return False


def test_sample_evaluation():
    """
    Test evaluation with a small sample dataset.
    """
    print("\n🧪 Testing sample evaluation...")
    
    try:
        # Create a simple evaluation sample
        sample = EvaluationSample(
            question="What is machine learning?",
            answer="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
            contexts=["Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."],
            ground_truth="Machine learning is a subset of AI that allows computers to learn from data.",
            chunk_index=0,
            main_topic="Machine Learning"
        )
        
        # Initialize evaluator
        evaluator = RAGEvaluator()
        evaluator.evaluation_samples = [sample]
        
        # Test evaluation (this will use either Gemini or default models)
        print("🔄 Running evaluation...")
        results = evaluator.evaluate_with_ragas()
        
        if results:
            print("✅ Evaluation completed successfully")
            print(f"   Results type: {type(results).__name__}")
            return True
        else:
            print("❌ Evaluation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return False


def main():
    """
    Main test function.
    """
    print("🚀 Testing Ragas with Gemini Configuration")
    print("=" * 50)
    
    # Check API key
    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "your-api-key-here":
        print("⚠️ Warning: GEMINI_API_KEY not configured")
        print("   Ragas will fall back to default models (requires OpenAI API key)")
    else:
        print("✅ GEMINI_API_KEY is configured")
    
    # Test configuration
    config_success = test_ragas_gemini_setup()
    
    # Test evaluation (commented out to avoid API costs during testing)
    # evaluation_success = test_sample_evaluation()
    
    print("\n" + "=" * 50)
    if config_success:
        print("✅ Ragas is configured to use Gemini models")
        print("\n📝 Next steps:")
        print("   1. Run the full RAG evaluation: python evaluate_rag.py")
        print("   2. Ragas will now use Gemini instead of OpenAI for evaluation")
    else:
        print("⚠️ Ragas will use default models (OpenAI API key required)")
        print("\n📝 To use Gemini:")
        print("   1. Ensure GEMINI_API_KEY is set in your environment")
        print("   2. Install: pip install langchain-google-genai")


if __name__ == "__main__":
    main()