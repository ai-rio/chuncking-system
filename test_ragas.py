#!/usr/bin/env python3
# test_ragas.py
"""
Simple test script to verify Ragas installation and basic functionality.
"""

import sys
import os

print("🔍 Testing Ragas Installation...")
print("=" * 40)

try:
    import ragas
    print(f"✅ Ragas version: {ragas.__version__}")
except ImportError as e:
    print(f"❌ Ragas import failed: {e}")
    sys.exit(1)

try:
    from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
    print("✅ Ragas metrics imported successfully")
except ImportError as e:
    print(f"❌ Ragas metrics import failed: {e}")
    sys.exit(1)

try:
    from datasets import Dataset
    print("✅ Datasets library imported successfully")
except ImportError as e:
    print(f"❌ Datasets import failed: {e}")
    sys.exit(1)

try:
    from langchain_core.documents import Document
    print("✅ LangChain Document imported successfully")
except ImportError as e:
    print(f"❌ LangChain import failed: {e}")
    sys.exit(1)

# Test basic functionality
print("\n🧪 Testing Basic Functionality...")
print("-" * 40)

# Create a simple test dataset
test_data = {
    'question': ['What is the capital of France?'],
    'answer': ['The capital of France is Paris.'],
    'contexts': [['France is a country in Europe. Its capital city is Paris.']],
    'ground_truth': ['Paris']
}

try:
    dataset = Dataset.from_dict(test_data)
    print("✅ Test dataset created successfully")
    print(f"   Dataset size: {len(dataset)}")
except Exception as e:
    print(f"❌ Dataset creation failed: {e}")
    sys.exit(1)

print("\n🎉 All tests passed! Ragas is ready to use.")
print("=" * 40)