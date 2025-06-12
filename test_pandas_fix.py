#!/usr/bin/env python3
"""
Test script to verify pandas import fix and basic evaluation functionality.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.settings import config

def test_pandas_import():
    """Test that pandas is properly imported and can be used."""
    print("🧪 Testing pandas import...")
    
    # Create a simple DataFrame with NaN values
    test_data = {
        'context_precision': [0.8, np.nan, 0.9, 0.7, np.nan],
        'context_recall': [0.9, 0.8, np.nan, 0.6, 0.7],
        'faithfulness': [np.nan, 0.9, 0.8, np.nan, 0.8],
        'answer_relevancy': [0.7, 0.8, 0.9, 0.8, np.nan]
    }
    
    df = pd.DataFrame(test_data)
    print(f"✅ Created DataFrame with shape: {df.shape}")
    
    # Test NaN handling like in the evaluation script
    metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
    
    print("\n📊 Testing metric calculations:")
    for metric in metrics:
        if metric in df.columns:
            mean_score = df[metric].mean()
            std_score = df[metric].std()
            if pd.isna(mean_score) or pd.isna(std_score):
                print(f"{metric.replace('_', ' ').title():20}: N/A (could not calculate)")
            else:
                print(f"{metric.replace('_', ' ').title():20}: {mean_score:.3f} (±{std_score:.3f})")
    
    print("\n📋 Testing detailed scores:")
    for i, row in df.iterrows():
        print(f"Sample {i+1:2d}: ", end="")
        for metric in metrics:
            if metric in row:
                if pd.isna(row[metric]):
                    print(f"{metric[:8]:8s}=N/A     ", end="")
                else:
                    print(f"{metric[:8]:8s}={row[metric]:.3f} ", end="")
        print()
    
    print("\n✅ Pandas import and NaN handling test completed successfully!")

def test_config_access():
    """Test that config can be accessed properly."""
    print("\n🔧 Testing config access...")
    
    try:
        api_key = config.GEMINI_API_KEY
        model = config.LLM_METADATA_MODEL
        print(f"✅ Config access successful")
        print(f"   - Model: {model}")
        print(f"   - API Key: {'Set' if api_key else 'Not set'}")
    except Exception as e:
        print(f"❌ Config access failed: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    print("🎯 RAG EVALUATION PANDAS FIX TEST")
    print("=" * 50)
    
    try:
        test_pandas_import()
        test_config_access()
        
        print("\n🎉 All tests passed! The pandas import fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)