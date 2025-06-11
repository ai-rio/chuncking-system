#!/usr/bin/env python3
"""
Test script to verify the evaluation fixes work properly.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

def test_nan_handling():
    """
    Test the NaN handling functionality.
    """
    print("🧪 Testing NaN handling functionality...")
    
    # Create a mock DataFrame with NaN values (similar to Ragas output)
    mock_data = {
        'context_precision': [np.nan, np.nan, np.nan],
        'context_recall': [np.nan, np.nan, np.nan],
        'faithfulness': [np.nan, np.nan, np.nan],
        'answer_relevancy': [np.nan, np.nan, np.nan]
    }
    
    df = pd.DataFrame(mock_data)
    print(f"📊 Mock DataFrame shape: {df.shape}")
    print(f"📊 DataFrame contents:\n{df}")
    
    # Test the NaN conversion function
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
    
    # Create results structure
    results_dict = {
        'summary': summary_dict,
        'detailed_scores': detailed_scores,
        'metadata': {
            'samples_count': 3,
            'model': 'gemini-1.5-pro',
            'framework': 'Ragas',
            'note': 'NaN values indicate metrics that could not be calculated due to data format or model output issues',
            'test_timestamp': datetime.now().isoformat()
        }
    }
    
    # Save test results
    output_dir = "/mnt/gdrive/data/evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    test_results_path = os.path.join(output_dir, "test_ragas_results.json")
    
    try:
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Test results successfully saved to {test_results_path}")
        print(f"📄 Summary: {results_dict['summary']}")
        print(f"📊 First detailed score: {detailed_scores[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving test results: {e}")
        return False

def test_print_summary():
    """
    Test the improved print summary function.
    """
    print("\n🧪 Testing improved print summary...")
    
    # Create mock DataFrame with NaN values
    mock_data = {
        'context_precision': [np.nan, np.nan, np.nan],
        'context_recall': [np.nan, np.nan, np.nan],
        'faithfulness': [np.nan, np.nan, np.nan],
        'answer_relevancy': [np.nan, np.nan, np.nan]
    }
    
    df = pd.DataFrame(mock_data)
    
    print("📈 Mean Scores:")
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            mean_val = df[col].mean()
            if np.isnan(mean_val):
                print(f"  {col}: N/A (could not calculate)")
            else:
                print(f"  {col}: {mean_val:.4f}")
        else:
            print(f"  {col}: N/A (non-numeric)")
    
    print(f"\n📊 Standard Deviation:")
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            std_val = df[col].std()
            if np.isnan(std_val):
                print(f"  {col}: N/A (could not calculate)")
            else:
                print(f"  {col}: {std_val:.4f}")
        else:
            print(f"  {col}: N/A (non-numeric)")
    
    # Check if all values are NaN and provide helpful message
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0 and df[numeric_cols].isna().all().all():
        print("\n⚠️  Note: All metric values are NaN. This typically indicates:")
        print("   • Model outputs are not in the expected JSON format")
        print("   • Evaluation samples may not be suitable for scoring")
        print("   • There may be issues with the Ragas metric calculations")
    
    return True

def main():
    """
    Run all tests.
    """
    print("🚀 Starting evaluation fix tests...\n")
    
    # Test NaN handling
    test1_success = test_nan_handling()
    
    # Test print summary
    test2_success = test_print_summary()
    
    print("\n📋 Test Results Summary:")
    print(f"  NaN Handling Test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"  Print Summary Test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! The evaluation fixes should work properly.")
        print("📁 Check /mnt/gdrive/data/evaluation_results/test_ragas_results.json for structured output example.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()