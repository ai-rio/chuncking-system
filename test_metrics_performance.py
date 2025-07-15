#!/usr/bin/env python3
"""
Performance test for MetricsRegistry optimization.
Tests the performance improvement from periodic cleanup vs cleanup on every registration.
"""

import time
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.observability import MetricsRegistry, MetricType, CustomMetric


def test_metrics_performance(num_metrics: int = 1000):
    """Test metrics registration performance."""
    print(f"Testing MetricsRegistry performance with {num_metrics} metrics...")
    
    # Create registry
    registry = MetricsRegistry()
    
    # Test metric registration performance
    start_time = time.time()
    
    for i in range(num_metrics):
        registry.record_metric(
            name=f"test.metric.{i % 10}",  # Create 10 different metric names
            value=i * 1.5,
            metric_type=MetricType.COUNTER,
            unit="operations",
            labels={"test": "performance", "iteration": str(i)}
        )
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Processed {i + 1:,} metrics in {elapsed:.2f}s ({rate:.1f} metrics/sec)")
    
    total_time = time.time() - start_time
    rate = num_metrics / total_time
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Rate: {rate:.1f} metrics/second")
    print(f"  Time per metric: {(total_time / num_metrics) * 1000:.3f} ms")
    
    # Check registry state
    print(f"\nRegistry state:")
    print(f"  Total metrics stored: {len(registry.metrics)}")
    print(f"  Unique metric names: {len(registry._metrics_dict)}")
    print(f"  Registration count: {registry._registration_count}")
    print(f"  Cleanup interval: {registry._cleanup_interval}")
    
    # Test cleanup performance
    print(f"\nTesting manual cleanup performance...")
    cleanup_start = time.time()
    registry.force_cleanup()
    cleanup_time = time.time() - cleanup_start
    print(f"  Cleanup took: {cleanup_time:.3f} seconds")
    print(f"  Metrics after cleanup: {len(registry.metrics)}")
    
    return total_time, rate


def benchmark_comparison():
    """Compare performance at different scales."""
    print("=" * 60)
    print("MetricsRegistry Performance Benchmark")
    print("=" * 60)
    
    test_sizes = [100, 500, 1000, 2000, 5000]
    
    for size in test_sizes:
        print(f"\n{'='*20} Testing {size:,} metrics {'='*20}")
        total_time, rate = test_metrics_performance(size)
        
        # Calculate performance grade
        if rate > 1000:
            grade = "EXCELLENT"
        elif rate > 500:
            grade = "GOOD"
        elif rate > 100:
            grade = "ACCEPTABLE"
        else:
            grade = "POOR"
        
        print(f"  Performance grade: {grade}")
        
        # Expected time for larger scales
        if size >= 1000:
            expected_10k = (10000 / rate)
            print(f"  Estimated time for 10,000 metrics: {expected_10k:.1f} seconds")


if __name__ == "__main__":
    benchmark_comparison()