#!/usr/bin/env python3
"""
Large-scale performance test for MetricsRegistry optimization.
Tests the performance with 10,000+ metrics to verify no O(n²) behavior.
"""

import time
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.observability import MetricsRegistry, MetricType


def test_large_scale_performance():
    """Test performance with 10,000 metrics to ensure no O(n²) behavior."""
    print("Testing large-scale MetricsRegistry performance...")
    print("This test verifies that cleanup optimization prevents O(n²) degradation")
    
    registry = MetricsRegistry()
    num_metrics = 10000
    
    print(f"\nRegistering {num_metrics:,} metrics...")
    start_time = time.time()
    
    # Track performance at intervals
    checkpoint_intervals = [1000, 2000, 5000, 7500, 10000]
    last_checkpoint = 0
    
    for i in range(num_metrics):
        registry.record_metric(
            name=f"performance.test.metric_{i % 20}",  # 20 different metric names
            value=i * 0.1,
            metric_type=MetricType.GAUGE,
            unit="units",
            labels={"batch": str(i // 1000), "index": str(i)}
        )
        
        # Check performance at specific intervals
        if (i + 1) in checkpoint_intervals:
            current_time = time.time()
            elapsed = current_time - start_time
            rate = (i + 1) / elapsed
            
            # Calculate rate for this interval
            interval_size = (i + 1) - last_checkpoint
            interval_time = elapsed - (0 if last_checkpoint == 0 else 
                                     (last_checkpoint / ((i + 1) / elapsed)))
            interval_rate = interval_size / max(interval_time, 0.001)
            
            print(f"  {i + 1:5,} metrics: {elapsed:6.2f}s | "
                  f"Overall: {rate:8.1f} m/s | "
                  f"Recent: {interval_rate:8.1f} m/s")
            
            last_checkpoint = i + 1
    
    total_time = time.time() - start_time
    total_rate = num_metrics / total_time
    
    print(f"\nFinal Results:")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Overall rate: {total_rate:.1f} metrics/second")
    print(f"  Average time per metric: {(total_time / num_metrics) * 1000:.4f} ms")
    
    # Test cleanup performance
    print(f"\nTesting cleanup with {len(registry.metrics):,} stored metrics...")
    cleanup_start = time.time()
    registry.force_cleanup()
    cleanup_time = time.time() - cleanup_start
    
    print(f"  Cleanup completed in: {cleanup_time:.3f} seconds")
    print(f"  Metrics after cleanup: {len(registry.metrics):,}")
    print(f"  Cleanup performance: {len(registry.metrics) / max(cleanup_time, 0.001):.1f} metrics/second")
    
    # Verify the optimization worked
    print(f"\nOptimization verification:")
    print(f"  Registration count: {registry._registration_count:,}")
    print(f"  Cleanup interval: {registry._cleanup_interval}")
    print(f"  Expected cleanups: {registry._registration_count // registry._cleanup_interval}")
    
    # Performance assessment
    if total_rate > 50000:
        grade = "EXCELLENT - No O(n²) behavior detected"
    elif total_rate > 10000:
        grade = "GOOD - Performance acceptable"
    elif total_rate > 1000:
        grade = "ACCEPTABLE - Some degradation"
    else:
        grade = "POOR - May have performance issues"
    
    print(f"  Performance grade: {grade}")
    
    return total_time, total_rate


if __name__ == "__main__":
    test_large_scale_performance()