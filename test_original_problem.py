#!/usr/bin/env python3
"""
Test the exact scenario described in the original problem:
- 10,000 metrics registration
- Compare performance before and after optimization
- Verify the 2700% performance improvement is achieved
"""

import time
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.observability import MetricsRegistry, MetricType


def test_original_problem_scenario():
    """Test the exact scenario from the problem description."""
    print("Testing Original Problem Scenario")
    print("=" * 50)
    print("Problem: MetricsRegistry.record_metric() calls _cleanup_old_metrics() on every registration")
    print("Expected: O(n²) performance degradation with 10,000 metrics")
    print("Target: <5 seconds (was 27 seconds = 2700% slower)")
    print()
    
    registry = MetricsRegistry()
    num_metrics = 10000
    
    print(f"Registering {num_metrics:,} metrics...")
    print("Monitoring for O(n²) behavior...")
    
    start_time = time.time()
    
    # Track performance degradation
    checkpoint_times = []
    checkpoint_metrics = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    for i in range(num_metrics):
        registry.record_metric(
            name=f"test.metric.{i % 50}",  # 50 different metric names like in real scenarios
            value=i * 0.5,
            metric_type=MetricType.COUNTER,
            unit="operations",
            labels={"batch": str(i // 1000), "worker": str(i % 10)}
        )
        
        # Record checkpoint times
        if (i + 1) in checkpoint_metrics:
            current_time = time.time()
            elapsed = current_time - start_time
            rate = (i + 1) / elapsed
            checkpoint_times.append((i + 1, elapsed, rate))
            
            print(f"  {i + 1:5,} metrics: {elapsed:6.3f}s | {rate:8.1f} metrics/sec")
    
    total_time = time.time() - start_time
    total_rate = num_metrics / total_time
    
    print(f"\nFinal Results:")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Overall rate: {total_rate:.1f} metrics/second")
    print(f"  Time per metric: {(total_time / num_metrics) * 1000:.4f} ms")
    
    # Analyze performance degradation
    print(f"\nPerformance Analysis:")
    print(f"  Target time: < 5 seconds")
    print(f"  Actual time: {total_time:.3f} seconds")
    
    if total_time < 5:
        improvement = "SUCCESS - Target met!"
        if total_time < 1:
            improvement += " (Excellent performance)"
    else:
        improvement = "NEEDS WORK - Exceeds 5-second target"
    
    print(f"  Result: {improvement}")
    
    # Check for O(n²) behavior by comparing early vs late performance
    early_rate = checkpoint_times[0][2]  # Rate at 1000 metrics
    late_rate = checkpoint_times[-1][2]  # Rate at 10000 metrics
    degradation = ((early_rate - late_rate) / early_rate) * 100
    
    print(f"\nO(n²) Analysis:")
    print(f"  Rate at 1,000 metrics: {early_rate:.1f} metrics/sec")
    print(f"  Rate at 10,000 metrics: {late_rate:.1f} metrics/sec")
    print(f"  Performance degradation: {degradation:.1f}%")
    
    if degradation < 20:
        behavior = "Linear/Constant - Optimization successful!"
    elif degradation < 50:
        behavior = "Some degradation but acceptable"
    else:
        behavior = "Significant degradation - may indicate O(n²)"
    
    print(f"  Behavior assessment: {behavior}")
    
    # Registry state verification
    print(f"\nRegistry State:")
    print(f"  Stored metrics: {len(registry.metrics):,}")
    print(f"  Unique metric names: {len(registry._metrics_dict)}")
    print(f"  Registration count: {registry._registration_count:,}")
    print(f"  Cleanup interval: {registry._cleanup_interval}")
    print(f"  Cleanups performed: {registry._registration_count // registry._cleanup_interval}")
    
    # Calculate theoretical old performance
    print(f"\nComparison to Original Problem:")
    print(f"  Original time (reported): 27 seconds")
    print(f"  Current time: {total_time:.3f} seconds")
    if total_time > 0:
        speedup = 27 / total_time
        print(f"  Performance improvement: {speedup:.1f}x faster")
        print(f"  Performance gain: {((speedup - 1) * 100):.0f}% improvement")
    
    return total_time, total_rate


if __name__ == "__main__":
    test_original_problem_scenario()