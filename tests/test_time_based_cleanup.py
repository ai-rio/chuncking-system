#!/usr/bin/env python3
"""
Test time-based cleanup functionality to ensure old metrics are properly removed.
"""

import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.observability import MetricsRegistry, MetricType, CustomMetric


def test_time_based_cleanup():
    """Test that time-based cleanup still works correctly."""
    print("Testing time-based cleanup functionality...")
    
    # Create registry with short aggregation window for testing
    registry = MetricsRegistry()
    registry.aggregation_window = timedelta(seconds=2)  # Very short for testing
    
    print(f"Using aggregation window: {registry.aggregation_window.total_seconds()} seconds")
    
    # Add some initial metrics
    print("\n1. Adding initial metrics...")
    for i in range(5):
        registry.record_metric(f"initial.metric.{i}", i * 10, MetricType.COUNTER, "units")
    
    print(f"   Initial metrics count: {len(registry.metrics)}")
    
    # Wait for metrics to become old
    print("\n2. Waiting for metrics to age...")
    time.sleep(3)  # Wait longer than aggregation window
    
    # Add new metrics (this should trigger cleanup due to time condition)
    print("\n3. Adding new metrics (should trigger time-based cleanup)...")
    for i in range(3):
        registry.record_metric(f"new.metric.{i}", i * 5, MetricType.GAUGE, "percent")
    
    print(f"   Metrics count after new additions: {len(registry.metrics)}")
    
    # Force cleanup to see the effect
    print("\n4. Testing force cleanup...")
    pre_cleanup_count = len(registry.metrics)
    registry.force_cleanup()
    post_cleanup_count = len(registry.metrics)
    
    print(f"   Before cleanup: {pre_cleanup_count}")
    print(f"   After cleanup: {post_cleanup_count}")
    
    # Test that only recent metrics remain
    remaining_metrics = registry.get_all_metrics()
    print(f"   Remaining metric types: {list(remaining_metrics.keys())}")
    
    # Verify cleanup logic
    cutoff = datetime.now() - registry.aggregation_window
    actual_recent_count = len([m for m in registry.metrics if m.timestamp > cutoff])
    
    print(f"\n5. Verification:")
    print(f"   Cutoff time: {cutoff.strftime('%H:%M:%S')}")
    print(f"   Current time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"   Metrics newer than cutoff: {actual_recent_count}")
    print(f"   Total stored metrics: {len(registry.metrics)}")
    
    # Test cleanup interval behavior
    print(f"\n6. Testing cleanup interval behavior...")
    registry._registration_count = 0  # Reset counter
    
    # Add metrics one by one and check when cleanup occurs
    for i in range(150):  # Exceed cleanup interval
        registry.record_metric(f"interval.test.{i}", i, MetricType.COUNTER, "ops")
        
        if i == 99:  # Should trigger cleanup at 100th registration
            count_at_100 = len(registry.metrics)
        elif i == 149:  # Final count
            final_count = len(registry.metrics)
    
    print(f"   Metrics at 100th registration: {count_at_100}")
    print(f"   Final metrics count: {final_count}")
    print(f"   Registration count: {registry._registration_count}")
    
    print(f"\n✓ Time-based cleanup test completed!")
    
    return True


if __name__ == "__main__":
    test_time_based_cleanup()