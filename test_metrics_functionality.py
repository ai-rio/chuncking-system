#!/usr/bin/env python3
"""
Functional test to verify MetricsRegistry still works correctly after optimization.
"""

import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.observability import MetricsRegistry, MetricType


def test_metrics_functionality():
    """Test that all MetricsRegistry functionality still works correctly."""
    print("Testing MetricsRegistry functionality after optimization...")
    
    registry = MetricsRegistry()
    
    # Test 1: Basic metric registration
    print("\n1. Testing basic metric registration...")
    registry.record_metric("test.counter", 10, MetricType.COUNTER, "requests")
    registry.record_metric("test.gauge", 75.5, MetricType.GAUGE, "percent")
    registry.record_metric("test.timer", 150, MetricType.TIMER, "ms")
    
    print(f"   Registered 3 metrics, stored: {len(registry.metrics)}")
    assert len(registry.metrics) == 3, "Should have 3 metrics"
    
    # Test 2: Multiple metrics with same name
    print("\n2. Testing multiple metrics with same name...")
    for i in range(5):
        registry.record_metric("test.series", i * 10, MetricType.COUNTER, "operations")
    
    series_metrics = registry.get_metrics_by_name("test.series")
    print(f"   Registered 5 series metrics, found: {len(series_metrics)}")
    assert len(series_metrics) == 5, "Should have 5 series metrics"
    
    # Test 3: Metric summary
    print("\n3. Testing metric summary...")
    summary = registry.get_metric_summary("test.series")
    print(f"   Summary: count={summary['count']}, min={summary['min']}, max={summary['max']}, avg={summary['avg']:.1f}")
    assert summary['count'] == 5, "Should count 5 metrics"
    assert summary['min'] == 0, "Min should be 0"
    assert summary['max'] == 40, "Max should be 40"
    assert summary['avg'] == 20, "Average should be 20"
    
    # Test 4: Get metrics by type
    print("\n4. Testing get metrics by type...")
    counter_metrics = registry.get_metrics_by_type(MetricType.COUNTER)
    gauge_metrics = registry.get_metrics_by_type(MetricType.GAUGE)
    timer_metrics = registry.get_metrics_by_type(MetricType.TIMER)
    
    print(f"   Counter metrics: {len(counter_metrics)}")
    print(f"   Gauge metrics: {len(gauge_metrics)}")
    print(f"   Timer metrics: {len(timer_metrics)}")
    
    assert len(counter_metrics) == 6, "Should have 6 counter metrics (1 + 5 series)"
    assert len(gauge_metrics) == 1, "Should have 1 gauge metric"
    assert len(timer_metrics) == 1, "Should have 1 timer metric"
    
    # Test 5: Prometheus export
    print("\n5. Testing Prometheus export...")
    prometheus_output = registry.export_prometheus_format()
    print(f"   Prometheus export length: {len(prometheus_output)} characters")
    assert "test.counter" in prometheus_output, "Should contain test.counter"
    assert "test.gauge" in prometheus_output, "Should contain test.gauge"
    
    # Test 6: Periodic cleanup (register 200 metrics to trigger cleanup)
    print("\n6. Testing periodic cleanup...")
    initial_count = len(registry.metrics)
    print(f"   Initial metrics count: {initial_count}")
    
    # Register enough metrics to trigger cleanup
    for i in range(200):
        registry.record_metric(f"cleanup.test.{i % 10}", i, MetricType.COUNTER, "items")
    
    print(f"   After 200 more registrations: {len(registry.metrics)}")
    print(f"   Registration count: {registry._registration_count}")
    
    # Test 7: Force cleanup
    print("\n7. Testing force cleanup...")
    pre_cleanup_count = len(registry.metrics)
    registry.force_cleanup()
    post_cleanup_count = len(registry.metrics)
    
    print(f"   Before cleanup: {pre_cleanup_count}")
    print(f"   After cleanup: {post_cleanup_count}")
    
    # Test 8: Thread safety (basic test)
    print("\n8. Testing thread safety...")
    import threading
    
    def worker():
        for i in range(50):
            registry.record_metric("thread.test", i, MetricType.COUNTER, "ops")
    
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    thread_metrics = registry.get_metrics_by_name("thread.test")
    print(f"   Thread test metrics: {len(thread_metrics)}")
    assert len(thread_metrics) == 200, "Should have 200 thread test metrics (4 threads × 50 each)"
    
    print(f"\n✓ All functionality tests passed!")
    print(f"   Final metrics count: {len(registry.metrics)}")
    print(f"   Unique metric names: {len(registry._metrics_dict)}")
    
    return True


if __name__ == "__main__":
    test_metrics_functionality()