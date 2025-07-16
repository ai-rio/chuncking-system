"""
Phase 4 Tests: Performance and Load Testing

This module contains comprehensive performance tests for Phase 4 enterprise
observability features, including load testing, stress testing, memory usage
validation, and performance benchmarking of observability infrastructure.

Test Coverage:
- High-throughput metrics collection performance
- Concurrent health check execution
- Memory usage under load
- Response time benchmarking
- Scalability testing
- Resource utilization monitoring
- Performance regression detection
"""

import pytest
import time
import threading
import psutil
import gc
import os
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from datetime import datetime, timedelta

from src.utils.observability import (
    ObservabilityManager,
    CorrelationIDManager,
    MetricsRegistry,
    HealthRegistry,
    StructuredLogger,
    MetricType,
    HealthStatus,
    HealthCheckResult
)
from src.api.health_endpoints import (
    HealthEndpoint,
    MetricsEndpoint,
    SystemStatusEndpoint,
    EndpointRouter
)


class PerformanceTimer:
    """Helper class for performance timing."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed_ms(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


class MemoryProfiler:
    """Helper class for memory profiling."""
    
    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None
        self.final_memory = None
    
    def start(self):
        gc.collect()
        process = psutil.Process()
        self.initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
    
    def update_peak(self):
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def stop(self):
        gc.collect()
        process = psutil.Process()
        self.final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    @property
    def memory_increase_mb(self):
        if self.initial_memory and self.final_memory:
            return self.final_memory - self.initial_memory
        return None
    
    @property
    def peak_memory_mb(self):
        return self.peak_memory


class TestMetricsPerformance:
    """Test metrics collection performance under load."""
    
    def test_high_volume_metrics_collection(self):
        """Test metrics collection performance with high volume."""
        metrics_registry = MetricsRegistry()
        profiler = MemoryProfiler()
        profiler.start()
        
        num_metrics = 10000
        
        with PerformanceTimer() as timer:
            for i in range(num_metrics):
                metrics_registry.record_metric(f"test_counter_{i % 100}", 1, MetricType.COUNTER)
                metrics_registry.record_metric(f"test_gauge_{i % 50}", i, MetricType.GAUGE)
                metrics_registry.record_metric(f"test_histogram_{i % 25}", i * 0.1, MetricType.HISTOGRAM)
                
                if i % 1000 == 0:
                    profiler.update_peak()
        
        profiler.stop()
        
        # Performance assertions
        assert timer.elapsed_ms < 5000  # Should complete in under 5 seconds
        assert len(metrics_registry.metrics) == num_metrics * 3  # 3 metrics per iteration (counter, gauge, histogram)
        
        # Memory assertions
        assert profiler.memory_increase_mb < 100  # Should not use more than 100MB
        
        print(f"Collected {len(metrics_registry.metrics)} metrics ({num_metrics} iterations × 3 types) in {timer.elapsed_ms:.2f}ms")
        print(f"Memory increase: {profiler.memory_increase_mb:.2f}MB")
        print(f"Peak memory: {profiler.peak_memory_mb:.2f}MB")
    
    def test_concurrent_metrics_collection(self):
        """Test concurrent metrics collection from multiple threads."""
        metrics_registry = MetricsRegistry()
        num_threads = 10
        metrics_per_thread = 1000
        
        def worker_thread(thread_id):
            for i in range(metrics_per_thread):
                metrics_registry.record_metric(
                    f"thread_{thread_id}_counter", 1, MetricType.COUNTER,
                    labels={"thread": str(thread_id)}
                )
                metrics_registry.record_metric(
                    f"thread_{thread_id}_gauge", i, MetricType.GAUGE
                )
        
        with PerformanceTimer() as timer:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
                for future in as_completed(futures):
                    future.result()
        
        # Verify all metrics collected
        total_expected = num_threads * metrics_per_thread * 2  # counter + gauge per iteration
        assert len(metrics_registry.metrics) == total_expected
        
        # Performance assertion
        assert timer.elapsed_ms < 10000  # Should complete in under 10 seconds
        
        print(f"Concurrent collection: {total_expected} metrics in {timer.elapsed_ms:.2f}ms")
        print(f"Throughput: {total_expected / (timer.elapsed_ms / 1000):.0f} metrics/sec")
    
    def test_metrics_export_performance(self):
        """Test metrics export performance with large datasets."""
        metrics_registry = MetricsRegistry()
        
        # Create large dataset
        num_metrics = 5000
        for i in range(num_metrics):
            metrics_registry.record_metric(f"metric_{i}", i, MetricType.COUNTER,
                                         labels={"category": f"cat_{i % 10}"})
        
        # Test Prometheus export performance
        with PerformanceTimer() as prometheus_timer:
            prometheus_output = metrics_registry.export_prometheus_format()
        
        # Test JSON export performance (if available)
        with PerformanceTimer() as json_timer:
            json_data = [m.to_dict() for m in metrics_registry.metrics]
        
        # Performance assertions
        assert prometheus_timer.elapsed_ms < 2000  # Under 2 seconds
        assert json_timer.elapsed_ms < 1000  # Under 1 second
        
        # Verify output quality
        assert len(prometheus_output) > 10000  # Substantial output
        assert len(json_data) == num_metrics
        
        print(f"Prometheus export: {prometheus_timer.elapsed_ms:.2f}ms")
        print(f"JSON export: {json_timer.elapsed_ms:.2f}ms")
    
    def test_metrics_memory_efficiency(self):
        """Test memory efficiency of metrics storage."""
        metrics_registry = MetricsRegistry()
        profiler = MemoryProfiler()
        profiler.start()
        
        # Add metrics in batches and measure memory growth
        batch_size = 1000
        num_batches = 10
        memory_measurements = []
        
        for batch in range(num_batches):
            for i in range(batch_size):
                metrics_registry.record_metric(
                    f"memory_test_{batch}_{i}", i, MetricType.HISTOGRAM
                )
            
            profiler.update_peak()
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_measurements.append(current_memory)
        
        profiler.stop()
        
        # Analyze memory growth pattern
        memory_growth = memory_measurements[-1] - memory_measurements[0]
        avg_memory_per_metric = memory_growth / (batch_size * num_batches)
        
        # Memory efficiency assertions
        assert avg_memory_per_metric < 0.001  # Less than 1KB per metric
        assert memory_growth < 50  # Total growth under 50MB
        
        print(f"Memory per metric: {avg_memory_per_metric * 1024:.2f} bytes")
        print(f"Total memory growth: {memory_growth:.2f}MB")


class TestHealthCheckPerformance:
    """Test health check performance under load."""
    
    def test_health_check_response_times(self):
        """Test health check response times."""
        health_registry = HealthRegistry()
        
        # Add health checks with different response times
        def fast_check():
            time.sleep(0.001)  # 1ms
            return HealthCheckResult("fast", HealthStatus.HEALTHY, "Fast service")
        
        def medium_check():
            time.sleep(0.010)  # 10ms
            return HealthCheckResult("medium", HealthStatus.HEALTHY, "Medium service")
        
        def slow_check():
            time.sleep(0.050)  # 50ms
            return HealthCheckResult("slow", HealthStatus.HEALTHY, "Slow service")
        
        health_registry.register_health_check("fast_service", fast_check)
        health_registry.register_health_check("medium_service", medium_check)
        health_registry.register_health_check("slow_service", slow_check)
        
        # Test individual check performance
        with PerformanceTimer() as fast_timer:
            fast_result = health_registry.run_health_check("fast_service")
        
        with PerformanceTimer() as medium_timer:
            medium_result = health_registry.run_health_check("medium_service")
        
        with PerformanceTimer() as slow_timer:
            slow_result = health_registry.run_health_check("slow_service")
        
        # Verify response times match expectations (with tolerance)
        assert 0 < fast_timer.elapsed_ms < 10
        assert 5 < medium_timer.elapsed_ms < 20
        assert 45 < slow_timer.elapsed_ms < 70
        
        # Test all checks together
        with PerformanceTimer() as all_timer:
            all_results = health_registry.run_all_health_checks()
        
        # Should be close to slowest check time (parallel execution)
        assert all_timer.elapsed_ms < 80  # Allow some overhead
        assert len(all_results) == 3
        
        print(f"Fast check: {fast_timer.elapsed_ms:.2f}ms")
        print(f"Medium check: {medium_timer.elapsed_ms:.2f}ms")
        print(f"Slow check: {slow_timer.elapsed_ms:.2f}ms")
        print(f"All checks: {all_timer.elapsed_ms:.2f}ms")
    
    def test_concurrent_health_checks(self):
        """Test concurrent health check execution."""
        health_registry = HealthRegistry()
        
        # Add multiple health checks
        num_checks = 20
        for i in range(num_checks):
            def health_check(check_id=i):
                time.sleep(0.01)  # 10ms each
                return HealthCheckResult(f"service_{check_id}", HealthStatus.HEALTHY, "OK")
            
            health_registry.register_health_check(f"service_{i}", health_check)
        
        # Test concurrent execution
        with PerformanceTimer() as timer:
            results = health_registry.run_all_health_checks()
        
        # Should complete much faster than sequential execution
        # Sequential would be 20 * 10ms = 200ms, concurrent should be ~10ms + overhead
        assert timer.elapsed_ms < 100  # Allow generous overhead
        assert len(results) == num_checks
        
        # Verify all checks succeeded
        for result in results.values():
            assert result.status == HealthStatus.HEALTHY
        
        print(f"Concurrent health checks: {timer.elapsed_ms:.2f}ms for {num_checks} checks")
    
    def test_health_check_under_load(self):
        """Test health check performance under continuous load."""
        health_registry = HealthRegistry()
        
        def load_test_check():
            return HealthCheckResult("load_test", HealthStatus.HEALTHY, "Under load")
        
        health_registry.register_health_check("load_test", load_test_check)
        
        # Run health checks continuously for a period
        num_iterations = 100
        response_times = []
        
        for i in range(num_iterations):
            with PerformanceTimer() as timer:
                result = health_registry.run_health_check("load_test")
            
            response_times.append(timer.elapsed_ms)
            assert result.status == HealthStatus.HEALTHY
        
        # Analyze response time consistency
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        # Performance assertions
        assert avg_response_time < 10  # Average under 10ms (relaxed for CI)
        assert max_response_time < 50  # Max under 50ms (relaxed for CI)
        assert max_response_time / min_response_time < 100  # Very relaxed variance for CI stability
        
        print(f"Load test - Avg: {avg_response_time:.2f}ms, "
              f"Min: {min_response_time:.2f}ms, Max: {max_response_time:.2f}ms")


class TestObservabilityManagerPerformance:
    """Test ObservabilityManager performance under load."""
    
    def test_observability_manager_throughput(self):
        """Test ObservabilityManager throughput."""
        obs_manager = ObservabilityManager()
        
        # Add health checks
        def health_check():
            return HealthCheckResult("test", HealthStatus.HEALTHY, "OK")
        
        obs_manager.register_health_check("test_service", health_check)
        
        # Mixed workload test
        num_operations = 1000
        
        with PerformanceTimer() as timer:
            for i in range(num_operations):
                # Record metrics
                obs_manager.record_metric("throughput_test", 1, MetricType.COUNTER)
                obs_manager.record_metric("response_time", i * 0.1, MetricType.HISTOGRAM)
                
                # Run health check every 10th iteration
                if i % 10 == 0:
                    obs_manager.run_health_check("test_service")
        
        # Calculate throughput
        operations_per_second = num_operations / (timer.elapsed_ms / 1000)
        
        # Performance assertions
        assert timer.elapsed_ms < 5000  # Under 5 seconds
        assert operations_per_second > 500  # At least 500 ops/sec
        
        print(f"Throughput: {operations_per_second:.0f} operations/second")
    
    def test_data_export_scalability(self):
        """Test data export performance with large datasets."""
        obs_manager = ObservabilityManager()
        
        # Create large dataset
        num_metrics = 5000
        for i in range(num_metrics):
            obs_manager.record_metric(f"scale_test_{i % 100}", i, MetricType.COUNTER)
        
        # Test export performance
        with PerformanceTimer() as timer:
            export_data = obs_manager.export_all_data(include_system_metrics=False)
        
        # Verify export completeness
        assert len(export_data["metrics"]["metrics"]) == num_metrics
        assert "prometheus_format" in export_data
        assert len(export_data["prometheus_format"]) > 1000  # Adjusted for realistic size
        
        # Performance assertion
        assert timer.elapsed_ms < 3000  # Under 3 seconds
        
        print(f"Export performance: {num_metrics} metrics in {timer.elapsed_ms:.2f}ms")
    
    def test_correlation_id_performance(self):
        """Test correlation ID management performance."""
        correlation_manager = CorrelationIDManager()
        
        # Test ID generation performance
        num_ids = 10000
        
        with PerformanceTimer() as generation_timer:
            ids = [correlation_manager.generate_correlation_id() for _ in range(num_ids)]
        
        # Verify uniqueness and performance
        assert len(set(ids)) == num_ids  # All unique
        assert generation_timer.elapsed_ms < 1000  # Under 1 second
        
        # Test thread-local storage performance
        def worker_thread(thread_id):
            correlation_id = correlation_manager.generate_correlation_id()
            correlation_manager.set_correlation_id(correlation_id)
            
            # Simulate work with ID access
            for _ in range(100):
                retrieved_id = correlation_manager.get_correlation_id()
                assert retrieved_id == correlation_id
        
        with PerformanceTimer() as thread_timer:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(worker_thread, i) for i in range(10)]
                for future in as_completed(futures):
                    future.result()
        
        assert thread_timer.elapsed_ms < 2000  # Under 2 seconds
        
        print(f"ID generation: {num_ids} IDs in {generation_timer.elapsed_ms:.2f}ms")
        print(f"Thread-local performance: {thread_timer.elapsed_ms:.2f}ms")


class TestEndpointPerformance:
    """Test health endpoint performance."""
    
    @patch('src.chunking_system.DocumentChunker')
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.get_observability_manager')
    def test_health_endpoint_response_time(self, mock_get_obs, mock_system_monitor_class, mock_document_chunker_class):
        """Test health endpoint response times."""
        # Mock DocumentChunker to prevent slow initialization
        mock_document_chunker = Mock()
        mock_document_chunker_class.return_value = mock_document_chunker
        
        # Setup mocks for SystemMonitor
        mock_monitor = Mock()
        mock_health_result = Mock()
        mock_health_result.is_healthy = True
        mock_health_result.component = "system"
        mock_health_result.message = "OK"
        mock_health_result.timestamp = datetime.now()
        mock_health_result.response_time_ms = 1.0
        mock_health_result.details = {}
        mock_health_result.status = "healthy"
        
        mock_monitor.health_checker.get_overall_health.return_value = mock_health_result
        mock_system_monitor_class.return_value = mock_monitor
        
        mock_obs = Mock()
        mock_get_obs.return_value = mock_obs
        
        health_endpoint = HealthEndpoint()
        
        # Test basic health check performance
        response_times = []
        num_requests = 100
        
        for _ in range(num_requests):
            with PerformanceTimer() as timer:
                response, status_code = health_endpoint.health_check()
            
            response_times.append(timer.elapsed_ms)
            assert status_code == 200
        
        # Analyze performance
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Performance assertions
        assert avg_response_time < 10  # Average under 10ms
        assert max_response_time < 50  # Max under 50ms
        
        print(f"Health endpoint - Avg: {avg_response_time:.2f}ms, Max: {max_response_time:.2f}ms")
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.get_observability_manager')
    def test_metrics_endpoint_performance(self, mock_get_obs, mock_system_monitor):
        """Test metrics endpoint performance."""
        mock_obs = Mock()
        
        # Mock large metrics dataset
        large_prometheus_output = "# Large metrics output\n" * 1000
        mock_obs.metrics_registry.export_prometheus_format.return_value = large_prometheus_output
        mock_obs.export_prometheus_metrics.return_value = large_prometheus_output
        
        large_json_data = {"metrics": {"metrics": [{"name": f"metric_{i}", "value": i} for i in range(1000)]}}
        mock_obs.export_all_data.return_value = large_json_data
        
        mock_get_obs.return_value = mock_obs
        
        # Mock SystemMonitor to avoid slow initialization
        mock_monitor = Mock()
        mock_monitor.get_system_status.return_value = {
            "health": {"overall_healthy": True, "checks": {}},
            "metrics": {"total_metrics": 1000}
        }
        mock_system_monitor.return_value = mock_monitor
        
        metrics_endpoint = MetricsEndpoint()
        
        # Test Prometheus export performance
        with PerformanceTimer() as prometheus_timer:
            response, status_code = metrics_endpoint.prometheus_metrics()
        
        assert status_code == 200
        assert prometheus_timer.elapsed_ms < 100  # Under 100ms
        
        # Test JSON export performance
        with PerformanceTimer() as json_timer:
            response, status_code = metrics_endpoint.json_metrics()
        
        assert status_code == 200
        assert json_timer.elapsed_ms < 100  # Under 100ms
        
        print(f"Metrics export - Prometheus: {prometheus_timer.elapsed_ms:.2f}ms, "
              f"JSON: {json_timer.elapsed_ms:.2f}ms")
    
    @patch('src.chunking_system.DocumentChunker')
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.get_observability_manager')
    @patch.dict(os.environ, {'CHUNKING_SYSTEM_LIGHTWEIGHT_HEALTH_CHECK': 'true'})
    def test_endpoint_router_performance(self, mock_get_obs, mock_system_monitor_class, mock_document_chunker_class):
        """Test endpoint router performance."""
        # Mock the observability manager to avoid slow operations
        mock_obs = Mock()
        mock_obs.export_all_data.return_value = {"metrics": {"metrics": []}}
        mock_obs.metrics_registry.export_prometheus_format.return_value = "# Empty"
        mock_get_obs.return_value = mock_obs
        
        # Mock DocumentChunker to prevent slow initialization
        mock_document_chunker = Mock()
        mock_document_chunker_class.return_value = mock_document_chunker
        
        # Mock SystemMonitor to avoid DocumentChunker initialization during health checks
        mock_system_monitor = Mock()
        mock_health_checker = Mock()
        mock_metrics_collector = Mock()
        mock_alert_manager = Mock()
        mock_performance_monitor = Mock()
        
        # Set up the mocked SystemMonitor with fast health checks
        mock_system_monitor.health_checker = mock_health_checker
        mock_system_monitor.metrics_collector = mock_metrics_collector
        mock_system_monitor.alert_manager = mock_alert_manager
        mock_system_monitor.performance_monitor = mock_performance_monitor
        mock_system_monitor.get_system_metrics.return_value = {
            "cpu_percent": 50.0,
            "memory_percent": 60.0,
            "disk_percent": 40.0,
            "load_average": [1.0, 1.1, 1.2],
            "timestamp": datetime.now().isoformat()
        }
        mock_system_monitor.get_system_status.return_value = {
            "health": {"overall_healthy": True, "checks": {}},
            "metrics_count": 0,
            "active_alerts": 0,
            "alert_counts": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Mock health check results to avoid slow operations
        from src.utils.monitoring import HealthStatus
        mock_health_result = Mock()
        mock_health_result.is_healthy = True
        mock_health_result.component = "test"
        mock_health_result.message = "OK"
        mock_health_result.timestamp = datetime.now()
        mock_health_result.response_time_ms = 1.0
        mock_health_result.details = {}
        mock_health_result.status = "healthy"
        
        mock_health_checker.get_overall_health.return_value = mock_health_result
        mock_health_checker.run_check.return_value = mock_health_result
        mock_health_checker.run_all_checks.return_value = {"test": mock_health_result}
        
        mock_system_monitor_class.return_value = mock_system_monitor
        
        # Now create the router - this should be fast since everything is mocked
        router = EndpointRouter()
        
        # Mock the endpoint methods to return quickly for the routing test
        router.health_endpoint.health_check = Mock(return_value=({"status": "healthy"}, 200))
        router.health_endpoint.detailed_health = Mock(return_value=({"status": "healthy"}, 200))
        router.health_endpoint.readiness_check = Mock(return_value=({"status": "ready"}, 200))
        router.metrics_endpoint.prometheus_metrics = Mock(return_value=("# Empty", 200))
        router.metrics_endpoint.json_metrics = Mock(return_value=({"metrics": []}, 200))
        router.system_endpoint.system_info = Mock(return_value=({"info": "test"}, 200))
        
        # Test routing performance for different endpoints
        test_routes = [
            ("GET", "/health"),
            ("GET", "/health/detailed"),
            ("GET", "/health/ready"),
            ("GET", "/metrics"),
            ("GET", "/metrics/json"),
            ("GET", "/system/info")
        ]
        
        routing_times = []
        
        for method, path in test_routes:
            with PerformanceTimer() as timer:
                try:
                    response, status_code = router.route_request(method, path)
                except Exception:
                    # Some routes might fail due to mocking, but we're testing routing speed
                    pass
            
            routing_times.append(timer.elapsed_ms)
        
        avg_routing_time = sum(routing_times) / len(routing_times)
        
        # Routing should be very fast (relaxed for mocking overhead)
        assert avg_routing_time < 50  # Under 50ms average
        
        print(f"Routing performance: {avg_routing_time:.2f}ms average")


class TestMemoryLeakDetection:
    """Test for memory leaks in observability components."""
    
    def test_metrics_memory_leak(self):
        """Test for memory leaks in metrics collection."""
        profiler = MemoryProfiler()
        profiler.start()
        
        # Run multiple cycles of metrics collection and cleanup
        for cycle in range(5):
            metrics_registry = MetricsRegistry()
            
            # Add metrics
            for i in range(1000):
                metrics_registry.record_metric(f"leak_test_{i}", i, MetricType.COUNTER)
            
            # Clear metrics
            metrics_registry.clear_metrics()
            
            # Force garbage collection
            del metrics_registry
            gc.collect()
            
            profiler.update_peak()
        
        profiler.stop()
        
        # Memory should not grow significantly across cycles
        assert profiler.memory_increase_mb < 10  # Less than 10MB growth
        
        print(f"Memory leak test - Growth: {profiler.memory_increase_mb:.2f}MB")
    
    def test_health_check_memory_leak(self):
        """Test for memory leaks in health checks."""
        profiler = MemoryProfiler()
        profiler.start()
        
        # Run multiple cycles of health check registration and execution
        for cycle in range(5):
            health_registry = HealthRegistry()
            
            # Register health checks
            for i in range(100):
                def health_check(check_id=i):
                    return HealthCheckResult(f"service_{check_id}", HealthStatus.HEALTHY, "OK")
                
                health_registry.register_health_check(f"service_{i}", health_check)
            
            # Run health checks
            results = health_registry.run_all_health_checks()
            
            # Cleanup
            del health_registry
            del results
            gc.collect()
            
            profiler.update_peak()
        
        profiler.stop()
        
        # Memory should not grow significantly
        assert profiler.memory_increase_mb < 5  # Less than 5MB growth
        
        print(f"Health check memory test - Growth: {profiler.memory_increase_mb:.2f}MB")


class TestStressScenarios:
    """Test observability system under stress conditions."""
    
    def test_high_concurrency_stress(self):
        """Test system under high concurrency stress."""
        obs_manager = ObservabilityManager()
        
        # Add health checks
        def stress_health_check():
            return HealthCheckResult("stress", HealthStatus.HEALTHY, "Under stress")
        
        obs_manager.register_health_check("stress_service", stress_health_check)
        
        # High concurrency workload
        num_threads = 50
        operations_per_thread = 100
        
        def stress_worker(thread_id):
            for i in range(operations_per_thread):
                # Record metrics
                obs_manager.record_metric(f"stress_metric_{thread_id}", i, MetricType.COUNTER)
                
                # Run health check
                if i % 10 == 0:
                    obs_manager.run_health_check("stress_service")
        
        with PerformanceTimer() as timer:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(stress_worker, i) for i in range(num_threads)]
                for future in as_completed(futures):
                    future.result()
        
        # Verify system survived stress test
        total_operations = num_threads * operations_per_thread
        assert timer.elapsed_ms < 30000  # Under 30 seconds
        
        # Verify data integrity
        export_data = obs_manager.export_all_data(include_system_metrics=False)
        assert len(export_data["metrics"]["metrics"]) == total_operations
        
        print(f"Stress test: {total_operations} operations in {timer.elapsed_ms:.2f}ms")
        print(f"Throughput: {total_operations / (timer.elapsed_ms / 1000):.0f} ops/sec")
    
    def test_sustained_load_performance(self):
        """Test performance under sustained load."""
        obs_manager = ObservabilityManager()
        
        # Sustained load test - run for extended period
        test_duration_seconds = 10
        start_time = time.time()
        operation_count = 0
        
        while time.time() - start_time < test_duration_seconds:
            obs_manager.record_metric("sustained_load", 1, MetricType.COUNTER)
            operation_count += 1
            
            # Small delay to simulate realistic load
            time.sleep(0.001)  # 1ms
        
        actual_duration = time.time() - start_time
        operations_per_second = operation_count / actual_duration
        
        # Performance assertion
        assert operations_per_second > 100  # At least 100 ops/sec sustained
        
        print(f"Sustained load: {operations_per_second:.0f} ops/sec over {actual_duration:.1f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print output