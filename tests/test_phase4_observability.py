"""
Phase 4 Tests: Enterprise Observability Infrastructure

This module contains comprehensive tests for the Phase 4 enterprise observability
features including distributed tracing, structured logging, metrics collection,
and health monitoring.

Test Coverage:
- TraceContext and correlation ID management
- StructuredLogger with JSON formatting
- MetricsRegistry with Prometheus export
- HealthRegistry with dependency tracking
- DashboardGenerator configuration
- ObservabilityManager integration
"""

import pytest
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.utils.observability import (
    TraceContext,
    CustomMetric,
    HealthCheckResult,
    CorrelationIDManager,
    StructuredLogger,
    MetricsRegistry,
    HealthRegistry,
    DashboardGenerator,
    ObservabilityManager,
    MetricType,
    HealthStatus
)


class TestTraceContext:
    """Test TraceContext for distributed tracing."""
    
    def test_trace_context_creation(self):
        """Test TraceContext initialization."""
        operation = "test_operation"
        trace_id = "trace123"
        span_id = "span456"
        
        context = TraceContext(operation, trace_id, span_id)
        
        assert context.operation == operation
        assert context.trace_id == trace_id
        assert context.span_id == span_id
        assert isinstance(context.start_time, datetime)
        assert context.parent_span_id is None
        assert context.tags == {}
        assert context.logs == []
    
    def test_trace_context_with_parent(self):
        """Test TraceContext with parent span."""
        context = TraceContext(
            "child_operation",
            "trace123",
            "span456",
            parent_span_id="parent789"
        )
        
        assert context.parent_span_id == "parent789"
    
    def test_add_tag(self):
        """Test adding tags to trace context."""
        context = TraceContext("test", "trace", "span")
        
        context.add_tag("user_id", "user123")
        context.add_tag("environment", "production")
        
        assert context.tags["user_id"] == "user123"
        assert context.tags["environment"] == "production"
    
    def test_add_log(self):
        """Test adding logs to trace context."""
        context = TraceContext("test", "trace", "span")
        
        context.add_log("info", "Operation started")
        context.add_log("error", "Something went wrong", {"error_code": 500})
        
        assert len(context.logs) == 2
        assert context.logs[0]["level"] == "info"
        assert context.logs[0]["message"] == "Operation started"
        assert context.logs[1]["level"] == "error"
        assert context.logs[1]["data"]["error_code"] == 500
    
    def test_finish_trace(self):
        """Test finishing a trace context."""
        context = TraceContext("test", "trace", "span")
        time.sleep(0.01)  # Small delay to ensure duration > 0
        
        context.finish()
        
        assert context.end_time is not None
        assert context.duration_ms > 0
    
    def test_to_dict(self):
        """Test converting trace context to dictionary."""
        context = TraceContext("test_op", "trace123", "span456")
        context.add_tag("key", "value")
        context.add_log("info", "test message")
        context.finish()
        
        trace_dict = context.to_dict()
        
        assert trace_dict["operation"] == "test_op"
        assert trace_dict["trace_id"] == "trace123"
        assert trace_dict["span_id"] == "span456"
        assert "start_time" in trace_dict
        assert "end_time" in trace_dict
        assert "duration_ms" in trace_dict
        assert trace_dict["tags"]["key"] == "value"
        assert len(trace_dict["logs"]) == 1


class TestCustomMetric:
    """Test CustomMetric data structure."""
    
    def test_custom_metric_creation(self):
        """Test CustomMetric initialization."""
        metric = CustomMetric(
            name="test_metric",
            value=42.5,
            metric_type=MetricType.GAUGE,
            unit="bytes",
            labels={"component": "test"}
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.metric_type == MetricType.GAUGE
        assert metric.unit == "bytes"
        assert metric.labels["component"] == "test"
        assert isinstance(metric.timestamp, datetime)
    
    def test_custom_metric_to_dict(self):
        """Test converting metric to dictionary."""
        metric = CustomMetric("cpu_usage", 75.0, MetricType.GAUGE, "percent")
        metric_dict = metric.to_dict()
        
        assert metric_dict["name"] == "cpu_usage"
        assert metric_dict["value"] == 75.0
        assert metric_dict["type"] == "gauge"
        assert metric_dict["unit"] == "percent"
        assert "timestamp" in metric_dict


class TestHealthCheckResult:
    """Test HealthCheckResult data structure."""
    
    def test_health_check_result_creation(self):
        """Test HealthCheckResult initialization."""
        result = HealthCheckResult(
            component="database",
            status=HealthStatus.HEALTHY,
            message="Database connection successful",
            response_time_ms=50.0,
            details={"connection_pool": "active"}
        )
        
        assert result.component == "database"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Database connection successful"
        assert result.response_time_ms == 50.0
        assert result.details["connection_pool"] == "active"
        assert isinstance(result.timestamp, datetime)
    
    def test_health_check_result_to_dict(self):
        """Test converting health result to dictionary."""
        result = HealthCheckResult("cache", HealthStatus.DEGRADED, "Cache slow")
        result_dict = result.to_dict()
        
        assert result_dict["component"] == "cache"
        assert result_dict["status"] == "degraded"
        assert result_dict["message"] == "Cache slow"
        assert "timestamp" in result_dict


class TestCorrelationIDManager:
    """Test CorrelationIDManager for request tracking."""
    
    def test_correlation_id_generation(self):
        """Test correlation ID generation."""
        manager = CorrelationIDManager()
        
        correlation_id = manager.generate_correlation_id()
        
        assert len(correlation_id) == 32  # UUID without hyphens
        assert correlation_id.isalnum()
    
    def test_correlation_id_uniqueness(self):
        """Test that correlation IDs are unique."""
        manager = CorrelationIDManager()
        
        ids = [manager.generate_correlation_id() for _ in range(100)]
        
        assert len(set(ids)) == 100  # All unique
    
    def test_set_get_correlation_id(self):
        """Test setting and getting correlation ID for current thread."""
        manager = CorrelationIDManager()
        correlation_id = "test123"
        
        manager.set_correlation_id(correlation_id)
        retrieved_id = manager.get_correlation_id()
        
        assert retrieved_id == correlation_id
    
    def test_clear_correlation_id(self):
        """Test clearing correlation ID."""
        manager = CorrelationIDManager()
        
        manager.set_correlation_id("test123")
        assert manager.get_correlation_id() == "test123"
        
        manager.clear_correlation_id()
        assert manager.get_correlation_id() is None
    
    def test_thread_safety(self):
        """Test thread safety of correlation ID manager."""
        manager = CorrelationIDManager()
        results = {}
        
        def worker(thread_id):
            correlation_id = f"thread_{thread_id}"
            manager.set_correlation_id(correlation_id)
            time.sleep(0.01)  # Small delay
            results[thread_id] = manager.get_correlation_id()
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Each thread should have its own correlation ID
        for i in range(10):
            assert results[i] == f"thread_{i}"


class TestStructuredLogger:
    """Test StructuredLogger for JSON logging."""
    
    @patch('src.utils.observability.logging.getLogger')
    def test_structured_logger_creation(self, mock_get_logger):
        """Test StructuredLogger initialization."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = StructuredLogger("test_component")
        
        assert logger.component == "test_component"
        mock_get_logger.assert_called_once_with("test_component")
    
    @patch('src.utils.observability.logging.getLogger')
    def test_info_logging(self, mock_get_logger):
        """Test info level logging."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = StructuredLogger("test")
        logger.info("Test message", extra_field="value")
        
        # Verify log was called with structured format
        mock_logger.info.assert_called_once()
        args = mock_logger.info.call_args[0][0]
        log_data = json.loads(args)
        
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "Test message"
        assert log_data["component"] == "test"
        assert log_data["extra_field"] == "value"
        assert "timestamp" in log_data
    
    @patch('src.utils.observability.logging.getLogger')
    @patch('src.utils.observability.CorrelationIDManager')
    def test_correlation_id_inclusion(self, mock_manager_class, mock_get_logger):
        """Test correlation ID inclusion in logs."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        mock_manager = Mock()
        mock_manager.get_correlation_id.return_value = "corr123"
        mock_manager_class.return_value = mock_manager
        
        logger = StructuredLogger("test")
        logger.info("Test message")
        
        args = mock_logger.info.call_args[0][0]
        log_data = json.loads(args)
        
        assert log_data["correlation_id"] == "corr123"
    
    @patch('src.utils.observability.logging.getLogger')
    def test_error_logging_with_exception(self, mock_get_logger):
        """Test error logging with exception details."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = StructuredLogger("test")
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            logger.error("Error occurred", exception=e)
        
        args = mock_logger.error.call_args[0][0]
        log_data = json.loads(args)
        
        assert log_data["level"] == "ERROR"
        assert log_data["message"] == "Error occurred"
        assert "exception" in log_data


class TestMetricsRegistry:
    """Test MetricsRegistry for metrics collection."""
    
    def test_metrics_registry_creation(self):
        """Test MetricsRegistry initialization."""
        registry = MetricsRegistry()
        
        assert len(registry.metrics) == 0
        assert isinstance(registry.metrics, list)
    
    def test_record_counter_metric(self):
        """Test recording counter metrics."""
        registry = MetricsRegistry()
        
        registry.record_metric("requests_total", 1, MetricType.COUNTER)
        registry.record_metric("requests_total", 2, MetricType.COUNTER)
        
        # Should have aggregated counter values
        counter_metrics = [m for m in registry.metrics if m.metric_type == MetricType.COUNTER]
        total_value = sum(m.value for m in counter_metrics if m.name == "requests_total")
        assert total_value == 3
    
    def test_record_gauge_metric(self):
        """Test recording gauge metrics."""
        registry = MetricsRegistry()
        
        registry.record_metric("cpu_usage", 50.0, MetricType.GAUGE, "percent")
        registry.record_metric("cpu_usage", 75.0, MetricType.GAUGE, "percent")
        
        # Gauge should keep latest value
        gauge_metrics = [m for m in registry.metrics if m.name == "cpu_usage"]
        assert len(gauge_metrics) == 2  # Both recorded
        assert gauge_metrics[-1].value == 75.0  # Latest value
    
    def test_record_histogram_metric(self):
        """Test recording histogram metrics."""
        registry = MetricsRegistry()
        
        # Record multiple values for histogram
        for value in [10, 20, 30, 40, 50]:
            registry.record_metric("response_time", value, MetricType.HISTOGRAM, "ms")
        
        histogram_metrics = [m for m in registry.metrics if m.name == "response_time"]
        assert len(histogram_metrics) == 5
    
    def test_get_metrics_by_name(self):
        """Test retrieving metrics by name."""
        registry = MetricsRegistry()
        
        registry.record_metric("test_metric", 1, MetricType.COUNTER)
        registry.record_metric("test_metric", 2, MetricType.COUNTER)
        registry.record_metric("other_metric", 3, MetricType.GAUGE)
        
        test_metrics = registry.get_metrics_by_name("test_metric")
        
        assert len(test_metrics) == 2
        assert all(m.name == "test_metric" for m in test_metrics)
    
    def test_get_metrics_by_type(self):
        """Test retrieving metrics by type."""
        registry = MetricsRegistry()
        
        registry.record_metric("counter1", 1, MetricType.COUNTER)
        registry.record_metric("counter2", 2, MetricType.COUNTER)
        registry.record_metric("gauge1", 3, MetricType.GAUGE)
        
        counters = registry.get_metrics_by_type(MetricType.COUNTER)
        gauges = registry.get_metrics_by_type(MetricType.GAUGE)
        
        assert len(counters) == 2
        assert len(gauges) == 1
    
    def test_clear_metrics(self):
        """Test clearing all metrics."""
        registry = MetricsRegistry()
        
        registry.record_metric("test", 1, MetricType.COUNTER)
        assert len(registry.metrics) == 1
        
        registry.clear_metrics()
        assert len(registry.metrics) == 0
    
    def test_export_prometheus_format(self):
        """Test exporting metrics in Prometheus format."""
        registry = MetricsRegistry()
        
        registry.record_metric("http_requests_total", 100, MetricType.COUNTER, 
                             labels={"method": "GET", "status": "200"})
        registry.record_metric("cpu_usage_percent", 75.5, MetricType.GAUGE)
        
        prometheus_output = registry.export_prometheus_format()
        
        assert "http_requests_total" in prometheus_output
        assert "cpu_usage_percent" in prometheus_output
        assert "method=\"GET\"" in prometheus_output
        assert "status=\"200\"" in prometheus_output
        assert "100" in prometheus_output
        assert "75.5" in prometheus_output


class TestHealthRegistry:
    """Test HealthRegistry for health monitoring."""
    
    def test_health_registry_creation(self):
        """Test HealthRegistry initialization."""
        registry = HealthRegistry()
        
        assert len(registry.health_checks) == 0
        assert isinstance(registry.health_checks, dict)
    
    def test_register_health_check(self):
        """Test registering health checks."""
        registry = HealthRegistry()
        
        def mock_check():
            return HealthCheckResult("test", HealthStatus.HEALTHY, "OK")
        
        registry.register_health_check("test_component", mock_check)
        
        assert "test_component" in registry.health_checks
        assert callable(registry.health_checks["test_component"])
    
    def test_run_health_check(self):
        """Test running a specific health check."""
        registry = HealthRegistry()
        
        def mock_check():
            return HealthCheckResult("test", HealthStatus.HEALTHY, "All good")
        
        registry.register_health_check("test", mock_check)
        result = registry.run_health_check("test")
        
        assert result.component == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
    
    def test_run_all_health_checks(self):
        """Test running all health checks."""
        registry = HealthRegistry()
        
        def healthy_check():
            return HealthCheckResult("healthy_component", HealthStatus.HEALTHY, "OK")
        
        def unhealthy_check():
            return HealthCheckResult("unhealthy_component", HealthStatus.UNHEALTHY, "Failed")
        
        registry.register_health_check("healthy", healthy_check)
        registry.register_health_check("unhealthy", unhealthy_check)
        
        results = registry.run_all_health_checks()
        
        assert len(results) == 2
        assert results["healthy"].status == HealthStatus.HEALTHY
        assert results["unhealthy"].status == HealthStatus.UNHEALTHY
    
    def test_get_overall_health_status(self):
        """Test getting overall health status."""
        registry = HealthRegistry()
        
        def healthy_check():
            return HealthCheckResult("healthy", HealthStatus.HEALTHY, "OK")
        
        def degraded_check():
            return HealthCheckResult("degraded", HealthStatus.DEGRADED, "Slow")
        
        # All healthy
        registry.register_health_check("healthy1", healthy_check)
        registry.register_health_check("healthy2", healthy_check)
        
        overall_status = registry.get_overall_health_status()
        assert overall_status == HealthStatus.HEALTHY
        
        # Add degraded component
        registry.register_health_check("degraded", degraded_check)
        overall_status = registry.get_overall_health_status()
        assert overall_status == HealthStatus.DEGRADED


class TestDashboardGenerator:
    """Test DashboardGenerator for monitoring configurations."""
    
    def test_dashboard_generator_creation(self):
        """Test DashboardGenerator initialization."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        assert generator.metrics_registry == metrics_registry
        assert generator.health_registry == health_registry
    
    def test_generate_grafana_dashboard(self):
        """Test generating Grafana dashboard configuration."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        # Add some metrics
        metrics_registry.record_metric("requests_total", 100, MetricType.COUNTER)
        metrics_registry.record_metric("cpu_usage", 75, MetricType.GAUGE)
        
        dashboard_config = generator.generate_grafana_dashboard()
        
        assert "dashboard" in dashboard_config
        assert "panels" in dashboard_config["dashboard"]
        assert dashboard_config["dashboard"]["title"] == "Document Chunking System - Phase 4 Observability"
    
    def test_generate_prometheus_config(self):
        """Test generating Prometheus configuration."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        prometheus_config = generator.generate_prometheus_config()
        
        assert "global" in prometheus_config
        assert "scrape_configs" in prometheus_config
        assert "rule_files" in prometheus_config
    
    def test_generate_alert_rules(self):
        """Test generating alert rules."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        alert_rules = generator.generate_alert_rules()
        
        assert "groups" in alert_rules
        assert len(alert_rules["groups"]) > 0
        assert "chunking_system_alerts" in [group["name"] for group in alert_rules["groups"]]


class TestObservabilityManager:
    """Test ObservabilityManager central coordination."""
    
    def test_observability_manager_creation(self):
        """Test ObservabilityManager initialization."""
        manager = ObservabilityManager()
        
        assert isinstance(manager.metrics_registry, MetricsRegistry)
        assert isinstance(manager.health_registry, HealthRegistry)
        assert isinstance(manager.dashboard_generator, DashboardGenerator)
        assert hasattr(manager, 'logger')
    
    @patch('src.utils.observability.StructuredLogger')
    def test_record_metric_integration(self, mock_logger_class):
        """Test recording metrics through manager."""
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger
        
        manager = ObservabilityManager()
        manager.record_metric("test_metric", 42, MetricType.GAUGE)
        
        # Verify metric was recorded
        metrics = manager.metrics_registry.get_metrics_by_name("test_metric")
        assert len(metrics) == 1
        assert metrics[0].value == 42
    
    def test_health_check_integration(self):
        """Test health check through manager."""
        manager = ObservabilityManager()
        
        def test_check():
            return HealthCheckResult("test", HealthStatus.HEALTHY, "OK")
        
        manager.register_health_check("test_component", test_check)
        result = manager.run_health_check("test_component")
        
        assert result.component == "test"
        assert result.status == HealthStatus.HEALTHY
    
    def test_export_all_data(self):
        """Test exporting all observability data."""
        manager = ObservabilityManager()
        
        # Add some data
        manager.record_metric("test_metric", 100, MetricType.COUNTER)
        
        def health_check():
            return HealthCheckResult("test", HealthStatus.HEALTHY, "OK")
        
        manager.register_health_check("test", health_check)
        
        export_data = manager.export_all_data()
        
        assert "metrics" in export_data
        assert "health_checks" in export_data
        assert "prometheus_format" in export_data
        assert len(export_data["metrics"]["metrics"]) == 1
        assert len(export_data["health_checks"]["components"]) >= 1


class TestIntegrationScenarios:
    """Test integration scenarios across observability components."""
    
    def test_end_to_end_observability_flow(self):
        """Test complete observability workflow."""
        manager = ObservabilityManager()
        
        # Set up health check
        def database_health():
            return HealthCheckResult("database", HealthStatus.HEALTHY, 
                                   "Connection pool active", response_time_ms=25.5)
        
        manager.register_health_check("database", database_health)
        
        # Record some metrics
        manager.record_metric("requests_total", 100, MetricType.COUNTER, 
                            labels={"endpoint": "/api/health"})
        manager.record_metric("response_time_ms", 150, MetricType.HISTOGRAM)
        manager.record_metric("active_connections", 25, MetricType.GAUGE)
        
        # Export data
        export_data = manager.export_all_data()
        
        # Verify complete data export
        assert len(export_data["metrics"]["metrics"]) == 3
        assert len(export_data["health_checks"]["components"]) >= 1
        assert export_data["health_checks"]["components"]["database"]["status"] == "healthy"
        assert "requests_total" in export_data["prometheus_format"]
        assert "response_time_ms" in export_data["prometheus_format"]
        assert "active_connections" in export_data["prometheus_format"]
    
    def test_correlation_id_flow(self):
        """Test correlation ID propagation through operations."""
        manager = ObservabilityManager()
        correlation_manager = CorrelationIDManager()
        
        # Start operation with correlation ID
        correlation_id = correlation_manager.generate_correlation_id()
        correlation_manager.set_correlation_id(correlation_id)
        
        # Record metrics with correlation context
        manager.record_metric("operation_started", 1, MetricType.COUNTER)
        
        # Verify correlation ID is maintained
        retrieved_id = correlation_manager.get_correlation_id()
        assert retrieved_id == correlation_id
        
        # Clean up
        correlation_manager.clear_correlation_id()
        assert correlation_manager.get_correlation_id() is None
    
    def test_dashboard_generation_with_real_data(self):
        """Test dashboard generation with realistic data."""
        manager = ObservabilityManager()
        
        # Add realistic metrics
        metrics_data = [
            ("chunking_operations_total", 1250, MetricType.COUNTER),
            ("chunking_duration_ms", 850, MetricType.HISTOGRAM),
            ("system_cpu_percent", 65.5, MetricType.GAUGE),
            ("system_memory_percent", 78.2, MetricType.GAUGE),
            ("cache_hit_rate", 89.5, MetricType.GAUGE),
        ]
        
        for name, value, metric_type in metrics_data:
            manager.record_metric(name, value, metric_type)
        
        # Add health checks
        def system_health():
            return HealthCheckResult("system", HealthStatus.HEALTHY, "All systems operational")
        
        def cache_health():
            return HealthCheckResult("cache", HealthStatus.DEGRADED, "High miss rate")
        
        manager.register_health_check("system", system_health)
        manager.register_health_check("cache", cache_health)
        
        # Generate dashboard
        dashboard_config = manager.dashboard_generator.generate_grafana_dashboard()
        
        # Verify dashboard contains expected panels
        panels = dashboard_config["dashboard"]["panels"]
        panel_titles = [panel["title"] for panel in panels]
        
        expected_panels = [
            "System Health Status",
            "CPU Usage", 
            "Memory Usage",
            "Chunking Operations Rate",
            "Processing Duration"
        ]
        
        for expected_panel in expected_panels:
            assert expected_panel in panel_titles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])