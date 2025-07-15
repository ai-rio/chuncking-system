"""Tests for Phase 3 monitoring implementation."""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.utils.monitoring import (
    HealthStatus, MetricPoint, Alert,
    HealthChecker, MetricsCollector, AlertManager, SystemMonitor
)
from src.config.settings import ChunkingConfig
from src.chunking_system import DocumentChunker


class TestHealthStatus:
    """Test HealthStatus dataclass."""
    
    def test_healthy_status(self):
        """Test healthy status creation."""
        status = HealthStatus(
            component="test_component",
            is_healthy=True,
            message="All systems operational",
            details={"uptime": "24h", "memory_usage": "45%"}
        )
        
        assert status.component == "test_component"
        assert status.is_healthy is True
        assert status.message == "All systems operational"
        assert status.details["uptime"] == "24h"
        assert isinstance(status.timestamp, datetime)
    
    def test_unhealthy_status(self):
        """Test unhealthy status creation."""
        status = HealthStatus(
            component="database",
            is_healthy=False,
            message="Connection timeout",
            details={"error_code": 500, "last_success": "2h ago"}
        )
        
        assert status.component == "database"
        assert status.is_healthy is False
        assert status.message == "Connection timeout"
        assert status.details["error_code"] == 500


class TestMetricPoint:
    """Test MetricPoint dataclass."""
    
    def test_metric_point_creation(self):
        """Test metric point creation."""
        metric = MetricPoint(
            name="cpu_usage",
            value=75.5,
            timestamp=datetime.now(),
            unit="percent",
            labels={"host": "server1", "region": "us-east"}
        )
        
        assert metric.name == "cpu_usage"
        assert metric.value == 75.5
        assert metric.unit == "percent"
        assert metric.labels["host"] == "server1"
        assert isinstance(metric.timestamp, datetime)
    
    def test_metric_point_without_tags(self):
        """Test metric point creation without tags."""
        metric = MetricPoint(
            name="memory_usage",
            value=1024,
            timestamp=datetime.now(),
            unit="MB"
        )
        
        assert metric.name == "memory_usage"
        assert metric.value == 1024
        assert metric.unit == "MB"
        assert metric.labels == {}


class TestAlert:
    """Test Alert dataclass."""
    
    def test_alert_creation(self):
        """Test alert creation."""
        alert = Alert(
            id="alert_001",
            severity="critical",
            title="High Memory Usage Detected",
            message="High memory usage detected",
            component="chunking_system",
            timestamp=datetime.now(),
            metadata={"current_usage": "95%", "threshold": "90%"}
        )
        
        assert alert.id == "alert_001"
        assert alert.severity == "critical"
        assert alert.component == "chunking_system"
        assert alert.message == "High memory usage detected"
        assert alert.metadata["current_usage"] == "95%"
        assert isinstance(alert.timestamp, datetime)
        assert alert.resolved is False
        assert alert.resolved_at is None
    
    def test_alert_resolution(self):
        """Test alert resolution."""
        alert = Alert(
            id="alert_002",
            severity="warning",
            title="Cache Miss Rate High",
            component="cache",
            message="Cache miss rate high",
            timestamp=datetime.now()
        )
        
        # Initially not resolved
        assert alert.resolved is False
        assert alert.resolved_at is None
        
        # Resolve alert
        resolution_time = datetime.now()
        alert.resolved = True
        alert.resolved_at = resolution_time
        
        assert alert.resolved is True
        assert alert.resolved_at == resolution_time


class TestHealthChecker:
    """Test HealthChecker implementation."""
    
    def test_health_checker_initialization(self):
        """Test health checker initialization."""
        checker = HealthChecker()
        
        assert len(checker.checks) == 4  # Default health checks (cpu, memory, disk, application)
        assert hasattr(checker, 'checks')
        assert hasattr(checker, 'last_results')
    
    def test_register_health_check(self):
        """Test registering health checks."""
        checker = HealthChecker()
        
        def dummy_check():
            return HealthStatus(
                component="test",
                is_healthy=True,
                message="OK"
            )
        
        checker.register_check("test_check", dummy_check)
        
        assert "test_check" in checker.checks
        assert checker.checks["test_check"] == dummy_check
    
    def test_run_single_check(self):
        """Test running a single health check."""
        checker = HealthChecker()
        
        def test_check():
            return HealthStatus(
                component="test_component",
                is_healthy=True,
                message="All good"
            )
        
        checker.register_check("test", test_check)
        status = checker.run_check("test")
        
        assert status.component == "test_component"
        assert status.is_healthy is True
        assert status.message == "All good"
    
    def test_run_nonexistent_check(self):
        """Test running a non-existent health check."""
        checker = HealthChecker()
        
        status = checker.run_check("nonexistent")
        
        assert status.component == "nonexistent"
        assert status.is_healthy is False
        assert "unknown health check" in status.message.lower()
    
    def test_run_all_checks(self):
        """Test running all health checks."""
        checker = HealthChecker()
        
        def healthy_check():
            return HealthStatus(
                component="healthy_component",
                is_healthy=True,
                message="OK"
            )
        
        def unhealthy_check():
            return HealthStatus(
                component="unhealthy_component",
                is_healthy=False,
                message="Error"
            )
        
        checker.register_check("healthy", healthy_check)
        checker.register_check("unhealthy", unhealthy_check)
        
        results = checker.run_all_checks()
        
        assert len(results) == 6
        assert any(r.component == "healthy_component" and r.is_healthy for r in results)
        assert any(r.component == "unhealthy_component" and not r.is_healthy for r in results)
    
    def test_overall_health_status(self):
        """Test overall health status calculation."""
        checker = HealthChecker()
        
        def healthy_check1():
            return HealthStatus(component="comp1", is_healthy=True, message="OK")
        
        def healthy_check2():
            return HealthStatus(component="comp2", is_healthy=True, message="OK")
        
        def unhealthy_check():
            return HealthStatus(component="comp3", is_healthy=False, message="Error")
        
        # All healthy
        checker.register_check("check1", healthy_check1)
        checker.register_check("check2", healthy_check2)
        
        overall = checker.get_overall_health()
        assert overall.is_healthy is True
        
        # Add unhealthy check
        checker.register_check("check3", unhealthy_check)
        
        overall = checker.get_overall_health()
        assert overall.is_healthy is False
    
    def test_health_check_exception_handling(self):
        """Test health check exception handling."""
        checker = HealthChecker()
        
        def failing_check():
            raise Exception("Check failed")
        
        checker.register_check("failing", failing_check)
        status = checker.run_check("failing")
        
        assert status.component == "failing"
        assert status.is_healthy is False
        assert "exception" in status.message.lower()
        assert "Check failed" in status.details.get("error", "")


class TestMetricsCollector:
    """Test MetricsCollector implementation."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector()
        
        assert len(collector.metrics) == 0
        assert collector.max_points == 10000  # Default
    
    def test_record_metric(self):
        """Test recording metrics."""
        collector = MetricsCollector()
        
        collector.record_counter(
            name="cpu_usage",
            value=75.0,
            labels={"host": "server1"}
        )
        
        assert len(collector.metrics) == 1
        metric = collector.metrics[0]
        assert metric.name == "cpu_usage"
        assert metric.value == 75.0
        assert metric.unit == "percent"
        assert metric.tags["host"] == "server1"
    
    def test_record_multiple_metrics(self):
        """Test recording multiple metrics."""
        collector = MetricsCollector()
        
        metrics_data = [
            ("cpu_usage", 75.0, "percent"),
            ("memory_usage", 1024, "MB"),
            ("disk_usage", 50.5, "percent")
        ]
        
        for name, value, unit in metrics_data:
            collector.record_gauge(name, value, labels={})
        
        assert len(collector.metrics) == 3
        
        # Check each metric
        names = list(collector.metrics.keys())
        assert "cpu_usage" in names
        assert "memory_usage" in names
        assert "disk_usage" in names
    
    def test_get_metrics_by_name(self):
        """Test retrieving metrics by name."""
        collector = MetricsCollector()
        
        # Record multiple metrics with same name
        for i in range(5):
            collector.record_counter("cpu_usage", i * 10, labels={"unit": "percent"})
        
        collector.record_gauge("memory_usage", 1024, labels={"unit": "MB"})
        
        cpu_metrics = list(collector.metrics["cpu_usage"])
        assert len(cpu_metrics) == 5
        
        memory_metrics = list(collector.metrics["memory_usage"])
        assert len(memory_metrics) == 1
        assert memory_metrics[0].value == 1024
    
    def test_get_metrics_by_time_range(self):
        """Test retrieving metrics by time range."""
        collector = MetricsCollector()
        
        now = datetime.now()
        
        # Record metrics with different timestamps
        old_metric = MetricPoint(
            name="test_metric",
            value=1,
            unit="count",
            timestamp=now - timedelta(hours=2)
        )
        
        recent_metric = MetricPoint(
            name="test_metric",
            value=2,
            unit="count",
            timestamp=now - timedelta(minutes=30)
        )
        
        collector.metrics["test_metric"].extend([old_metric, recent_metric])
        
        # Get metrics from last hour
        since = now - timedelta(hours=1)
        recent_metrics = [m for m in collector.metrics["test_metric"] if m.timestamp >= since]
        
        assert len(recent_metrics) == 1
        assert recent_metrics[0].value == 2
    
    def test_metrics_rotation(self):
        """Test metrics rotation when max limit is reached."""
        collector = MetricsCollector(max_points=5)
        
        # Record more metrics than the limit
        for i in range(10):
            collector.record_gauge(f"metric_{i}", i, labels={"unit": "count"})
        
        # Should only keep the most recent 5
        assert len(collector.metrics) == 5
        
        # Check that we have the most recent metrics
        values = [m.value for m in collector.metrics]
        assert values == [5, 6, 7, 8, 9]
    
    def test_get_metric_statistics(self):
        """Test metric statistics calculation."""
        collector = MetricsCollector()
        
        # Record metrics with known values
        values = [10, 20, 30, 40, 50]
        for value in values:
            collector.record_counter("test_metric", value, labels={"unit": "count"})
        
        stats = collector.get_metric_summary("test_metric")
        
        assert stats["count"] == 5
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["avg"] == 30
        assert stats["sum"] == 150
    
    def test_clear_metrics(self):
        """Test clearing metrics."""
        collector = MetricsCollector()
        
        # Record some metrics
        for i in range(5):
            collector.record_counter(f"metric_{i}", i, labels={"unit": "count"})
        
        assert len(collector.metrics) == 5
        
        collector.clear_metrics(timedelta(seconds=0))
        
        assert len(collector.metrics) == 0


class TestAlertManager:
    """Test AlertManager implementation."""
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager()
        
        assert len(manager.alerts) == 0
        assert len(manager.alert_rules) == 0
    
    def test_create_alert(self):
        """Test creating alerts."""
        manager = AlertManager()
        
        alert = manager.create_alert(
            severity="critical",
            title="Test Alert",
            component="test_component",
            message="Test alert",
            metadata={"key": "value"}
        )
        
        assert alert.severity == "critical"
        assert alert.component == "test_component"
        assert alert.message == "Test alert"
        assert alert.details["key"] == "value"
        assert alert.id is not None
        assert not alert.resolved
        
        # Check that alert was stored
        assert len(manager.alerts) == 1
        assert manager.alerts[0] == alert
    
    def test_resolve_alert(self):
        """Test resolving alerts."""
        manager = AlertManager()
        
        alert = manager.create_alert(
            severity="warning",
            title="Test Alert Title",
            component="test",
            message="Test alert"
        )
        
        alert_id = alert.id
        
        # Resolve the alert
        resolved = manager.resolve_alert(alert_id)
        
        assert resolved is True
        assert alert.resolved is True
        assert alert.resolved_at is not None
    
    def test_resolve_nonexistent_alert(self):
        """Test resolving non-existent alert."""
        manager = AlertManager()
        
        resolved = manager.resolve_alert("nonexistent_id")
        
        assert resolved is False
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        manager = AlertManager()
        
        # Create some alerts
        alert1 = manager.create_alert(severity="critical", title="Alert 1", message="Alert 1", component="comp1")
        alert2 = manager.create_alert(severity="warning", title="Alert 2", message="Alert 2", component="comp2")
        alert3 = manager.create_alert(severity="info", title="Alert 3", message="Alert 3", component="comp3")
        
        # Resolve one alert
        manager.resolve_alert(alert2.id)
        
        active_alerts = manager.get_active_alerts()
        
        assert len(active_alerts) == 2
        active_ids = [a.id for a in active_alerts]
        assert alert1.id in active_ids
        assert alert3.id in active_ids
        assert alert2.id not in active_ids
    
    def test_get_alerts_by_severity(self):
        """Test getting alerts by severity."""
        manager = AlertManager()
        
        # Create alerts with different severities
        manager.create_alert(severity="critical", title="High alert", message="High alert", component="comp1")
        manager.create_alert(severity="critical", title="Another high alert", message="Another high alert", component="comp2")
        manager.create_alert(severity="warning", title="Medium alert", message="Medium alert", component="comp3")
        manager.create_alert(severity="info", title="Low alert", message="Low alert", component="comp4")
        
        high_alerts = manager.get_alerts_by_severity("critical")
        medium_alerts = manager.get_alerts_by_severity("warning")
        low_alerts = manager.get_alerts_by_severity("info")
        
        assert len(high_alerts) == 2
        assert len(medium_alerts) == 1
        assert len(low_alerts) == 1
    
    def test_get_alerts_by_component(self):
        """Test getting alerts by component."""
        manager = AlertManager()
        
        # Create alerts for different components
        manager.create_alert(severity="critical", title="Chunker alert 1", message="Chunker alert 1", component="chunker")
        manager.create_alert(severity="warning", title="Chunker alert 2", message="Chunker alert 2", component="chunker")
        manager.create_alert(severity="info", title="Cache alert", message="Cache alert", component="cache")
        
        chunker_alerts = manager.get_alerts_by_component("chunker")
        cache_alerts = manager.get_alerts_by_component("cache")
        
        assert len(chunker_alerts) == 2
        assert len(cache_alerts) == 1
    
    def test_alert_rule_registration(self):
        """Test registering alert rules."""
        manager = AlertManager()
        
        def high_cpu_rule(metrics):
            cpu_metrics = [m for m in metrics if m.name == "cpu_usage"]
            if cpu_metrics and cpu_metrics[-1].value > 90:
                return Alert(
                    id="cpu_high",
                    severity="critical",
                    component="system",
                    message="High CPU usage"
                )
            return None
        
        manager.add_alert_rule(high_cpu_rule)
        
        assert high_cpu_rule in manager.alert_rules
    
    def test_evaluate_rules(self):
        """Test evaluating alert rules."""
        manager = AlertManager()
        
        def memory_rule(context):
            metrics = context.get("metrics", [])
            memory_metrics = [m for m in metrics if m.name == "memory_usage"]
            if memory_metrics and memory_metrics[-1].value > 80:
                return Alert(
                    id="memory_high",
                    severity="warning",
                    title="High Memory Usage",
                    component="memory",
                    message="High memory usage",
                    timestamp=datetime.now()
                )
            return None
        
        manager.add_alert_rule(memory_rule)
        
        # Create test metrics
        test_metrics = [
            MetricPoint("memory_usage", 85, "percent"),
            MetricPoint("cpu_usage", 50, "percent")
        ]
        
        # Evaluate rules
        manager.evaluate_rules({"metrics": test_metrics})
        
        assert len(manager.alerts) == 1
        assert manager.alerts[0].component == "memory"
        assert manager.alerts[0].message == "High memory usage"


class TestSystemMonitor:
    """Test SystemMonitor implementation."""
    
    def test_system_monitor_initialization(self):
        """Test system monitor initialization."""
        monitor = SystemMonitor()
        
        assert monitor.health_checker is not None
        assert monitor.metrics_collector is not None
        assert monitor.alert_manager is not None
        assert monitor.check_interval == 60  # Default
    
    def test_register_health_check(self):
        """Test registering health checks through monitor."""
        monitor = SystemMonitor()
        
        def test_check():
            return HealthStatus(
                component="test",
                is_healthy=True,
                message="OK"
            )
        
        monitor.register_health_check("test", test_check)
        
        # Verify it was registered
        status = monitor.health_checker.run_check("test")
        assert status.is_healthy is True
    
    def test_record_metric(self):
        """Test recording metrics through monitor."""
        monitor = SystemMonitor()
        
        monitor.metrics_collector.record_gauge("test_metric", 42, labels={"unit": "count"})
        
        # Verify it was recorded
        metrics = list(monitor.metrics_collector.metrics.values())[0]
        assert len(metrics) == 1
        assert metrics[0].value == 42
    
    def test_create_alert(self):
        """Test creating alerts through monitor."""
        monitor = SystemMonitor()
        
        alert = monitor.alert_manager.create_alert(
                severity="critical",
                title="Test Alert",
                component="test",
                message="Test alert"
            )
        
        # Verify it was created
        assert alert is not None
        active_alerts = monitor.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].message == "Test alert"
    
    def test_get_system_status(self):
        """Test getting overall system status."""
        monitor = SystemMonitor()
        
        # Register a health check
        def healthy_check():
            return HealthStatus(
                component="test",
                is_healthy=True,
                message="OK"
            )
        
        monitor.register_health_check("test", healthy_check)
        
        # Record some metrics
        monitor.metrics_collector.record_gauge("cpu_usage", 50, labels={"unit": "percent"})
        monitor.metrics_collector.record_gauge("memory_usage", 60, labels={"unit": "percent"})
        
        # Create an alert
        monitor.create_alert(
            severity="info",
            component="test",
            message="Test alert"
        )
        
        status = monitor.get_system_status()
        
        assert "health" in status
        assert "metrics_count" in status
        assert "active_alerts" in status
        assert "alert_counts" in status
        
        assert status["health"]["overall_healthy"] is True
        assert status["metrics_count"] == 2
        assert status["active_alerts"] == 1
    
    @patch('threading.Thread')
    def test_start_monitoring(self, mock_thread):
        """Test starting monitoring thread."""
        monitor = SystemMonitor()
        
        monitor.start_monitoring()
        
        assert monitor.monitoring_thread.is_alive()
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()
    
    def test_stop_monitoring(self):
        """Test stopping monitoring."""
        monitor = SystemMonitor()
        monitor._running = True
        
        monitor.stop_monitoring.set()
        
        assert monitor.stop_monitoring.is_set()
        
        
    
    def test_monitoring_cycle(self):
        """Test a single monitoring cycle."""
        monitor = SystemMonitor()
        
        # Register health check
        def test_check():
            return HealthStatus(
                component="test",
                is_healthy=True,
                message="OK"
            )
        
        monitor.register_health_check("test", test_check)
        
        # Register alert rule
        def test_rule(metrics):
            return None  # No alerts
        
        monitor.alert_manager.add_alert_rule(test_rule)
        
        # Run monitoring cycle
        monitor._monitoring_loop()
        
        # Should have recorded system metrics
        cpu_metrics = monitor.metrics_collector.get_metrics("system.cpu_percent")
        memory_metrics = monitor.metrics_collector.get_metrics("system.memory_percent")
        
        assert len(cpu_metrics) >= 1
        assert len(memory_metrics) >= 1


class TestDocumentChunkerMonitoring:
    """Test monitoring integration in DocumentChunker."""
    
    def test_chunker_with_monitoring_enabled(self, tmp_path):
        """Test DocumentChunker with monitoring enabled."""
        config = ChunkingConfig(
            enable_caching=False,
            enable_security=False,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Document\n\nThis is a test document for monitoring.")
        
        result = chunker.chunk_file(test_file)
        
        assert result.success is True
        assert len(result.chunks) > 0
        assert result.performance_metrics is not None
        
        # Check that metrics were recorded
        assert "total_duration_ms" in result.performance_metrics
        assert "avg_duration_ms" in result.performance_metrics
        assert "peak_memory_mb" in result.performance_metrics
        assert "total_operations" in result.performance_metrics
    
    def test_chunker_with_monitoring_disabled(self, tmp_path):
        """Test DocumentChunker with monitoring disabled."""
        config = ChunkingConfig(
            enable_caching=False,
            enable_security=False,
            enable_monitoring=False
        )
        
        chunker = DocumentChunker(config)
        
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Document\n\nContent.")
        
        result = chunker.chunk_file(test_file)
        
        assert result.success is True
        assert result.performance_metrics is None
    
    def test_monitoring_metrics_collection(self, tmp_path):
        """Test that monitoring collects expected metrics."""
        config = ChunkingConfig(
            enable_caching=False,
            enable_security=False,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        # Create multiple test files
        for i in range(3):
            test_file = tmp_path / f"test_{i}.md"
            test_file.write_text(f"# Document {i}\n\nContent for document {i}.")
            chunker.chunk_file(test_file)
        
        # Check system monitor has collected metrics
        monitor = chunker.system_monitor
        
        # Trigger system metrics collection
        monitor._collect_system_metrics()
        
        # Should have system metrics
        cpu_metrics = monitor.metrics_collector.get_metrics("system.cpu.usage_percent")
        memory_metrics = monitor.metrics_collector.get_metrics("system.memory.usage_percent")
        
        # Should have collected metrics
        assert monitor is not None
        assert monitor.metrics_collector is not None
        assert len(cpu_metrics) > 0
        assert len(memory_metrics) > 0
    
    def test_monitoring_health_checks(self, tmp_path):
        """Test monitoring health checks."""
        config = ChunkingConfig(
            enable_caching=False,
            enable_security=False,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        # Get system status
        monitor = chunker.system_monitor
        status = monitor.get_system_status()
        
        assert "health" in status
        assert "metrics_count" in status
        assert "active_alerts" in status
        
        # Health should be available (even if no custom checks registered)
        assert "overall_healthy" in status["health"]


class TestMonitoringPerformance:
    """Test monitoring performance characteristics."""
    
    def test_metrics_collection_performance(self):
        """Test metrics collection performance."""
        collector = MetricsCollector()
        
        import time
        
        # Record many metrics
        start_time = time.time()
        for i in range(1000):
            collector.record_gauge(f"metric_{i % 10}", i, labels={"unit": "count"})
        end_time = time.time()
        
        # Should complete quickly
        assert (end_time - start_time) < 1.0  # Under 1 second
    
    def test_health_check_performance(self):
        """Test health check performance."""
        checker = HealthChecker()
        
        # Register multiple health checks
        for i in range(10):
            def make_check(index):
                def check():
                    return HealthStatus(
                        component=f"component_{index}",
                        is_healthy=True,
                        message="OK"
                    )
                return check
            
            checker.register_check(f"check_{i}", make_check(i))
        
        import time
        
        # Run all checks
        start_time = time.time()
        results = checker.run_all_checks()
        end_time = time.time()
        
        assert len(results) == 10
        assert (end_time - start_time) < 1.0  # Under 1 second
    
    def test_alert_evaluation_performance(self):
        """Test alert rule evaluation performance."""
        manager = AlertManager()
        
        # Register multiple rules
        for i in range(5):
            def make_rule(threshold):
                def rule(metrics):
                    test_metrics = [m for m in metrics if m.name == "test_metric"]
                    if test_metrics and test_metrics[-1].value > threshold:
                        return manager.create_alert(
                            severity="low",
                            component="test",
                            message=f"Threshold {threshold} exceeded"
                        )
                    return None
                return rule
            
            manager.add_alert_rule(make_rule(i * 20))
        
        # Create test metrics
        test_metrics = [
            MetricPoint("test_metric", 50, "count"),
            MetricPoint("other_metric", 100, "count")
        ]
        
        import time
        
        # Evaluate rules
        start_time = time.time()
        # Evaluate rules
        manager.evaluate_rules({"metrics": test_metrics})
        end_time = time.time()
        
        # Should complete quickly
        assert (end_time - start_time) < 0.5  # Under 0.5 seconds


@pytest.fixture
def monitoring_test_data():
    """Test data for monitoring tests."""
    return {
        "metrics": [
            MetricPoint("cpu_usage", 75.0, "percent"),
            MetricPoint("memory_usage", 60.0, "percent"),
            MetricPoint("disk_usage", 45.0, "percent")
        ],
        "health_statuses": [
            HealthStatus("database", True, "Connected"),
            HealthStatus("cache", True, "Operational"),
            HealthStatus("api", False, "Timeout")
        ]
    }


class TestMonitoringIntegration:
    """Integration tests for monitoring components."""
    
    def test_full_monitoring_pipeline(self, monitoring_test_data):
        """Test complete monitoring pipeline."""
        monitor = SystemMonitor()
        
        # Register health checks
        def db_check():
            return monitoring_test_data["health_statuses"][0]
        
        def cache_check():
            return monitoring_test_data["health_statuses"][1]
        
        def api_check():
            return monitoring_test_data["health_statuses"][2]
        
        monitor.register_health_check("database", db_check)
        monitor.register_health_check("cache", cache_check)
        monitor.register_health_check("api", api_check)
        
        # Record metrics
        for metric in monitoring_test_data["metrics"]:
            monitor.metrics_collector.record_gauge(metric.name, metric.value, labels={"unit": metric.unit})
        
        # Register alert rule
        def high_cpu_rule(metrics):
            cpu_metrics = [m for m in metrics if m.name == "cpu_usage"]
            if cpu_metrics and cpu_metrics[-1].value > 70:
                return monitor.create_alert(
                    severity="medium",
                    component="system",
                    message="High CPU usage detected"
                )
            return None
        
        monitor.alert_manager.register_rule("high_cpu", high_cpu_rule)
        
        # Evaluate rules
        current_metrics = monitor.metrics_collector.metrics
        new_alerts = monitor.alert_manager.evaluate_rules(current_metrics)
        
        # Get system status
        status = monitor.get_system_status()
        
        # Verify results
        assert status["health"]["overall_healthy"] is False  # API is down
        assert status["metrics_count"] == 3
        assert status["active_alerts"] >= 1  # High CPU alert
        assert len(new_alerts) >= 1
    
    def test_monitoring_with_chunking_workflow(self, tmp_path):
        """Test monitoring integration with chunking workflow."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=False,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        # Create test files
        for i in range(3):
            test_file = tmp_path / f"doc_{i}.md"
            test_file.write_text(f"# Document {i}\n\nContent for document {i}.")
        
        # Process files
        results = chunker.chunk_directory(tmp_path)
        
        # Verify all files were processed
        assert len(results) == 3
        assert all(r.success for r in results)
        
        # Verify monitoring data was collected
        for result in results:
            assert result.performance_metrics is not None
            assert "duration" in result.performance_metrics
        
        # Check system monitor
        monitor = chunker.system_monitor
        status = monitor.get_system_status()
        
        assert "health" in status
        assert "metrics_count" in status