"""Monitoring and observability utilities for the document chunking system.

This module provides comprehensive monitoring capabilities including:
- Health checks and system status monitoring
- Metrics collection and reporting
- Application performance monitoring (APM)
- Resource usage tracking
- Alert management
"""

import os
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import json
from contextlib import contextmanager
from src.utils.logger import get_logger
from src.utils.performance import PerformanceMonitor


@dataclass
class HealthStatus:
    """Health check status information."""
    component: str
    message: str
    is_healthy: bool = True
    status: str = field(default="")
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set status based on is_healthy if not provided."""
        if not self.status:
            self.status = "healthy" if self.is_healthy else "unhealthy"


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class Alert:
    """Alert information."""
    id: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    title: str
    message: str
    timestamp: datetime
    component: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """System health monitoring and checks."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.checks: Dict[str, Callable[[], HealthStatus]] = {}
        self.last_results: Dict[str, HealthStatus] = {}
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default system health checks."""
        self.register_check("system_memory", self._check_memory)
        self.register_check("system_disk", self._check_disk)
        self.register_check("system_cpu", self._check_cpu)
        self.register_check("application_status", self._check_application)
    
    def register_check(self, name: str, check_func: Callable[[], HealthStatus]):
        """Register a custom health check."""
        self.checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    def _check_memory(self) -> HealthStatus:
        """Check system memory usage."""
        start_time = time.time()
        memory = psutil.virtual_memory()
        response_time = (time.time() - start_time) * 1000
        
        usage_percent = memory.percent
        
        if usage_percent > 90:
            status = "unhealthy"
            message = f"High memory usage: {usage_percent:.1f}%"
        elif usage_percent > 80:
            status = "degraded"
            message = f"Elevated memory usage: {usage_percent:.1f}%"
        else:
            status = "healthy"
            message = f"Memory usage normal: {usage_percent:.1f}%"
        
        return HealthStatus(
            component="system_memory",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=response_time,
            details={
                "usage_percent": usage_percent,
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3)
            }
        )
    
    def _check_disk(self) -> HealthStatus:
        """Check disk space usage."""
        start_time = time.time()
        disk = psutil.disk_usage('/')
        response_time = (time.time() - start_time) * 1000
        
        usage_percent = (disk.used / disk.total) * 100
        
        if usage_percent > 95:
            status = "unhealthy"
            message = f"Critical disk usage: {usage_percent:.1f}%"
        elif usage_percent > 85:
            status = "degraded"
            message = f"High disk usage: {usage_percent:.1f}%"
        else:
            status = "healthy"
            message = f"Disk usage normal: {usage_percent:.1f}%"
        
        return HealthStatus(
            component="system_disk",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=response_time,
            details={
                "usage_percent": usage_percent,
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "used_gb": disk.used / (1024**3)
            }
        )
    
    def _check_cpu(self) -> HealthStatus:
        """Check CPU usage."""
        start_time = time.time()
        cpu_percent = psutil.cpu_percent(interval=1)
        response_time = (time.time() - start_time) * 1000
        
        if cpu_percent > 90:
            status = "unhealthy"
            message = f"High CPU usage: {cpu_percent:.1f}%"
        elif cpu_percent > 80:
            status = "degraded"
            message = f"Elevated CPU usage: {cpu_percent:.1f}%"
        else:
            status = "healthy"
            message = f"CPU usage normal: {cpu_percent:.1f}%"
        
        return HealthStatus(
            component="system_cpu",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=response_time,
            details={
                "usage_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        )
    
    def _check_application(self) -> HealthStatus:
        """Check application-specific health."""
        start_time = time.time()
        
        try:
            # Check if main modules can be imported
            from src.chunking_system import DocumentChunker
            from src.utils.cache import CacheManager
            
            # Basic functionality test
            chunker = DocumentChunker()
            cache_manager = CacheManager()
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthStatus(
                component="application_status",
                status="healthy",
                message="Application components operational",
                timestamp=datetime.now(),
                response_time_ms=response_time,
                details={
                    "chunker_available": True,
                    "cache_available": True,
                    "startup_time_ms": response_time
                }
            )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                component="application_status",
                status="unhealthy",
                message=f"Application error: {str(e)}",
                timestamp=datetime.now(),
                response_time_ms=response_time,
                details={"error": str(e)}
            )
    
    def run_check(self, check_name: str) -> HealthStatus:
        """Run a specific health check."""
        if check_name not in self.checks:
            return HealthStatus(
                component=check_name,
                status="unhealthy",
                message=f"Unknown health check: {check_name}",
                timestamp=datetime.now()
            )
        
        try:
            result = self.checks[check_name]()
            self.last_results[check_name] = result
            return result
        except Exception as e:
            self.logger.error(f"Health check failed: {check_name}", error=str(e))
            result = HealthStatus(
                component=check_name,
                status="unhealthy",
                message=f"Check failed: {str(e)}",
                timestamp=datetime.now()
            )
            self.last_results[check_name] = result
            return result
    
    def run_all_checks(self) -> Dict[str, HealthStatus]:
        """Run all registered health checks."""
        results = {}
        for check_name in self.checks:
            results[check_name] = self.run_check(check_name)
        return results
    
    def get_overall_status(self) -> str:
        """Get overall system health status."""
        results = self.run_all_checks()
        
        if any(result.status == "unhealthy" for result in results.values()):
            return "unhealthy"
        elif any(result.status == "degraded" for result in results.values()):
            return "degraded"
        else:
            return "healthy"
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health as HealthStatus object."""
        results = self.run_all_checks()
        overall_status = self.get_overall_status()
        
        healthy_count = sum(1 for r in results.values() if r.status == "healthy")
        total_count = len(results)
        
        message = f"System health: {healthy_count}/{total_count} components healthy"
        
        return HealthStatus(
            component="overall_system",
            status=overall_status,
            message=message,
            timestamp=datetime.now(),
            details={
                "total_components": total_count,
                "healthy_components": healthy_count,
                "component_statuses": {name: result.status for name, result in results.items()}
            }
        )


class MetricsCollector:
    """Metrics collection and aggregation."""
    
    def __init__(self, max_points: int = 10000):
        self.logger = get_logger(__name__)
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        with self._lock:
            key = self._make_key(name, tags)
            self.counters[key] += value
            
            metric_point = MetricPoint(
                name=name,
                value=self.counters[key],
                timestamp=datetime.now(),
                tags=tags or {},
                unit="count"
            )
            self.metrics[key].append(metric_point)
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        with self._lock:
            key = self._make_key(name, tags)
            self.gauges[key] = value
            
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                unit="gauge"
            )
            self.metrics[key].append(metric_point)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        with self._lock:
            key = self._make_key(name, tags)
            self.histograms[key].append(value)
            
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                unit="histogram"
            )
            self.metrics[key].append(metric_point)
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a unique key for metric with tags."""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def get_metric_summary(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        key = self._make_key(name, tags)
        
        if key not in self.metrics or not self.metrics[key]:
            return {"error": "No data available"}
        
        points = list(self.metrics[key])
        values = [point.value for point in points]
        
        if not values:
            return {"error": "No values available"}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[-1],
            "latest_timestamp": points[-1].timestamp.isoformat()
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all metrics."""
        result = {}
        
        with self._lock:
            for key in self.metrics:
                # Parse metric name from key
                if '[' in key:
                    name = key.split('[')[0]
                    tag_str = key.split('[')[1].rstrip(']')
                    tags = dict(tag.split('=') for tag in tag_str.split(',') if tag)
                else:
                    name = key
                    tags = None
                
                result[key] = self.get_metric_summary(name, tags)
        
        return result
    
    def clear_metrics(self, older_than: Optional[timedelta] = None):
        """Clear old metrics."""
        if older_than is None:
            older_than = timedelta(hours=24)
        
        cutoff_time = datetime.now() - older_than
        
        with self._lock:
            for key in list(self.metrics.keys()):
                # Filter out old points
                self.metrics[key] = deque(
                    (point for point in self.metrics[key] if point.timestamp > cutoff_time),
                    maxlen=self.metrics[key].maxlen
                )
                
                # Remove empty metrics
                if not self.metrics[key]:
                    del self.metrics[key]


class AlertManager:
    """Alert management and notification system."""
    
    def __init__(self, max_alerts: int = 1000):
        self.logger = get_logger(__name__)
        self.alerts: deque = deque(maxlen=max_alerts)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Callable[[Dict[str, Any]], Optional[Alert]]] = []
        self._lock = threading.Lock()
    
    def add_alert_rule(self, rule_func: Callable[[Dict[str, Any]], Optional[Alert]]):
        """Add an alert rule function."""
        self.alert_rules.append(rule_func)
    
    def create_alert(self, severity: str, title: str, message: str, component: str, 
                    alert_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a new alert."""
        if alert_id is None:
            alert_id = f"{component}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            component=component,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.alerts.append(alert)
            if severity in ['error', 'critical']:
                self.active_alerts[alert_id] = alert
        
        self.logger.warning(
            f"Alert created: {title}",
            alert_id=alert_id,
            severity=severity,
            component=component
        )
        
        return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                del self.active_alerts[alert_id]
                
                self.logger.info(f"Alert resolved: {alert.title}", alert_id=alert_id)
                return True
        
        return False
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """Get list of active alerts, optionally filtered by severity."""
        with self._lock:
            alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get recent alert history."""
        with self._lock:
            return list(self.alerts)[-limit:]
    
    def evaluate_rules(self, context: Dict[str, Any]):
        """Evaluate all alert rules against current context."""
        for rule_func in self.alert_rules:
            try:
                alert = rule_func(context)
                if alert:
                    self.create_alert(
                        severity=alert.severity,
                        title=alert.title,
                        message=alert.message,
                        component=alert.component,
                        alert_id=alert.id,
                        metadata=alert.metadata
                    )
            except Exception as e:
                self.logger.error(f"Alert rule evaluation failed: {e}")


class SystemMonitor:
    """Comprehensive system monitoring orchestrator."""
    
    def __init__(self, check_interval: int = 60):
        self.logger = get_logger(__name__)
        self.health_checker = HealthChecker()
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.performance_monitor = PerformanceMonitor()
        
        self.check_interval = check_interval
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        self._setup_default_alert_rules()
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        def memory_alert_rule(context: Dict[str, Any]) -> Optional[Alert]:
            health_results = context.get('health_results', {})
            memory_status = health_results.get('system_memory')
            
            if memory_status and memory_status.status == 'unhealthy':
                return Alert(
                    id="memory_critical",
                    severity="critical",
                    title="Critical Memory Usage",
                    message=memory_status.message,
                    timestamp=datetime.now(),
                    component="system_memory",
                    metadata=memory_status.details
                )
            return None
        
        def disk_alert_rule(context: Dict[str, Any]) -> Optional[Alert]:
            health_results = context.get('health_results', {})
            disk_status = health_results.get('system_disk')
            
            if disk_status and disk_status.status == 'unhealthy':
                return Alert(
                    id="disk_critical",
                    severity="critical",
                    title="Critical Disk Usage",
                    message=disk_status.message,
                    timestamp=datetime.now(),
                    component="system_disk",
                    metadata=disk_status.details
                )
            return None
        
        self.alert_manager.add_alert_rule(memory_alert_rule)
        self.alert_manager.add_alert_rule(disk_alert_rule)
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring_service(self):
        """Stop background monitoring."""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.wait(self.check_interval):
            try:
                # Run health checks
                health_results = self.health_checker.run_all_checks()
                
                # Collect system metrics
                self._collect_system_metrics()
                
                # Evaluate alert rules
                context = {
                    'health_results': health_results,
                    'timestamp': datetime.now()
                }
                self.alert_manager.evaluate_rules(context)
                
                # Clean up old data
                self.metrics_collector.clear_metrics(timedelta(hours=24))
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics_collector.record_gauge("system.memory.usage_percent", memory.percent)
        self.metrics_collector.record_gauge("system.memory.available_gb", memory.available / (1024**3))
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        self.metrics_collector.record_gauge("system.cpu.usage_percent", cpu_percent)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        self.metrics_collector.record_gauge("system.disk.usage_percent", disk_usage_percent)
        self.metrics_collector.record_gauge("system.disk.free_gb", disk.free / (1024**3))
    
    @contextmanager
    def monitor_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for monitoring operations."""
        start_time = time.time()
        
        try:
            yield
            # Record success
            self.metrics_collector.record_counter(f"operation.{operation_name}.success", tags=tags)
        except Exception as e:
            # Record failure
            self.metrics_collector.record_counter(f"operation.{operation_name}.failure", tags=tags)
            raise
        finally:
            # Record duration
            duration = (time.time() - start_time) * 1000
            self.metrics_collector.record_histogram(f"operation.{operation_name}.duration_ms", duration, tags=tags)
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthStatus]):
        """Register a health check function."""
        self.health_checker.register_check(name, check_func)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_results = self.health_checker.run_all_checks()
        overall_status = self.health_checker.get_overall_status()
        active_alerts = self.alert_manager.get_active_alerts()
        metrics_summary = self.metrics_collector.get_all_metrics()
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "health_checks": {name: {
                "status": result.status,
                "message": result.message,
                "response_time_ms": result.response_time_ms,
                "details": result.details
            } for name, result in health_results.items()},
            "active_alerts": [{
                "id": alert.id,
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "component": alert.component,
                "timestamp": alert.timestamp.isoformat()
            } for alert in active_alerts],
            "metrics": metrics_summary
        }
    
    def export_status_report(self, file_path: Union[str, Path]) -> None:
        """Export system status to file."""
        status = self.get_system_status()
        
        with open(file_path, 'w') as f:
            json.dump(status, f, indent=2, default=str)
        
        self.logger.info(f"System status exported to {file_path}")


# Global monitoring instance
default_system_monitor = SystemMonitor()


def get_system_monitor() -> SystemMonitor:
    """Get the default system monitor instance."""
    return default_system_monitor


def start_monitoring():
    """Start system monitoring."""
    default_system_monitor.start_monitoring()


def stop_monitoring():
    """Stop system monitoring."""
    default_system_monitor.stop_monitoring_service()


def get_health_status() -> Dict[str, Any]:
    """Get current system health status."""
    return default_system_monitor.get_system_status()