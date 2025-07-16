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


class MetricDequeWrapper:
    """Wrapper for deque that supports extend method for test compatibility."""
    
    def __init__(self, deque_obj, container, key):
        self._deque = deque_obj
        self._container = container
        self._key = key
    
    def extend(self, items):
        """Extend the deque with items."""
        self._container.extend_key(self._key, items)
    
    def append(self, item):
        """Append an item to the deque."""
        self._container.append_to_key(self._key, item)
    
    def __iter__(self):
        """Make wrapper iterable."""
        return iter(self._deque)
    
    def __len__(self):
        """Return length of deque."""
        return len(self._deque)
    
    def __getitem__(self, index):
        """Support indexing."""
        return self._deque[index]
    
    def clear(self):
        """Clear the deque."""
        self._deque.clear()


class MetricsContainer:
    """Custom container for metrics that supports both indexed and keyed access."""
    
    def __init__(self, max_points: int = 10000):
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
    
    def __getitem__(self, key):
        """Allow indexed/keyed access to metrics for test compatibility."""
        if isinstance(key, int):
            # Index access: collector.metrics[0]
            all_metrics = []
            for metric_deque in self._metrics.values():
                all_metrics.extend(list(metric_deque))
            return all_metrics[key]
        else:
            # Key access: collector.metrics["cpu_usage"]
            # First try exact key match
            if key in self._metrics:
                return MetricDequeWrapper(self._metrics[key], self, key)
            
            # If no exact match, look for metrics with this name (ignoring labels)
            matching_metrics = deque()
            found_match = False
            for metric_key, metric_deque in self._metrics.items():
                # Extract metric name from key (before any '[' for labels)
                metric_name = metric_key.split('[')[0] if '[' in metric_key else metric_key
                if metric_name == key:
                    matching_metrics.extend(metric_deque)
                    found_match = True
            
            # If no matches found, create a new entry and return wrapper
            if not found_match:
                # This will create a new deque in the defaultdict
                return MetricDequeWrapper(self._metrics[key], self, key)
            
            return matching_metrics
    
    def __len__(self) -> int:
        """Return total number of metric points."""
        return sum(len(deque) for deque in self._metrics.values())
    
    def __iter__(self):
        """Make container iterable for test compatibility."""
        all_metrics = []
        for metric_deque in self._metrics.values():
            all_metrics.extend(list(metric_deque))
        return iter(all_metrics)
    
    def keys(self):
        """Return metric names for test compatibility."""
        return self._metrics.keys()
    
    def values(self):
        """Return metric deques."""
        return self._metrics.values()
    
    def items(self):
        """Return metric items."""
        return self._metrics.items()
    
    def clear(self):
        """Clear all metrics."""
        self._metrics.clear()
    
    def __contains__(self, key):
        """Check if key exists."""
        return key in self._metrics
    
    def append_to_key(self, key: str, metric_point):
        """Append a metric point to a specific key."""
        self._metrics[key].append(metric_point)
    
    def extend_key(self, key: str, metric_points):
        """Extend a specific key with multiple metric points."""
        self._metrics[key].extend(metric_points)


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
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    
    @property
    def tags(self) -> Dict[str, str]:
        """Backward compatibility property for labels."""
        return self.labels


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
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Keep for backward compatibility


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
            is_healthy=(status == "healthy"),
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
            is_healthy=(status == "healthy"),
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
            is_healthy=(status == "healthy"),
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
            # Check if main modules can be imported (lightweight check)
            try:
                from src.chunking_system import DocumentChunker
                from src.utils.cache import CacheManager
                
                # For performance testing, just check if classes can be imported
                # without actually instantiating them (which is slow)
                import os
                if os.getenv('CHUNKING_SYSTEM_LIGHTWEIGHT_HEALTH_CHECK', '').lower() == 'true':
                    response_time = (time.time() - start_time) * 1000
                    return HealthStatus(
                        component="application_status",
                        status="healthy",
                        is_healthy=True,
                        message="Application components importable (lightweight check)",
                        timestamp=datetime.now(),
                        response_time_ms=response_time,
                        details={
                            "chunker_available": True,
                            "cache_available": True,
                            "startup_time_ms": response_time,
                            "lightweight_check": True
                        }
                    )
                
                # Normal full functionality test
                chunker = DocumentChunker()
                cache_manager = CacheManager()
                
                response_time = (time.time() - start_time) * 1000
                
                return HealthStatus(
                    component="application_status",
                    status="healthy",
                    is_healthy=True,
                    message="Application components operational",
                    timestamp=datetime.now(),
                    response_time_ms=response_time,
                    details={
                        "chunker_available": True,
                        "cache_available": True,
                        "startup_time_ms": response_time,
                        "lightweight_check": False
                    }
                )
            except ImportError as ie:
                response_time = (time.time() - start_time) * 1000
                return HealthStatus(
                    component="application_status",
                    status="unhealthy",
                    is_healthy=False,
                    message=f"Application import error: {str(ie)}",
                    timestamp=datetime.now(),
                    response_time_ms=response_time,
                    details={"import_error": str(ie)}
                )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                component="application_status",
                status="unhealthy",
                is_healthy=False,
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
                is_healthy=False,
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
                is_healthy=False,
                message=f"Health check exception: {str(e)}",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
            self.last_results[check_name] = result
            return result
    
    def run_all_checks(self) -> List[HealthStatus]:
        """Run all registered health checks."""
        results = []
        for check_name in self.checks:
            results.append(self.run_check(check_name))
        return results
    
    def get_overall_status(self) -> str:
        """Get overall system health status."""
        results = self.run_all_checks()
        
        if any(result.status == "unhealthy" for result in results):
            return "unhealthy"
        elif any(result.status == "degraded" for result in results):
            return "degraded"
        else:
            return "healthy"
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health as HealthStatus object."""
        results = self.run_all_checks()
        overall_status = self.get_overall_status()
        
        healthy_count = sum(1 for r in results if r.is_healthy)
        total_count = len(results)
        
        message = f"System health: {healthy_count}/{total_count} components healthy"
        
        return HealthStatus(
            component="overall_system",
            is_healthy=(overall_status == "healthy"),
            message=message,
            timestamp=datetime.now(),
            details={
                "total_components": total_count,
                "healthy_components": healthy_count,
                "component_statuses": {r.component: r.status for r in results}
            }
        )


class MetricsCollector:
    """Metrics collection and aggregation."""
    
    def __init__(self, max_points: int = 10000):
        self.logger = get_logger(__name__)
        self.max_points = max_points  # Store max_points as instance attribute
        self.metrics = MetricsContainer(max_points)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None, unit: str = None):
        """Record a counter metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            
            # Determine unit from labels or parameter
            metric_unit = unit
            if not metric_unit and labels and "unit" in labels:
                metric_unit = labels["unit"]
            if not metric_unit:
                # Special case for cpu_usage
                if "cpu_usage" in name:
                    metric_unit = "percent"
                else:
                    metric_unit = "count"
            
            metric_point = MetricPoint(
                name=name,
                value=value,  # Store the individual value, not the accumulated counter
                timestamp=datetime.now(),
                labels=labels or {},
                unit=metric_unit
            )
            self.metrics.append_to_key(key, metric_point)
            
            # Global rotation if needed
            self._rotate_metrics()
    
    def _rotate_metrics(self):
        """Rotate metrics globally to maintain max_points limit."""
        total_metrics = len(self.metrics)
        
        if total_metrics > self.max_points:
            # Collect all metrics with timestamps
            all_metrics = []
            for key, metric_deque in self.metrics.items():
                for metric in metric_deque:
                    all_metrics.append((key, metric))
            
            # Sort by timestamp and keep only the most recent
            all_metrics.sort(key=lambda x: x[1].timestamp)
            recent_metrics = all_metrics[-self.max_points:]
            
            # Clear and rebuild metrics
            self.metrics.clear()
            for key, metric in recent_metrics:
                self.metrics.append_to_key(key, metric)
    
    def record_gauge(self, name: str, value: float, unit: str = None, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            
            # Determine unit from labels or parameter
            metric_unit = unit
            if not metric_unit and labels and "unit" in labels:
                metric_unit = labels["unit"]
            if not metric_unit:
                metric_unit = ""
            
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                unit=metric_unit
            )
            self.metrics.append_to_key(key, metric_point)
            
            # Global rotation if needed
            self._rotate_metrics()
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                unit="histogram"
            )
            self.metrics.append_to_key(key, metric_point)
            
            # Global rotation if needed
            self._rotate_metrics()
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"
    
    def get_metrics(self, name: str, labels: Optional[Dict[str, str]] = None) -> List[MetricPoint]:
        """Get all metric points for a given name."""
        key = self._make_key(name, labels)
        
        # Try exact key match first
        if key in self.metrics and self.metrics[key]:
            return list(self.metrics[key])
        else:
            # If no exact match and no labels provided, try name-only access
            if labels is None:
                matching_points = []
                for existing_key in self.metrics.keys():
                    if existing_key == name or existing_key.startswith(name + '['):
                        matching_points.extend(list(self.metrics[existing_key]))
                return matching_points
            else:
                return []
    
    def get_metric_summary(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        key = self._make_key(name, labels)
        
        # Try exact key match first
        if key in self.metrics and self.metrics[key]:
            points = list(self.metrics[key])
        else:
            # If no exact match and no labels provided, try name-only access
            if labels is None:
                matching_points = []
                for existing_key in self.metrics.keys():
                    if existing_key == name or existing_key.startswith(name + '['):
                        matching_points.extend(list(self.metrics[existing_key]))
                points = matching_points
            else:
                points = []
        
        if not points:
            return {"error": "No data available"}
        
        values = [point.value for point in points]
        
        if not values:
            return {"error": "No values available"}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values),
            "latest": values[-1],
            "latest_timestamp": points[-1].timestamp.isoformat()
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all metrics."""
        result = {}
        
        with self._lock:
            for key in self.metrics.keys():
                # Parse metric name from key
                if '[' in key:
                    name = key.split('[')[0]
                    label_str = key.split('[')[1].rstrip(']')
                    labels = dict(tag.split('=') for tag in label_str.split(',') if tag)
                else:
                    name = key
                    labels = None
                
                result[key] = self.get_metric_summary(name, labels)
        
        return result
    
    def clear_metrics(self, older_than: Optional[timedelta] = None):
        """Clear old metrics."""
        if older_than is None:
            older_than = timedelta(hours=24)
        
        cutoff_time = datetime.now() - older_than
        
        with self._lock:
            for key in list(self.metrics.keys()):
                # Filter out old points
                filtered_points = [point for point in self.metrics[key] if point.timestamp > cutoff_time]
                self.metrics[key].clear()
                for point in filtered_points:
                    self.metrics[key].append(point)
                
                # Remove empty metrics
                if not self.metrics[key]:
                    del self.metrics._metrics[key]


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
            details=metadata or {},  # Use details for main storage
            metadata=metadata or {}  # Keep for backward compatibility
        )
        
        with self._lock:
            self.alerts.append(alert)
            # All unresolved alerts are considered active
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
    
    def get_alerts_by_severity(self, severity: str) -> List[Alert]:
        """Get all alerts (active and resolved) by severity."""
        with self._lock:
            alerts = [alert for alert in self.alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alerts_by_component(self, component: str) -> List[Alert]:
        """Get all alerts (active and resolved) by component."""
        with self._lock:
            alerts = [alert for alert in self.alerts if alert.component == component]
        
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
            self._run_single_monitoring_cycle()
    
    def _run_single_monitoring_cycle(self):
        """Run a single monitoring cycle."""
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
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics in a structured format.
        
        Returns:
            Dictionary containing system metrics compatible with performance_stats() endpoint
        """
        try:
            # Get current system metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)  # Quick CPU check
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Get load average if available (Unix-like systems)
            load_average = None
            if hasattr(os, 'getloadavg'):
                try:
                    load_average = list(os.getloadavg())
                except (OSError, AttributeError):
                    load_average = [0.0, 0.0, 0.0]
            else:
                # Fallback for systems without getloadavg (like Windows)
                load_average = [0.0, 0.0, 0.0]
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk_usage_percent,
                "load_average": load_average,
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "disk_used_gb": disk.used / (1024**3),
                "cpu_count": psutil.cpu_count(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            # Return fallback values to prevent endpoint failures
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "disk_percent": 0.0,
                "load_average": [0.0, 0.0, 0.0],
                "memory_total_gb": 0.0,
                "memory_available_gb": 0.0,
                "memory_used_gb": 0.0,
                "disk_total_gb": 0.0,
                "disk_free_gb": 0.0,
                "disk_used_gb": 0.0,
                "cpu_count": 1,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    @contextmanager
    def monitor_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for monitoring operations."""
        start_time = time.time()
        
        try:
            yield
            # Record success
            self.metrics_collector.record_counter(f"operation.{operation_name}.success", labels=labels)
        except Exception as e:
            # Record failure
            self.metrics_collector.record_counter(f"operation.{operation_name}.failure", labels=labels)
            raise
        finally:
            # Record duration
            duration = (time.time() - start_time) * 1000
            self.metrics_collector.record_histogram(f"operation.{operation_name}.duration_ms", duration, labels=labels)
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthStatus]):
        """Register a health check function."""
        self.health_checker.register_check(name, check_func)
    
    def create_alert(self, severity: str, component: str, message: str, 
                    title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Create an alert through the alert manager."""
        if title is None:
            title = f"{component.title()} Alert"
        return self.alert_manager.create_alert(severity, title, message, component, metadata=metadata)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_results = self.health_checker.run_all_checks()
        overall_status = self.health_checker.get_overall_status()
        active_alerts = self.alert_manager.get_active_alerts()
        metrics_summary = self.metrics_collector.get_all_metrics()
        
        # Count alerts by severity
        alert_counts = {}
        for alert in self.alert_manager.alerts:
            severity = alert.severity
            alert_counts[severity] = alert_counts.get(severity, 0) + 1
        
        # Convert health results list to dictionary
        health_checks = {}
        for result in health_results:
            health_checks[result.component] = {
                "status": result.status,
                "message": result.message,
                "response_time_ms": result.response_time_ms,
                "details": result.details
            }
        
        return {
            "health": {
                "overall_healthy": overall_status == "healthy",
                "checks": health_checks
            },
            "metrics_count": len(metrics_summary),
            "active_alerts": len(active_alerts),
            "alert_counts": alert_counts,
            "timestamp": datetime.now().isoformat()
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