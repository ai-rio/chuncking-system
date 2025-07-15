"""
Phase 4: Advanced Monitoring and Observability Infrastructure

This module implements enterprise-grade observability features including:
- Distributed tracing with correlation IDs
- Structured logging with context propagation
- Custom metrics collection and aggregation
- Health check endpoints with detailed diagnostics
- Real-time monitoring dashboards
"""

import uuid
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from enum import Enum
import logging
import psutil
import traceback
from pathlib import Path

from src.utils.logger import get_logger


class TraceLevel(Enum):
    """Trace severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Metric types for different data aggregations."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class TraceContext:
    """Distributed tracing context with correlation IDs."""
    operation: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "active"
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    def finish(self, status: str = "completed", error: Optional[str] = None):
        """Finish the trace span."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
        self.error = error
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to the trace context."""
        self.tags[key] = value
    
    def add_log(self, level: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Add a log entry to the trace context."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "data": data or {}
        }
        self.logs.append(log_entry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace context to dictionary."""
        result = asdict(self)
        if result.get('start_time'):
            result['start_time'] = result['start_time'].isoformat()
        if result.get('end_time'):
            result['end_time'] = result['end_time'].isoformat()
        return result


@dataclass
class CustomMetric:
    """Custom metric with metadata."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    help_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "help_text": self.help_text
        }


@dataclass
class HealthCheckResult:
    """Enhanced health check result with detailed diagnostics."""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['status'] = self.status.value  # Convert enum to string
        return result


class CorrelationIDManager:
    """Manages correlation IDs across thread boundaries."""
    
    _context: threading.local = threading.local()
    
    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get current correlation ID."""
        return getattr(cls._context, 'correlation_id', None)
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        """Set correlation ID for current thread."""
        cls._context.correlation_id = correlation_id
    
    @classmethod
    def generate_correlation_id(cls) -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4()).replace('-', '')  # Remove hyphens for compactness
    
    @classmethod
    def clear_correlation_id(cls):
        """Clear correlation ID for current thread."""
        if hasattr(cls._context, 'correlation_id'):
            delattr(cls._context, 'correlation_id')
    
    @classmethod
    @contextmanager
    def correlation_context(cls, correlation_id: Optional[str] = None):
        """Context manager for correlation ID."""
        old_id = cls.get_correlation_id()
        try:
            if correlation_id is None:
                correlation_id = cls.generate_correlation_id()
            cls.set_correlation_id(correlation_id)
            yield correlation_id
        finally:
            if old_id:
                cls.set_correlation_id(old_id)
            else:
                cls.clear_correlation_id()


class StructuredLogger:
    """Enhanced structured logger with correlation ID support."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.name = name
    
    def _format_message(self, message: str, **kwargs) -> Dict[str, Any]:
        """Format message with correlation ID and structured data."""
        log_data = {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "logger": self.name,
            "correlation_id": CorrelationIDManager.get_correlation_id(),
            **kwargs
        }
        return log_data
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        log_data = self._format_message(message, **kwargs)
        self.logger.debug(json.dumps(log_data))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        log_data = self._format_message(message, **kwargs)
        self.logger.info(json.dumps(log_data))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        log_data = self._format_message(message, **kwargs)
        self.logger.warning(json.dumps(log_data))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        log_data = self._format_message(message, **kwargs)
        if 'exc_info' not in kwargs:
            kwargs['traceback'] = traceback.format_exc()
        self.logger.error(json.dumps(log_data))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        log_data = self._format_message(message, **kwargs)
        self.logger.critical(json.dumps(log_data))
    
    @contextmanager
    def trace_operation(self, operation_name: str, **tags) -> TraceContext:
        """Context manager for tracing operations."""
        trace_ctx = TraceContext(operation_name=operation_name, tags=tags)
        
        self.info(f"Starting operation: {operation_name}", 
                 trace_id=trace_ctx.trace_id, 
                 span_id=trace_ctx.span_id, 
                 **tags)
        
        try:
            yield trace_ctx
            trace_ctx.finish("completed")
            self.info(f"Completed operation: {operation_name}", 
                     trace_id=trace_ctx.trace_id,
                     duration_ms=trace_ctx.duration_ms,
                     status=trace_ctx.status)
        except Exception as e:
            trace_ctx.finish("error", str(e))
            self.error(f"Failed operation: {operation_name}", 
                      trace_id=trace_ctx.trace_id,
                      error=str(e),
                      duration_ms=trace_ctx.duration_ms)
            raise


class MetricsRegistry:
    """Registry for custom metrics with aggregation capabilities."""
    
    def __init__(self):
        self.metrics: Dict[str, List[CustomMetric]] = {}
        self.lock = threading.Lock()
        self.aggregation_window = timedelta(minutes=5)
    
    def register_metric(self, metric: CustomMetric):
        """Register a new metric."""
        with self.lock:
            if metric.name not in self.metrics:
                self.metrics[metric.name] = []
            self.metrics[metric.name].append(metric)
            
            # Clean old metrics
            self._cleanup_old_metrics(metric.name)
    
    def _cleanup_old_metrics(self, metric_name: str):
        """Remove metrics older than aggregation window."""
        cutoff = datetime.now() - self.aggregation_window
        self.metrics[metric_name] = [
            m for m in self.metrics[metric_name] 
            if m.timestamp > cutoff
        ]
    
    def get_metric_summary(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get aggregated summary of a metric."""
        with self.lock:
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return None
            
            metrics = self.metrics[metric_name]
            values = [m.value for m in metrics]
            
            return {
                "name": metric_name,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "sum": sum(values),
                "latest": values[-1],
                "unit": metrics[-1].unit,
                "type": metrics[-1].metric_type.value,
                "window_minutes": self.aggregation_window.total_seconds() / 60
            }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metric summaries."""
        summaries = {}
        for metric_name in self.metrics:
            summary = self.get_metric_summary(metric_name)
            if summary:
                summaries[metric_name] = summary
        return summaries
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for metric_name, metric_list in self.metrics.items():
            if not metric_list:
                continue
                
            latest_metric = metric_list[-1]
            
            # Add help text
            if latest_metric.help_text:
                lines.append(f"# HELP {metric_name} {latest_metric.help_text}")
            
            # Add type
            lines.append(f"# TYPE {metric_name} {latest_metric.metric_type.value}")
            
            # Add metric with tags
            tag_string = ""
            if latest_metric.tags:
                tag_pairs = [f'{k}="{v}"' for k, v in latest_metric.tags.items()]
                tag_string = "{" + ",".join(tag_pairs) + "}"
            
            lines.append(f"{metric_name}{tag_string} {latest_metric.value}")
        
        return "\n".join(lines)


class HealthRegistry:
    """Registry for health checks with caching and dependencies."""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self.cache: Dict[str, HealthCheckResult] = {}
        self.cache_ttl = timedelta(seconds=30)
        self.dependencies: Dict[str, List[str]] = {}
        self.lock = threading.Lock()
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheckResult], 
                      dependencies: Optional[List[str]] = None):
        """Register a health check."""
        with self.lock:
            self.checks[name] = check_func
            self.dependencies[name] = dependencies or []
    
    def run_check(self, name: str, use_cache: bool = True) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        if name not in self.checks:
            return None
        
        # Check cache first
        if use_cache and name in self.cache:
            cached_result = self.cache[name]
            if datetime.now() - cached_result.timestamp < self.cache_ttl:
                return cached_result
        
        # Run the check
        start_time = time.time()
        try:
            result = self.checks[name]()
            response_time = (time.time() - start_time) * 1000
            result.response_time_ms = response_time
            result.dependencies = self.dependencies[name]
            
            # Cache the result
            with self.lock:
                self.cache[name] = result
            
            return result
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                component=name,
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                details={"error": str(e), "traceback": traceback.format_exc()},
                response_time_ms=response_time,
                dependencies=self.dependencies[name]
            )
            
            with self.lock:
                self.cache[name] = result
            
            return result
    
    def run_all_checks(self, use_cache: bool = True) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name, use_cache)
        return results
    
    def get_dependency_status(self, component: str) -> Dict[str, Any]:
        """Get status of component dependencies."""
        if component not in self.dependencies:
            return {}
        
        dep_status = {}
        for dep in self.dependencies[component]:
            result = self.run_check(dep)
            if result:
                dep_status[dep] = {
                    "status": result.status,
                    "healthy": result.is_healthy,
                    "message": result.message
                }
        
        return dep_status


class DashboardGenerator:
    """Generates monitoring dashboard configurations."""
    
    def __init__(self, metrics_registry: MetricsRegistry, health_registry: HealthRegistry):
        self.metrics_registry = metrics_registry
        self.health_registry = health_registry
    
    def generate_grafana_dashboard(self) -> Dict[str, Any]:
        """Generate Grafana dashboard configuration."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Document Chunking System - Observability",
                "tags": ["chunking", "monitoring", "phase4"],
                "timezone": "browser",
                "panels": [],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
        
        panel_id = 1
        
        # Add system metrics panels
        system_panels = self._create_system_panels(panel_id)
        dashboard["dashboard"]["panels"].extend(system_panels)
        panel_id += len(system_panels)
        
        # Add application metrics panels
        app_panels = self._create_application_panels(panel_id)
        dashboard["dashboard"]["panels"].extend(app_panels)
        panel_id += len(app_panels)
        
        # Add health check panels
        health_panels = self._create_health_panels(panel_id)
        dashboard["dashboard"]["panels"].extend(health_panels)
        
        return dashboard
    
    def _create_system_panels(self, start_id: int) -> List[Dict[str, Any]]:
        """Create system monitoring panels."""
        panels = [
            {
                "id": start_id,
                "title": "CPU Usage",
                "type": "stat",
                "targets": [{"expr": "system_cpu_percent"}],
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
            },
            {
                "id": start_id + 1,
                "title": "Memory Usage",
                "type": "stat",
                "targets": [{"expr": "system_memory_percent"}],
                "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
            },
            {
                "id": start_id + 2,
                "title": "Disk Usage",
                "type": "stat",
                "targets": [{"expr": "system_disk_percent"}],
                "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0}
            },
            {
                "id": start_id + 3,
                "title": "System Load",
                "type": "graph",
                "targets": [{"expr": "system_load_average"}],
                "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0}
            }
        ]
        return panels
    
    def _create_application_panels(self, start_id: int) -> List[Dict[str, Any]]:
        """Create application-specific panels."""
        panels = [
            {
                "id": start_id,
                "title": "Chunking Operations",
                "type": "graph",
                "targets": [{"expr": "chunking_operations_total"}],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
            },
            {
                "id": start_id + 1,
                "title": "Average Processing Time",
                "type": "graph",
                "targets": [{"expr": "avg(chunking_duration_ms)"}],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
            },
            {
                "id": start_id + 2,
                "title": "Error Rate",
                "type": "stat",
                "targets": [{"expr": "rate(chunking_errors_total[5m])"}],
                "gridPos": {"h": 8, "w": 8, "x": 0, "y": 16}
            },
            {
                "id": start_id + 3,
                "title": "Cache Hit Rate",
                "type": "stat",
                "targets": [{"expr": "cache_hit_rate"}],
                "gridPos": {"h": 8, "w": 8, "x": 8, "y": 16}
            },
            {
                "id": start_id + 4,
                "title": "Queue Size",
                "type": "stat",
                "targets": [{"expr": "processing_queue_size"}],
                "gridPos": {"h": 8, "w": 8, "x": 16, "y": 16}
            }
        ]
        return panels
    
    def _create_health_panels(self, start_id: int) -> List[Dict[str, Any]]:
        """Create health monitoring panels."""
        panels = [
            {
                "id": start_id,
                "title": "Component Health Status",
                "type": "table",
                "targets": [{"expr": "component_health_status"}],
                "gridPos": {"h": 8, "w": 24, "x": 0, "y": 24}
            }
        ]
        return panels
    
    def generate_prometheus_config(self) -> Dict[str, Any]:
        """Generate Prometheus scrape configuration."""
        config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "scrape_configs": [
                {
                    "job_name": "chunking-system",
                    "static_configs": [
                        {
                            "targets": ["localhost:8000"]  # Health endpoint
                        }
                    ],
                    "metrics_path": "/metrics",
                    "scrape_interval": "5s"
                }
            ],
            "rule_files": ["chunking_alerts.yml"]
        }
        return config
    
    def generate_alert_rules(self) -> Dict[str, Any]:
        """Generate Prometheus alert rules."""
        rules = {
            "groups": [
                {
                    "name": "chunking_system_alerts",
                    "rules": [
                        {
                            "alert": "HighErrorRate",
                            "expr": "rate(chunking_errors_total[5m]) > 0.1",
                            "for": "2m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "High error rate in chunking system",
                                "description": "Error rate is {{ $value }} errors/sec"
                            }
                        },
                        {
                            "alert": "HighMemoryUsage",
                            "expr": "system_memory_percent > 90",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "High memory usage",
                                "description": "Memory usage is {{ $value }}%"
                            }
                        },
                        {
                            "alert": "ComponentUnhealthy",
                            "expr": "component_health_status == 0",
                            "for": "1m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "Component health check failed",
                                "description": "Component {{ $labels.component }} is unhealthy"
                            }
                        }
                    ]
                }
            ]
        }
        return rules


class ObservabilityManager:
    """Central manager for all observability features."""
    
    def __init__(self):
        self.logger = StructuredLogger("observability_manager")
        self.metrics_registry = MetricsRegistry()
        self.health_registry = HealthRegistry()
        self.dashboard_generator = DashboardGenerator(
            self.metrics_registry, 
            self.health_registry
        )
        
        # Register default health checks
        self._register_default_health_checks()
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _register_default_health_checks(self):
        """Register default system health checks."""
        
        def cpu_health() -> HealthCheckResult:
            cpu_percent = psutil.cpu_percent(interval=1)
            status = "healthy" if cpu_percent < 80 else "degraded" if cpu_percent < 95 else "unhealthy"
            
            recommendations = []
            if cpu_percent > 80:
                recommendations.append("Consider reducing processing load")
            if cpu_percent > 95:
                recommendations.append("Immediate action required - system overloaded")
            
            return HealthCheckResult(
                component="cpu",
                status=status,
                message=f"CPU usage: {cpu_percent}%",
                details={"cpu_percent": cpu_percent},
                recommendations=recommendations
            )
        
        def memory_health() -> HealthCheckResult:
            memory = psutil.virtual_memory()
            status = "healthy" if memory.percent < 80 else "degraded" if memory.percent < 95 else "unhealthy"
            
            recommendations = []
            if memory.percent > 80:
                recommendations.append("Monitor memory usage closely")
            if memory.percent > 95:
                recommendations.append("Free up memory or scale resources")
            
            return HealthCheckResult(
                component="memory",
                status=status,
                message=f"Memory usage: {memory.percent}%",
                details={
                    "percent": memory.percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                },
                recommendations=recommendations
            )
        
        def disk_health() -> HealthCheckResult:
            disk = psutil.disk_usage('/')
            status = "healthy" if disk.percent < 80 else "degraded" if disk.percent < 95 else "unhealthy"
            
            recommendations = []
            if disk.percent > 80:
                recommendations.append("Clean up disk space")
            if disk.percent > 95:
                recommendations.append("Critical - expand storage immediately")
            
            return HealthCheckResult(
                component="disk",
                status=status,
                message=f"Disk usage: {disk.percent}%",
                details={
                    "percent": disk.percent,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3)
                },
                recommendations=recommendations
            )
        
        def process_health() -> HealthCheckResult:
            import os
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            return HealthCheckResult(
                component="process",
                status="healthy",
                message=f"Process healthy - CPU: {cpu_percent}%, Memory: {memory_info.rss/(1024**2):.1f}MB",
                details={
                    "pid": process.pid,
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_info.rss / (1024**2),
                    "threads": process.num_threads(),
                    "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
                }
            )
        
        self.health_registry.register_check("cpu", cpu_health)
        self.health_registry.register_check("memory", memory_health, dependencies=["cpu"])
        self.health_registry.register_check("disk", disk_health)
        self.health_registry.register_check("process", process_health, dependencies=["cpu", "memory"])
    
    def _start_background_monitoring(self):
        """Start background monitoring thread."""
        def monitor_loop():
            while True:
                try:
                    # Collect system metrics
                    self.record_metric("system.cpu_percent", psutil.cpu_percent(), MetricType.GAUGE, "percent")
                    
                    memory = psutil.virtual_memory()
                    self.record_metric("system.memory_percent", memory.percent, MetricType.GAUGE, "percent")
                    
                    disk = psutil.disk_usage('/')
                    self.record_metric("system.disk_percent", disk.percent, MetricType.GAUGE, "percent")
                    
                    # Load average (Unix systems only)
                    try:
                        load_avg = psutil.getloadavg()[0]  # 1-minute load average
                        self.record_metric("system.load_average", load_avg, MetricType.GAUGE, "load")
                    except AttributeError:
                        pass  # Windows doesn't have load average
                    
                    time.sleep(10)  # Collect every 10 seconds
                    
                except Exception as e:
                    self.logger.error("Background monitoring error", error=str(e))
                    time.sleep(30)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def record_metric(self, name: str, value: Union[int, float], 
                     metric_type: MetricType, unit: str, tags: Optional[Dict[str, str]] = None,
                     help_text: Optional[str] = None):
        """Record a custom metric."""
        metric = CustomMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            unit=unit,
            tags=tags or {},
            help_text=help_text
        )
        self.metrics_registry.register_metric(metric)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        all_checks = self.health_registry.run_all_checks()
        
        overall_healthy = all(check.is_healthy for check in all_checks.values() if check)
        overall_status = "healthy" if overall_healthy else "unhealthy"
        
        return {
            "overall_status": overall_status,
            "overall_healthy": overall_healthy,
            "components": {name: check.to_dict() for name, check in all_checks.items()},
            "timestamp": datetime.now().isoformat()
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "metrics": self.metrics_registry.get_all_metrics(),
            "timestamp": datetime.now().isoformat()
        }
    
    def export_prometheus_metrics(self) -> str:
        """Export all metrics in Prometheus format."""
        return self.metrics_registry.export_prometheus_format()
    
    def generate_dashboard_config(self, dashboard_type: str = "grafana") -> Dict[str, Any]:
        """Generate dashboard configuration."""
        if dashboard_type.lower() == "grafana":
            return self.dashboard_generator.generate_grafana_dashboard()
        else:
            raise ValueError(f"Unsupported dashboard type: {dashboard_type}")
    
    def get_trace_context(self) -> Optional[TraceContext]:
        """Get current trace context."""
        correlation_id = CorrelationIDManager.get_correlation_id()
        if correlation_id:
            return TraceContext(trace_id=correlation_id)
        return None
    
    @contextmanager
    def trace_operation(self, operation_name: str, **tags):
        """Create a traced operation context."""
        with CorrelationIDManager.correlation_context():
            with self.logger.trace_operation(operation_name, **tags) as trace_ctx:
                yield trace_ctx


# Global observability manager instance
_observability_manager = None


def get_observability_manager() -> ObservabilityManager:
    """Get global observability manager instance."""
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = ObservabilityManager()
    return _observability_manager


def record_metric(name: str, value: Union[int, float], metric_type: MetricType, 
                 unit: str, tags: Optional[Dict[str, str]] = None, help_text: Optional[str] = None):
    """Convenience function to record metrics."""
    manager = get_observability_manager()
    manager.record_metric(name, value, metric_type, unit, tags, help_text)


def get_structured_logger(name: str) -> StructuredLogger:
    """Get a structured logger with correlation ID support."""
    return StructuredLogger(name)


@contextmanager
def trace_operation(operation_name: str, **tags):
    """Convenience function for tracing operations."""
    manager = get_observability_manager()
    with manager.trace_operation(operation_name, **tags) as trace_ctx:
        yield trace_ctx