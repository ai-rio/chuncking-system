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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        
        # Handle both enum and string status values
        if hasattr(self.status, 'value'):
            result['status'] = self.status.value
        else:
            result['status'] = self.status
        
        # Handle datetime serialization
        if isinstance(result.get('timestamp'), datetime):
            result['timestamp'] = result['timestamp'].isoformat()
        
        # Filter sensitive data from details
        if 'details' in result and isinstance(result['details'], dict):
            sensitive_patterns = {
                'password', 'secret', 'key', 'token', 'credential', 'auth',
                'api_key', 'session_id', 'user_email', 'connection_string'
            }
            filtered_details = {}
            for key, value in result['details'].items():
                if any(pattern in key.lower() for pattern in sensitive_patterns):
                    filtered_details[key] = "[FILTERED]"
                elif isinstance(value, str) and any(pattern in value.lower() for pattern in ['password=', 'secret=', 'key=', 'token=']):
                    filtered_details[key] = "[FILTERED]"
                else:
                    filtered_details[key] = value
            result['details'] = filtered_details
        
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
    
    # Sensitive data patterns to filter
    SENSITIVE_PATTERNS = {
        'password', 'secret', 'key', 'token', 'credential', 'auth',
        'api_key', 'session_id', 'user_email', 'connection_string'
    }
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
        self.component = name
        self.correlation_manager = CorrelationIDManager()
    
    def _filter_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive data from log entries."""
        filtered_data = {}
        for key, value in data.items():
            if any(pattern in key.lower() for pattern in self.SENSITIVE_PATTERNS):
                filtered_data[key] = "[FILTERED]"
            elif isinstance(value, str) and any(pattern in value.lower() for pattern in ['password=', 'secret=', 'key=', 'token=']):
                filtered_data[key] = "[FILTERED]"
            else:
                filtered_data[key] = value
        return filtered_data
    
    def _serialize_value(self, value: Any) -> Any:
        """Safely serialize values for JSON output."""
        # Handle None values
        if value is None:
            return None
        
        # Handle basic types that are already JSON serializable
        if isinstance(value, (str, int, float, bool, list, dict)):
            return value
        
        # Handle Exception objects
        if isinstance(value, Exception):
            return {
                "type": type(value).__name__,
                "message": str(value),
                "args": list(value.args) if value.args else []
            }
        
        # Handle Mock objects (for testing)
        if hasattr(value, '_mock_name') or str(type(value)).find('Mock') != -1:
            return str(value)
        
        # Handle datetime objects
        if hasattr(value, 'isoformat'):
            return value.isoformat()
        
        # For any other object, convert to string
        return str(value)
    
    def _format_message(self, level: str, message: str, **kwargs) -> Dict[str, Any]:
        """Format message with correlation ID and structured data."""
        # Filter sensitive data from kwargs
        filtered_kwargs = self._filter_sensitive_data(kwargs)
        
        # Serialize all values to ensure JSON compatibility
        serialized_kwargs = {}
        for key, value in filtered_kwargs.items():
            serialized_kwargs[key] = self._serialize_value(value)
        
        log_data = {
            "level": level,
            "message": message,
            "component": self.component,
            "timestamp": datetime.now().isoformat(),
            "logger": self.name,
            "correlation_id": self._serialize_value(self.correlation_manager.get_correlation_id()),
            **serialized_kwargs
        }
        return log_data
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        log_data = self._format_message("DEBUG", message, **kwargs)
        self.logger.debug(json.dumps(log_data))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        log_data = self._format_message("INFO", message, **kwargs)
        self.logger.info(json.dumps(log_data))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        log_data = self._format_message("WARNING", message, **kwargs)
        self.logger.warning(json.dumps(log_data))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        if 'exc_info' not in kwargs:
            kwargs['traceback'] = traceback.format_exc()
        log_data = self._format_message("ERROR", message, **kwargs)
        self.logger.error(json.dumps(log_data))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        log_data = self._format_message("CRITICAL", message, **kwargs)
        self.logger.critical(json.dumps(log_data))
    
    @contextmanager
    def trace_operation(self, operation_name: str, **tags) -> TraceContext:
        """Context manager for tracing operations."""
        trace_id = CorrelationIDManager.generate_correlation_id()
        span_id = CorrelationIDManager.generate_correlation_id()
        trace_ctx = TraceContext(operation=operation_name, trace_id=trace_id, span_id=span_id)
        
        # Set tags after creation
        for key, value in tags.items():
            trace_ctx.add_tag(key, value)
        
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
        self.metrics: List[CustomMetric] = []
        self._metrics_dict: Dict[str, List[CustomMetric]] = {}
        self.lock = threading.Lock()
        self.aggregation_window = timedelta(minutes=5)
        # Performance optimization: only cleanup periodically
        self._registration_count = 0
        self._cleanup_interval = 100  # Run cleanup every 100 registrations
        self._last_cleanup_time = datetime.now()
    
    def register_metric(self, metric: CustomMetric):
        """Register a new metric."""
        with self.lock:
            self.metrics.append(metric)
            if metric.name not in self._metrics_dict:
                self._metrics_dict[metric.name] = []
            self._metrics_dict[metric.name].append(metric)
            
            # Increment registration counter
            self._registration_count += 1
            
            # Only run cleanup periodically to avoid O(n²) performance
            if (self._registration_count % self._cleanup_interval == 0 or 
                datetime.now() - self._last_cleanup_time > self.aggregation_window):
                self._cleanup_old_metrics_all()
                self._last_cleanup_time = datetime.now()
    
    def _cleanup_old_metrics(self, metric_name: str):
        """Remove metrics older than aggregation window for a specific metric name."""
        cutoff = datetime.now() - self.aggregation_window
        if metric_name in self._metrics_dict:
            self._metrics_dict[metric_name] = [
                m for m in self._metrics_dict[metric_name] 
                if m.timestamp > cutoff
            ]
    
    def _cleanup_old_metrics_all(self):
        """Remove metrics older than aggregation window for all metrics.
        
        This is called periodically instead of on every registration to avoid O(n²) performance.
        """
        cutoff = datetime.now() - self.aggregation_window
        
        # Clean per-metric dictionaries
        for metric_name in list(self._metrics_dict.keys()):
            self._metrics_dict[metric_name] = [
                m for m in self._metrics_dict[metric_name] 
                if m.timestamp > cutoff
            ]
            # Remove empty metric entries
            if not self._metrics_dict[metric_name]:
                del self._metrics_dict[metric_name]
        
        # Clean main metrics list
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff]
    
    def force_cleanup(self):
        """Force immediate cleanup of old metrics. Use sparingly for performance."""
        with self.lock:
            self._cleanup_old_metrics_all()
            self._last_cleanup_time = datetime.now()
    
    def get_metric_summary(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get aggregated summary of a metric."""
        with self.lock:
            if metric_name not in self._metrics_dict or not self._metrics_dict[metric_name]:
                return None
            
            metrics = self._metrics_dict[metric_name]
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
        for metric_name in self._metrics_dict:
            summary = self.get_metric_summary(metric_name)
            if summary:
                summaries[metric_name] = summary
        return summaries
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Sensitive label patterns to filter
        sensitive_key_patterns = {
            'password', 'secret', 'key', 'token', 'credential', 'auth',
            'api_key', 'session_id', 'user_email', 'user_token', 'email', 'user_id'
        }
        
        for metric_name, metric_list in self._metrics_dict.items():
            if not metric_list:
                continue
                
            latest_metric = metric_list[-1]
            
            # Add help text
            if latest_metric.help_text:
                lines.append(f"# HELP {metric_name} {latest_metric.help_text}")
            
            # Add type
            lines.append(f"# TYPE {metric_name} {latest_metric.metric_type.value}")
            
            # Add metric with filtered labels
            tag_string = ""
            if latest_metric.labels:
                # Filter out sensitive labels
                filtered_labels = {}
                for k, v in latest_metric.labels.items():
                    # Skip if key contains sensitive patterns
                    if any(pattern in k.lower() for pattern in sensitive_key_patterns):
                        continue
                    # Skip if value looks like an email address
                    if isinstance(v, str) and '@' in v and '.' in v:
                        continue
                    # Skip if value contains other sensitive patterns
                    if isinstance(v, str) and any(pattern in v.lower() for pattern in {'password', 'secret', 'token'}):
                        continue
                    filtered_labels[k] = v
                
                if filtered_labels:
                    tag_pairs = [f'{k}="{v}"' for k, v in filtered_labels.items()]
                    tag_string = "{" + ",".join(tag_pairs) + "}"
            
            lines.append(f"{metric_name}{tag_string} {latest_metric.value}")
        
        return "\n".join(lines)
    
    def record_metric(self, name: str, value: Union[int, float], metric_type: MetricType, 
                     unit: str = "units", labels: Optional[Dict[str, str]] = None,
                     help_text: Optional[str] = None):
        """Record a metric (alias for register_metric)."""
        metric = CustomMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            unit=unit,
            labels=labels or {},
            help_text=help_text
        )
        self.register_metric(metric)
    
    def get_metrics_by_name(self, name: str) -> List[CustomMetric]:
        """Get all metrics by name."""
        with self.lock:
            return self._metrics_dict.get(name, [])
    
    def get_metrics_by_type(self, metric_type: MetricType) -> List[CustomMetric]:
        """Get all metrics by type."""
        with self.lock:
            result = []
            for metric_list in self._metrics_dict.values():
                for metric in metric_list:
                    if metric.metric_type == metric_type:
                        result.append(metric)
            return result
    
    def export_all_data(self) -> Dict[str, Any]:
        """Export all metrics data."""
        with self.lock:
            # Sensitive label patterns to filter
            sensitive_key_patterns = {
                'password', 'secret', 'key', 'token', 'credential', 'auth',
                'api_key', 'session_id', 'user_email', 'user_token', 'email', 'user_id'
            }
            
            # Convert metrics to serializable format as a flat list
            metrics_list = []
            for name, metrics in self._metrics_dict.items():
                for metric in metrics:
                    # Filter out sensitive labels
                    filtered_labels = {}
                    if metric.labels:
                        for k, v in metric.labels.items():
                            # Skip if key contains sensitive patterns
                            if any(pattern in k.lower() for pattern in sensitive_key_patterns):
                                continue
                            # Skip if value looks like an email address
                            if isinstance(v, str) and '@' in v and '.' in v:
                                continue
                            # Skip if value contains other sensitive patterns
                            if isinstance(v, str) and any(pattern in v.lower() for pattern in {'password', 'secret', 'token'}):
                                continue
                            filtered_labels[k] = v
                    
                    metric_data = {
                        "name": metric.name,
                        "value": metric.value,
                        "type": metric.metric_type.value if hasattr(metric.metric_type, 'value') else str(metric.metric_type),
                        "unit": metric.unit,
                        "timestamp": metric.timestamp.isoformat() if hasattr(metric.timestamp, 'isoformat') else str(metric.timestamp),
                        "labels": filtered_labels,
                        "help_text": metric.help_text
                    }
                    metrics_list.append(metric_data)
            
            return {
                "metrics": metrics_list,
                "export_time": datetime.now().isoformat()
            }
    
    def clear_metrics(self):
        """Clear all metrics."""
        with self.lock:
            self.metrics.clear()
            self._metrics_dict.clear()


class HealthRegistry:
    """Registry for health checks with caching and dependencies."""
    
    def __init__(self, max_concurrent_checks: int = 10, check_timeout_seconds: float = 30.0):
        self.health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self.checks: Dict[str, Callable[[], HealthCheckResult]] = {}  # Keep for backward compatibility
        self.cache: Dict[str, HealthCheckResult] = {}
        self.cache_ttl = timedelta(seconds=30)
        self.dependencies: Dict[str, List[str]] = {}
        self.lock = threading.Lock()
        self.max_concurrent_checks = max_concurrent_checks
        self.check_timeout_seconds = check_timeout_seconds
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheckResult], 
                      dependencies: Optional[List[str]] = None):
        """Register a health check."""
        with self.lock:
            self.health_checks[name] = check_func
            self.checks[name] = check_func  # Keep for backward compatibility
            self.dependencies[name] = dependencies or []
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthCheckResult], 
                             dependencies: Optional[List[str]] = None):
        """Register a health check (alias for register_check)."""
        self.register_check(name, check_func, dependencies)
    
    def run_check(self, name: str, use_cache: bool = True) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        if name not in self.health_checks:
            return None
        
        # Check cache first
        if use_cache and name in self.cache:
            cached_result = self.cache[name]
            if datetime.now() - cached_result.timestamp < self.cache_ttl:
                return cached_result
        
        # Run the check
        start_time = time.time()
        try:
            result = self.health_checks[name]()
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
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e), "traceback": traceback.format_exc()},
                response_time_ms=response_time,
                dependencies=self.dependencies[name]
            )
            
            with self.lock:
                self.cache[name] = result
            
            return result
    
    def _execute_check_with_timeout(self, name: str, use_cache: bool = True) -> tuple[str, Optional[HealthCheckResult]]:
        """Execute a single health check with timeout handling.
        
        Returns a tuple of (check_name, result) to handle futures properly.
        """
        try:
            result = self.run_check(name, use_cache)
            return name, result
        except Exception as e:
            # Create an error result for unexpected exceptions
            error_result = HealthCheckResult(
                component=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check execution failed: {str(e)}",
                details={"error": str(e), "traceback": traceback.format_exc()},
                dependencies=self.dependencies.get(name, [])
            )
            return name, error_result
    
    def run_all_checks(self, use_cache: bool = True) -> Dict[str, HealthCheckResult]:
        """Run all health checks in parallel with concurrency limits and timeout handling."""
        if not self.health_checks:
            return {}
        
        results = {}
        
        # Determine the optimal number of workers (don't exceed the limit or the number of checks)
        # Ensure we have at least 1 worker
        max_workers = max(1, min(self.max_concurrent_checks, len(self.health_checks)))
        
        # If we only have one check, run it directly to avoid overhead
        if len(self.health_checks) == 1:
            name = next(iter(self.health_checks))
            result = self.run_check(name, use_cache)
            if result:
                results[name] = result
            return results
        
        # Run health checks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="health-check") as executor:
            # Submit all health checks
            future_to_name = {
                executor.submit(self._execute_check_with_timeout, name, use_cache): name
                for name in self.health_checks
            }
            
            # Collect results with timeout handling
            for future in as_completed(future_to_name, timeout=self.check_timeout_seconds):
                try:
                    name, result = future.result(timeout=self.check_timeout_seconds)
                    if result:
                        results[name] = result
                except Exception as e:
                    # Handle timeout or other execution failures
                    check_name = future_to_name[future]
                    error_result = HealthCheckResult(
                        component=check_name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check timed out or failed: {str(e)}",
                        details={
                            "error": str(e),
                            "timeout_seconds": self.check_timeout_seconds,
                            "traceback": traceback.format_exc()
                        },
                        dependencies=self.dependencies.get(check_name, [])
                    )
                    results[check_name] = error_result
        
        return results
    
    def run_all_health_checks(self, use_cache: bool = True) -> Dict[str, HealthCheckResult]:
        """Run all health checks (alias for backward compatibility)."""
        return self.run_all_checks(use_cache)
    
    def run_health_check(self, name: str, use_cache: bool = True) -> Optional[HealthCheckResult]:
        """Run a specific health check (alias for backward compatibility)."""
        return self.run_check(name, use_cache)
    
    def get_overall_health_status(self) -> HealthStatus:
        """Get overall system health status."""
        results = self.run_all_checks()
        
        if not results:
            return HealthStatus.HEALTHY
        
        has_degraded = False
        
        # Check status of all components
        for result in results.values():
            if result and hasattr(result, 'status'):
                # Check for explicitly unhealthy components
                if result.status == HealthStatus.UNHEALTHY or result.status == "unhealthy":
                    return HealthStatus.UNHEALTHY
                # Track if we have any degraded components
                elif result.status == HealthStatus.DEGRADED or result.status == "degraded":
                    has_degraded = True
        
        # Return degraded if we found any degraded components
        if has_degraded:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
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
                "title": "Document Chunking System - Phase 4 Observability",
                "tags": ["chunking", "monitoring", "phase4", "observability"],
                "style": "dark",
                "timezone": "browser",
                "editable": True,
                "graphTooltip": 0,
                "panels": [],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "timepicker": {
                    "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"]
                },
                "refresh": "5s",
                "schemaVersion": 30,
                "version": 1
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
        
        # Add system health status panel at the beginning
        system_health_panel = {
            "id": 0,
            "title": "System Health Status",
            "type": "stat",
            "targets": [{"expr": "component_health_status", "refId": "A"}],
            "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "yellow", "value": 0.5},
                            {"color": "green", "value": 0.8}
                        ]
                    }
                }
            },
            "options": {
                "colorMode": "background"
            }
        }
        dashboard["dashboard"]["panels"].insert(0, system_health_panel)
        
        return dashboard
    
    def _create_system_panels(self, start_id: int) -> List[Dict[str, Any]]:
        """Create system monitoring panels."""
        panels = [
            {
                "id": start_id,
                "title": "CPU Usage",
                "type": "stat",
                "targets": [{"expr": "system_cpu_percent", "refId": "A"}],
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 70},
                                {"color": "red", "value": 90}
                            ]
                        },
                        "unit": "percent"
                    }
                },
                "options": {
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "auto",
                    "orientation": "auto"
                }
            },
            {
                "id": start_id + 1,
                "title": "Memory Usage",
                "type": "stat",
                "targets": [{"expr": "system_memory_percent", "refId": "A"}],
                "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 70},
                                {"color": "red", "value": 90}
                            ]
                        },
                        "unit": "percent"
                    }
                },
                "options": {
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "auto",
                    "orientation": "auto"
                }
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
                "title": "Chunking Operations Rate",
                "type": "graph",
                "targets": [{"expr": "rate(chunking_operations_total[5m])", "refId": "A"}],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"}
                    }
                },
                "options": {
                    "legend": {"displayMode": "table", "values": ["last", "max"]}
                }
            },
            {
                "id": start_id + 1,
                "title": "Processing Duration",
                "type": "graph",
                "targets": [
                    {"expr": "histogram_quantile(0.95, chunking_duration_ms)", "refId": "A", "legendFormat": "95th percentile"},
                    {"expr": "histogram_quantile(0.50, chunking_duration_ms)", "refId": "B", "legendFormat": "50th percentile"}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "ms"
                    }
                },
                "options": {
                    "legend": {"displayMode": "table", "values": ["last", "max"]}
                }
            },
            {
                "id": start_id + 2,
                "title": "Error Rate",
                "type": "stat",
                "targets": [{"expr": "rate(chunking_errors_total[5m])", "refId": "A"}],
                "gridPos": {"h": 8, "w": 8, "x": 0, "y": 16},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 0.01},
                                {"color": "red", "value": 0.05}
                            ]
                        },
                        "unit": "reqps"
                    }
                },
                "options": {
                    "colorMode": "value"
                }
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
                },
                {
                    "job_name": "chunking-system-health",
                    "static_configs": [
                        {
                            "targets": ["localhost:8000"]
                        }
                    ],
                    "metrics_path": "/health",
                    "scrape_interval": "30s"
                }
            ],
            "rule_files": ["prometheus-alerts.yml"],
            "alerting": {
                "alertmanagers": [
                    {
                        "static_configs": [
                            {
                                "targets": ["localhost:9093"]
                            }
                        ],
                        "timeout": "10s",
                        "api_version": "v1",
                        "path_prefix": "/"
                    }
                ]
            },
            "external_labels": {
                "cluster": "chunking-system",
                "environment": "production",
                "service": "document-chunking"
            }
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
                            "alert": "SystemHealthDown",
                            "expr": "component_health_status == 0",
                            "for": "30s",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "System health check failed",
                                "description": "System component {{ $labels.component }} is down"
                            }
                        },
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
                            "alert": "ProcessingLatencyHigh",
                            "expr": "histogram_quantile(0.95, chunking_duration_ms) > 5000",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "High processing latency detected",
                                "description": "95th percentile latency is {{ $value }}ms"
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
                },
                {
                    "name": "chunking_system_sla",
                    "rules": [
                        {
                            "alert": "SLAErrorRateBreach",
                            "expr": "rate(chunking_errors_total[1h]) > 0.05",
                            "for": "15m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "SLA error rate breach",
                                "description": "Hourly error rate {{ $value }} exceeds SLA threshold of 5%"
                            }
                        },
                        {
                            "alert": "SLALatencyBreach",
                            "expr": "histogram_quantile(0.95, chunking_duration_ms) > 3000",
                            "for": "15m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "SLA latency breach",
                                "description": "95th percentile latency {{ $value }}ms exceeds SLA threshold"
                            }
                        },
                        {
                            "alert": "SLAAvailabilityBreach",
                            "expr": "avg_over_time(component_health_status[1h]) < 0.99",
                            "for": "5m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "SLA availability breach",
                                "description": "Hourly availability {{ $value }} is below 99% SLA"
                            }
                        }
                    ]
                },
                {
                    "name": "security_alerts",
                    "rules": [
                        {
                            "alert": "SecurityViolationDetected",
                            "expr": "rate(security_violations_total[5m]) > 0",
                            "for": "1m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "Security violation detected",
                                "description": "Security violations detected at rate {{ $value }}/sec"
                            }
                        },
                        {
                            "alert": "UnauthorizedAccessAttempt",
                            "expr": "rate(failed_auth_attempts[5m]) > 5",
                            "for": "2m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "High rate of unauthorized access attempts",
                                "description": "Failed authentication attempts at {{ $value }}/sec"
                            }
                        },
                        {
                            "alert": "SuspiciousActivityDetected",
                            "expr": "rate(security_event_suspicious_activity[5m]) > 0",
                            "for": "30s",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "Suspicious activity detected",
                                "description": "Suspicious security events detected"
                            }
                        },
                        {
                            "alert": "DataExfiltrationAttempt",
                            "expr": "rate(security_event_data_exfiltration_attempt[5m]) > 0",
                            "for": "1m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "Data exfiltration attempt detected",
                                "description": "Potential data breach detected"
                            }
                        }
                    ]
                },
                {
                    "name": "chunking_system_recording_rules",
                    "rules": [
                        {
                            "record": "chunking:error_rate_5m",
                            "expr": "rate(chunking_errors_total[5m])"
                        },
                        {
                            "record": "chunking:throughput_5m",
                            "expr": "rate(chunking_operations_total[5m])"
                        },
                        {
                            "record": "chunking:latency_p95_5m",
                            "expr": "histogram_quantile(0.95, rate(chunking_duration_ms_bucket[5m]))"
                        },
                        {
                            "record": "system:cpu_utilization_avg",
                            "expr": "avg(system_cpu_percent)"
                        },
                        {
                            "record": "system:memory_utilization_avg",
                            "expr": "avg(system_memory_percent)"
                        }
                    ]
                }
            ]
        }
        return rules


class ObservabilityManager:
    """Central manager for all observability features."""
    
    def __init__(self, max_concurrent_health_checks: int = 10, health_check_timeout: float = 30.0):
        self.logger = StructuredLogger("observability_manager")
        self.metrics_registry = MetricsRegistry()
        self.health_registry = HealthRegistry(
            max_concurrent_checks=max_concurrent_health_checks,
            check_timeout_seconds=health_check_timeout
        )
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
            status = HealthStatus.HEALTHY if cpu_percent < 80 else HealthStatus.DEGRADED if cpu_percent < 95 else HealthStatus.UNHEALTHY
            
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
            status = HealthStatus.HEALTHY if memory.percent < 80 else HealthStatus.DEGRADED if memory.percent < 95 else HealthStatus.UNHEALTHY
            
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
            status = HealthStatus.HEALTHY if disk.percent < 80 else HealthStatus.DEGRADED if disk.percent < 95 else HealthStatus.UNHEALTHY
            
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
                status=HealthStatus.HEALTHY,
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
                     metric_type: MetricType, unit: str = "units", labels: Optional[Dict[str, str]] = None,
                     help_text: Optional[str] = None):
        """Record a custom metric."""
        metric = CustomMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            unit=unit,
            labels=labels or {},
            help_text=help_text
        )
        self.metrics_registry.register_metric(metric)
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthCheckResult], 
                             dependencies: Optional[List[str]] = None):
        """Register a health check function."""
        self.health_registry.register_check(name, check_func, dependencies)
    
    def run_health_check(self, name: str, use_cache: bool = True) -> Optional[HealthCheckResult]:
        """Run a health check."""
        return self.health_registry.run_check(name, use_cache)
    
    def export_all_data(self, include_system_metrics: bool = True) -> Dict[str, Any]:
        """Export all observability data."""
        # Use MetricsRegistry's export method which includes sensitive data filtering
        metrics_data = self.metrics_registry.export_all_data()
        
        # Filter out system metrics if requested (useful for testing)
        if not include_system_metrics:
            filtered_metrics = [
                metric for metric in metrics_data["metrics"] 
                if not metric["name"].startswith("system.")
            ]
            metrics_data["metrics"] = filtered_metrics
        
        return {
            "metrics": {
                "metrics": metrics_data["metrics"],
                "export_time": metrics_data["export_time"]
            },
            "health_checks": self.get_health_status(),
            "prometheus_format": self.metrics_registry.export_prometheus_format(),
            "system_info": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        }
    
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
            span_id = CorrelationIDManager.generate_correlation_id()
            return TraceContext(operation="get_trace_context", trace_id=correlation_id, span_id=span_id)
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
                 unit: str, labels: Optional[Dict[str, str]] = None, help_text: Optional[str] = None):
    """Convenience function to record metrics."""
    manager = get_observability_manager()
    manager.record_metric(name, value, metric_type, unit, labels, help_text)


def get_structured_logger(name: str) -> StructuredLogger:
    """Get a structured logger with correlation ID support."""
    return StructuredLogger(name)


@contextmanager
def trace_operation(operation_name: str, **tags):
    """Convenience function for tracing operations."""
    manager = get_observability_manager()
    with manager.trace_operation(operation_name, **tags) as trace_ctx:
        yield trace_ctx