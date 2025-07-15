"""
Phase 4: Health Check and Monitoring Endpoints

This module provides HTTP endpoints for health checks, metrics, and monitoring
capabilities. Designed to be framework-agnostic and can be integrated with
Flask, FastAPI, or any other web framework.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import asdict
from http import HTTPStatus

from src.utils.observability import get_observability_manager, MetricType, ObservabilityManager
from src.utils.monitoring import SystemMonitor
from src.utils.logger import get_logger


class HealthEndpoint:
    """Health check endpoint handler."""
    
    def __init__(self, system_monitor: Optional[SystemMonitor] = None):
        self.logger = get_logger(__name__)
        self.system_monitor = system_monitor or SystemMonitor()
        self.observability = get_observability_manager()
        
    def health_check(self, component: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
        """
        Basic health check endpoint.
        
        Args:
            component: Optional component name to check specifically
            
        Returns:
            Tuple of (response_dict, http_status_code)
        """
        try:
            start_time = time.time()
            
            if component:
                # Check specific component
                result = self.system_monitor.health_checker.run_check(component)
                if not result:
                    return {
                        "status": "error",
                        "message": f"Component '{component}' not found",
                        "timestamp": datetime.now().isoformat()
                    }, HTTPStatus.NOT_FOUND
                
                response = {
                    "status": "healthy" if result.is_healthy else "unhealthy",
                    "component": result.component,
                    "message": result.message,
                    "details": result.details,
                    "timestamp": result.timestamp.isoformat(),
                    "response_time_ms": result.response_time_ms
                }
                
                status_code = HTTPStatus.OK if result.is_healthy else HTTPStatus.SERVICE_UNAVAILABLE
                
            else:
                # Overall health check
                overall_health = self.system_monitor.health_checker.get_overall_health()
                
                response = {
                    "status": "healthy" if overall_health.is_healthy else "unhealthy",
                    "message": overall_health.message,
                    "timestamp": overall_health.timestamp.isoformat(),
                    "response_time_ms": (time.time() - start_time) * 1000
                }
                
                status_code = HTTPStatus.OK if overall_health.is_healthy else HTTPStatus.SERVICE_UNAVAILABLE
            
            # Record metrics
            self.observability.record_metric(
                "health_check_requests_total",
                1,
                MetricType.COUNTER,
                "requests",
                {"component": component or "all", "status": response["status"]}
            )
            
            return response, status_code
            
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {
                "status": "error",
                "message": "Health check failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }, HTTPStatus.INTERNAL_SERVER_ERROR
    
    def detailed_health(self) -> Tuple[Dict[str, Any], int]:
        """
        Detailed health check with all components.
        
        Returns:
            Tuple of (response_dict, http_status_code)
        """
        try:
            start_time = time.time()
            
            # Get all health checks
            all_checks = self.system_monitor.health_checker.run_all_checks()
            overall_health = self.system_monitor.health_checker.get_overall_health()
            
            # Format component details
            components = {}
            for name, result in all_checks.items():
                if result:
                    components[name] = {
                        "status": "healthy" if result.is_healthy else "unhealthy",
                        "message": result.message,
                        "details": result.details,
                        "response_time_ms": result.response_time_ms,
                        "timestamp": result.timestamp.isoformat()
                    }
            
            response = {
                "status": "healthy" if overall_health.is_healthy else "unhealthy",
                "message": overall_health.message,
                "timestamp": overall_health.timestamp.isoformat(),
                "response_time_ms": (time.time() - start_time) * 1000,
                "components": components,
                "summary": {
                    "total_components": len(components),
                    "healthy_components": sum(1 for c in components.values() if c["status"] == "healthy"),
                    "unhealthy_components": sum(1 for c in components.values() if c["status"] == "unhealthy")
                }
            }
            
            status_code = HTTPStatus.OK if overall_health.is_healthy else HTTPStatus.SERVICE_UNAVAILABLE
            
            # Record metrics
            self.observability.record_metric(
                "detailed_health_requests_total",
                1,
                MetricType.COUNTER,
                "requests"
            )
            
            return response, status_code
            
        except Exception as e:
            self.logger.error("Detailed health check failed", error=str(e))
            return {
                "status": "error",
                "message": "Detailed health check failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }, HTTPStatus.INTERNAL_SERVER_ERROR
    
    def readiness_check(self) -> Tuple[Dict[str, Any], int]:
        """
        Readiness check - indicates if the service is ready to serve traffic.
        
        Returns:
            Tuple of (response_dict, http_status_code)
        """
        try:
            # Check critical components for readiness
            critical_checks = ["system_memory", "system_disk", "application_status"]
            
            ready = True
            failed_components = []
            
            for component in critical_checks:
                result = self.system_monitor.health_checker.run_check(component)
                if not result or not result.is_healthy:
                    ready = False
                    failed_components.append(component)
            
            response = {
                "ready": ready,
                "status": "ready" if ready else "not_ready",
                "message": "Service is ready" if ready else f"Service not ready: {', '.join(failed_components)}",
                "failed_components": failed_components,
                "timestamp": datetime.now().isoformat()
            }
            
            status_code = HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE
            
            # Record metrics
            self.observability.record_metric(
                "readiness_checks_total",
                1,
                MetricType.COUNTER,
                "requests",
                {"status": "ready" if ready else "not_ready"}
            )
            
            return response, status_code
            
        except Exception as e:
            self.logger.error("Readiness check failed", error=str(e))
            return {
                "ready": False,
                "status": "error",
                "message": "Readiness check failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }, HTTPStatus.INTERNAL_SERVER_ERROR
    
    def liveness_check(self) -> Tuple[Dict[str, Any], int]:
        """
        Liveness check - indicates if the service is alive and responsive.
        
        Returns:
            Tuple of (response_dict, http_status_code)
        """
        try:
            # Simple liveness check - if we can respond, we're alive
            response = {
                "alive": True,
                "status": "alive",
                "message": "Service is alive and responsive",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - self._get_start_time()
            }
            
            # Record metrics
            self.observability.record_metric(
                "liveness_checks_total",
                1,
                MetricType.COUNTER,
                "requests"
            )
            
            return response, HTTPStatus.OK
            
        except Exception as e:
            self.logger.error("Liveness check failed", error=str(e))
            return {
                "alive": False,
                "status": "error",
                "message": "Liveness check failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }, HTTPStatus.INTERNAL_SERVER_ERROR
    
    def _get_start_time(self) -> float:
        """Get application start time (mock implementation)."""
        # In a real application, this would track actual start time
        return time.time() - 3600  # Mock: started 1 hour ago


class MetricsEndpoint:
    """Metrics collection and export endpoints."""
    
    def __init__(self, system_monitor: Optional[SystemMonitor] = None):
        self.logger = get_logger(__name__)
        self.system_monitor = system_monitor or SystemMonitor()
        self.observability = get_observability_manager()
    
    def prometheus_metrics(self) -> Tuple[str, int]:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Tuple of (prometheus_text, http_status_code)
        """
        try:
            # Get Prometheus formatted metrics
            metrics_text = self.observability.export_prometheus_metrics()
            
            # Add additional system metrics
            system_status = self.system_monitor.get_system_status()
            
            additional_metrics = []
            
            # Add health check metrics
            if "health" in system_status:
                health_status = 1 if system_status["health"]["overall_healthy"] else 0
                additional_metrics.append(f"chunking_system_health_status {health_status}")
            
            # Add metrics count
            if "metrics_count" in system_status:
                additional_metrics.append(f"chunking_system_metrics_total {system_status['metrics_count']}")
            
            # Add active alerts
            if "active_alerts" in system_status:
                additional_metrics.append(f"chunking_system_active_alerts {system_status['active_alerts']}")
            
            # Combine metrics
            if additional_metrics:
                metrics_text += "\n" + "\n".join(additional_metrics)
            
            return metrics_text, HTTPStatus.OK
            
        except Exception as e:
            self.logger.error("Failed to export Prometheus metrics", error=str(e))
            return f"# Error exporting metrics: {str(e)}", HTTPStatus.INTERNAL_SERVER_ERROR
    
    def json_metrics(self) -> Tuple[Dict[str, Any], int]:
        """
        Export metrics in JSON format.
        
        Returns:
            Tuple of (metrics_dict, http_status_code)
        """
        try:
            # Get metrics summary
            metrics_summary = self.observability.get_metrics_summary()
            
            # Add system status
            system_status = self.system_monitor.get_system_status()
            
            response = {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics_summary.get("metrics", {}),
                "system_status": system_status,
                "meta": {
                    "export_format": "json",
                    "version": "1.0"
                }
            }
            
            return response, HTTPStatus.OK
            
        except Exception as e:
            self.logger.error("Failed to export JSON metrics", error=str(e))
            return {
                "error": "Failed to export metrics",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }, HTTPStatus.INTERNAL_SERVER_ERROR
    
    def metric_details(self, metric_name: str) -> Tuple[Dict[str, Any], int]:
        """
        Get details for a specific metric.
        
        Args:
            metric_name: Name of the metric to get details for
            
        Returns:
            Tuple of (metric_details, http_status_code)
        """
        try:
            # Get specific metric summary
            metric_summary = self.observability.metrics_registry.get_metric_summary(metric_name)
            
            if not metric_summary:
                return {
                    "error": "Metric not found",
                    "metric_name": metric_name,
                    "timestamp": datetime.now().isoformat()
                }, HTTPStatus.NOT_FOUND
            
            response = {
                "metric_name": metric_name,
                "summary": metric_summary,
                "timestamp": datetime.now().isoformat()
            }
            
            return response, HTTPStatus.OK
            
        except Exception as e:
            self.logger.error(f"Failed to get metric details for {metric_name}", error=str(e))
            return {
                "error": "Failed to get metric details",
                "metric_name": metric_name,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }, HTTPStatus.INTERNAL_SERVER_ERROR


class SystemStatusEndpoint:
    """System status and information endpoints."""
    
    def __init__(self, system_monitor: Optional[SystemMonitor] = None):
        self.logger = get_logger(__name__)
        self.system_monitor = system_monitor or SystemMonitor()
        self.observability = get_observability_manager()
    
    def system_info(self) -> Tuple[Dict[str, Any], int]:
        """
        Get comprehensive system information.
        
        Returns:
            Tuple of (system_info_dict, http_status_code)
        """
        try:
            import platform
            import sys
            
            # Get system status
            system_status = self.system_monitor.get_system_status()
            
            # Get health status
            health_status = self.observability.get_health_status()
            
            response = {
                "system": {
                    "platform": platform.platform(),
                    "python_version": sys.version,
                    "architecture": platform.architecture()[0],
                    "processor": platform.processor(),
                    "hostname": platform.node()
                },
                "application": {
                    "name": "Document Chunking System",
                    "version": "1.0.0",
                    "phase": "4",
                    "uptime_seconds": time.time() - self._get_start_time()
                },
                "health": health_status,
                "monitoring": system_status,
                "timestamp": datetime.now().isoformat()
            }
            
            return response, HTTPStatus.OK
            
        except Exception as e:
            self.logger.error("Failed to get system info", error=str(e))
            return {
                "error": "Failed to get system information",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }, HTTPStatus.INTERNAL_SERVER_ERROR
    
    def performance_stats(self) -> Tuple[Dict[str, Any], int]:
        """
        Get performance statistics.
        
        Returns:
            Tuple of (performance_stats, http_status_code)
        """
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get application metrics
            metrics_summary = self.observability.get_metrics_summary()
            
            response = {
                "system_resources": {
                    "cpu_percent": cpu_percent,
                    "memory": {
                        "total_gb": memory.total / (1024**3),
                        "available_gb": memory.available / (1024**3),
                        "used_percent": memory.percent
                    },
                    "disk": {
                        "total_gb": disk.total / (1024**3),
                        "free_gb": disk.free / (1024**3),
                        "used_percent": (disk.used / disk.total) * 100
                    }
                },
                "application_metrics": metrics_summary,
                "timestamp": datetime.now().isoformat()
            }
            
            return response, HTTPStatus.OK
            
        except Exception as e:
            self.logger.error("Failed to get performance stats", error=str(e))
            return {
                "error": "Failed to get performance statistics",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }, HTTPStatus.INTERNAL_SERVER_ERROR
    
    def _get_start_time(self) -> float:
        """Get application start time (mock implementation)."""
        # In a real application, this would track actual start time
        return time.time() - 3600  # Mock: started 1 hour ago


class EndpointRouter:
    """Simple router for health and monitoring endpoints."""
    
    def __init__(self, system_monitor: Optional[SystemMonitor] = None):
        self.health_endpoint = HealthEndpoint(system_monitor)
        self.metrics_endpoint = MetricsEndpoint(system_monitor)
        self.system_endpoint = SystemStatusEndpoint(system_monitor)
        
        # Define routes
        self.routes = {
            "GET /health": self.health_endpoint.health_check,
            "GET /health/detailed": self.health_endpoint.detailed_health,
            "GET /health/ready": self.health_endpoint.readiness_check,
            "GET /health/live": self.health_endpoint.liveness_check,
            "GET /metrics": self.metrics_endpoint.prometheus_metrics,
            "GET /metrics/json": self.metrics_endpoint.json_metrics,
            "GET /system/info": self.system_endpoint.system_info,
            "GET /system/performance": self.system_endpoint.performance_stats,
        }
    
    def handle_request(self, method: str, path: str, query_params: Optional[Dict[str, str]] = None) -> Tuple[Any, int]:
        """
        Handle incoming requests.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            query_params: Optional query parameters
            
        Returns:
            Tuple of (response, http_status_code)
        """
        route_key = f"{method.upper()} {path}"
        
        if route_key in self.routes:
            handler = self.routes[route_key]
            
            # Handle special cases with parameters
            if path == "/health" and query_params and "component" in query_params:
                return handler(component=query_params["component"])
            elif path.startswith("/metrics/") and path != "/metrics/json":
                # Handle metric details: /metrics/{metric_name}
                metric_name = path.split("/")[-1]
                return self.metrics_endpoint.metric_details(metric_name)
            else:
                return handler()
        else:
            return {
                "error": "Not Found",
                "message": f"Route {route_key} not found",
                "available_routes": list(self.routes.keys()),
                "timestamp": datetime.now().isoformat()
            }, HTTPStatus.NOT_FOUND
    
    def route_request(self, method: str, path: str, query_params: Optional[Dict[str, str]] = None) -> Tuple[Any, int]:
        """Route request (alias for handle_request)."""
        return self.handle_request(method, path, query_params)


# Convenience functions for framework integration

def create_flask_blueprint(system_monitor: Optional[SystemMonitor] = None):
    """
    Create a Flask blueprint with health and monitoring endpoints.
    
    Args:
        system_monitor: Optional SystemMonitor instance
        
    Returns:
        Flask Blueprint object
    """
    try:
        from flask import Blueprint, jsonify, request
        
        bp = Blueprint('monitoring', __name__)
        router = EndpointRouter(system_monitor)
        
        @bp.route('/health')
        def health():
            component = request.args.get('component')
            response, status_code = router.health_endpoint.health_check(component)
            return jsonify(response), status_code
        
        @bp.route('/health/detailed')
        def detailed_health():
            response, status_code = router.health_endpoint.detailed_health()
            return jsonify(response), status_code
        
        @bp.route('/health/ready')
        def readiness():
            response, status_code = router.health_endpoint.readiness_check()
            return jsonify(response), status_code
        
        @bp.route('/health/live')
        def liveness():
            response, status_code = router.health_endpoint.liveness_check()
            return jsonify(response), status_code
        
        @bp.route('/metrics')
        def metrics():
            response, status_code = router.metrics_endpoint.prometheus_metrics()
            return response, status_code, {'Content-Type': 'text/plain'}
        
        @bp.route('/metrics/json')
        def metrics_json():
            response, status_code = router.metrics_endpoint.json_metrics()
            return jsonify(response), status_code
        
        @bp.route('/metrics/<metric_name>')
        def metric_details(metric_name):
            response, status_code = router.metrics_endpoint.metric_details(metric_name)
            return jsonify(response), status_code
        
        @bp.route('/system/info')
        def system_info():
            response, status_code = router.system_endpoint.system_info()
            return jsonify(response), status_code
        
        @bp.route('/system/performance')
        def performance_stats():
            response, status_code = router.system_endpoint.performance_stats()
            return jsonify(response), status_code
        
        return bp
        
    except ImportError:
        raise ImportError("Flask is required to create Flask blueprint")


def create_fastapi_router(system_monitor: Optional[SystemMonitor] = None):
    """
    Create a FastAPI router with health and monitoring endpoints.
    
    Args:
        system_monitor: Optional SystemMonitor instance
        
    Returns:
        FastAPI APIRouter object
    """
    try:
        from fastapi import APIRouter, Query, HTTPException
        from fastapi.responses import PlainTextResponse
        
        api_router = APIRouter()
        router = EndpointRouter(system_monitor)
        
        @api_router.get("/health")
        async def health_check(component: Optional[str] = Query(None)):
            response, status_code = router.health_endpoint.health_check(component)
            if status_code != HTTPStatus.OK:
                raise HTTPException(status_code=status_code, detail=response)
            return response
        
        @api_router.get("/health/detailed")
        async def detailed_health():
            response, status_code = router.health_endpoint.detailed_health()
            if status_code != HTTPStatus.OK:
                raise HTTPException(status_code=status_code, detail=response)
            return response
        
        @api_router.get("/health/ready")
        async def readiness_check():
            response, status_code = router.health_endpoint.readiness_check()
            if status_code != HTTPStatus.OK:
                raise HTTPException(status_code=status_code, detail=response)
            return response
        
        @api_router.get("/health/live")
        async def liveness_check():
            response, status_code = router.health_endpoint.liveness_check()
            return response
        
        @api_router.get("/metrics", response_class=PlainTextResponse)
        async def prometheus_metrics():
            response, status_code = router.metrics_endpoint.prometheus_metrics()
            if status_code != HTTPStatus.OK:
                raise HTTPException(status_code=status_code, detail=response)
            return response
        
        @api_router.get("/metrics/json")
        async def metrics_json():
            response, status_code = router.metrics_endpoint.json_metrics()
            if status_code != HTTPStatus.OK:
                raise HTTPException(status_code=status_code, detail=response)
            return response
        
        @api_router.get("/metrics/{metric_name}")
        async def metric_details(metric_name: str):
            response, status_code = router.metrics_endpoint.metric_details(metric_name)
            if status_code != HTTPStatus.OK:
                raise HTTPException(status_code=status_code, detail=response)
            return response
        
        @api_router.get("/system/info")
        async def system_info():
            response, status_code = router.system_endpoint.system_info()
            return response
        
        @api_router.get("/system/performance")
        async def performance_stats():
            response, status_code = router.system_endpoint.performance_stats()
            return response
        
        return api_router
        
    except ImportError:
        raise ImportError("FastAPI is required to create FastAPI router")


# Simple HTTP server for standalone usage
def run_standalone_server(host: str = "localhost", port: int = 8000, system_monitor: Optional[SystemMonitor] = None):
    """
    Run a simple standalone HTTP server with health endpoints.
    
    Args:
        host: Host to bind to
        port: Port to bind to  
        system_monitor: Optional SystemMonitor instance
    """
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from urllib.parse import urlparse, parse_qs
        import json
        
        router = EndpointRouter(system_monitor)
        
        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed_path = urlparse(self.path)
                path = parsed_path.path
                query_params = {k: v[0] for k, v in parse_qs(parsed_path.query).items()}
                
                try:
                    response, status_code = router.handle_request("GET", path, query_params)
                    
                    self.send_response(status_code)
                    
                    if path == "/metrics":
                        self.send_header('Content-Type', 'text/plain')
                        self.end_headers()
                        self.wfile.write(response.encode('utf-8'))
                    else:
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
                        
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    error_response = {
                        "error": "Internal Server Error",
                        "message": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    self.wfile.write(json.dumps(error_response).encode('utf-8'))
            
            def log_message(self, format, *args):
                # Suppress default logging
                pass
        
        server = HTTPServer((host, port), HealthHandler)
        print(f"Health check server running on http://{host}:{port}")
        print("Available endpoints:")
        for route in router.routes.keys():
            print(f"  {route}")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down health check server...")
            server.shutdown()
            
    except ImportError as e:
        raise ImportError(f"Required dependency missing for standalone server: {e}")


if __name__ == "__main__":
    # Run standalone server for testing
    run_standalone_server()