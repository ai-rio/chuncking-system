"""
Phase 4 Tests: Health Check and Monitoring Endpoints

This module contains comprehensive tests for the Phase 4 HTTP health check
and monitoring endpoints including REST API functionality, response formats,
and framework integrations.

Test Coverage:
- HealthEndpoint basic and detailed health checks
- MetricsEndpoint Prometheus and JSON exports
- SystemStatusEndpoint system information
- EndpointRouter request routing
- Flask and FastAPI integration
- HTTP response validation
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Tuple
from datetime import datetime

from src.api.health_endpoints import (
    HealthEndpoint,
    MetricsEndpoint, 
    SystemStatusEndpoint,
    EndpointRouter,
    create_flask_blueprint,
    create_fastapi_router,
    run_standalone_server
)
from src.utils.monitoring import SystemMonitor, HealthChecker
from src.utils.observability import (
    ObservabilityManager,
    HealthStatus,
    HealthCheckResult,
    MetricType
)


class TestHealthEndpoint:
    """Test HealthEndpoint for basic and detailed health checks."""
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_health_endpoint_initialization(self, mock_obs_manager, mock_system_monitor):
        """Test HealthEndpoint initialization."""
        mock_monitor = Mock()
        mock_system_monitor.return_value = mock_monitor
        mock_obs = Mock()
        mock_obs_manager.return_value = mock_obs
        
        endpoint = HealthEndpoint()
        
        assert endpoint.system_monitor == mock_monitor
        assert endpoint.observability_manager == mock_obs
        mock_system_monitor.assert_called_once()
        mock_obs_manager.assert_called_once()
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_basic_health_check_healthy(self, mock_obs_manager, mock_system_monitor):
        """Test basic health check with healthy system."""
        # Setup mocks
        mock_monitor = Mock()
        mock_monitor.health_checker.get_overall_health.return_value = HealthCheckResult(
            "system", HealthStatus.HEALTHY, "All systems operational"
        )
        mock_system_monitor.return_value = mock_monitor
        
        mock_obs = Mock()
        mock_obs_manager.return_value = mock_obs
        
        endpoint = HealthEndpoint()
        response, status_code = endpoint.health_check()
        
        assert status_code == 200
        assert response["status"] == "healthy"
        assert response["message"] == "All systems operational"
        assert "timestamp" in response
        assert "uptime" in response
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_basic_health_check_unhealthy(self, mock_obs_manager, mock_system_monitor):
        """Test basic health check with unhealthy system."""
        mock_monitor = Mock()
        mock_monitor.health_checker.get_overall_health.return_value = HealthCheckResult(
            "system", HealthStatus.UNHEALTHY, "Database connection failed"
        )
        mock_system_monitor.return_value = mock_monitor
        
        mock_obs = Mock()
        mock_obs_manager.return_value = mock_obs
        
        endpoint = HealthEndpoint()
        response, status_code = endpoint.health_check()
        
        assert status_code == 503
        assert response["status"] == "unhealthy"
        assert response["message"] == "Database connection failed"
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_detailed_health_check(self, mock_obs_manager, mock_system_monitor):
        """Test detailed health check with component breakdown."""
        mock_monitor = Mock()
        mock_monitor.health_checker.run_all_checks.return_value = {
            "database": HealthCheckResult("database", HealthStatus.HEALTHY, "Connected"),
            "cache": HealthCheckResult("cache", HealthStatus.DEGRADED, "High latency"),
            "filesystem": HealthCheckResult("filesystem", HealthStatus.HEALTHY, "Available")
        }
        mock_system_monitor.return_value = mock_monitor
        
        mock_obs = Mock()
        mock_obs_manager.return_value = mock_obs
        
        endpoint = HealthEndpoint()
        response, status_code = endpoint.detailed_health()
        
        assert status_code == 200
        assert "overall_status" in response
        assert "components" in response
        assert len(response["components"]) == 3
        assert response["components"]["database"]["status"] == "healthy"
        assert response["components"]["cache"]["status"] == "degraded"
        assert response["components"]["filesystem"]["status"] == "healthy"
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_component_specific_health_check(self, mock_obs_manager, mock_system_monitor):
        """Test health check for specific component."""
        mock_monitor = Mock()
        mock_monitor.health_checker.run_check.return_value = HealthCheckResult(
            "database", HealthStatus.HEALTHY, "Connection pool active", response_time_ms=25.5
        )
        mock_system_monitor.return_value = mock_monitor
        
        mock_obs = Mock()
        mock_obs_manager.return_value = mock_obs
        
        endpoint = HealthEndpoint()
        response, status_code = endpoint.health_check(component="database")
        
        assert status_code == 200
        assert response["component"] == "database"
        assert response["status"] == "healthy"
        assert response["response_time_ms"] == 25.5
        mock_monitor.health_checker.run_check.assert_called_once_with("database")
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_readiness_check(self, mock_obs_manager, mock_system_monitor):
        """Test Kubernetes readiness probe."""
        mock_monitor = Mock()
        mock_monitor.health_checker.get_overall_health.return_value = HealthCheckResult(
            "system", HealthStatus.HEALTHY, "Ready to serve traffic"
        )
        mock_system_monitor.return_value = mock_monitor
        
        mock_obs = Mock()
        mock_obs_manager.return_value = mock_obs
        
        endpoint = HealthEndpoint()
        response, status_code = endpoint.readiness_check()
        
        assert status_code == 200
        assert response["ready"] is True
        assert response["message"] == "Ready to serve traffic"
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_liveness_check(self, mock_obs_manager, mock_system_monitor):
        """Test Kubernetes liveness probe."""
        mock_monitor = Mock()
        mock_obs = Mock()
        mock_obs_manager.return_value = mock_obs
        mock_system_monitor.return_value = mock_monitor
        
        endpoint = HealthEndpoint()
        response, status_code = endpoint.liveness_check()
        
        assert status_code == 200
        assert response["alive"] is True
        assert "uptime" in response


class TestMetricsEndpoint:
    """Test MetricsEndpoint for metrics export."""
    
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_metrics_endpoint_initialization(self, mock_obs_manager):
        """Test MetricsEndpoint initialization."""
        mock_obs = Mock()
        mock_obs_manager.return_value = mock_obs
        
        endpoint = MetricsEndpoint()
        
        assert endpoint.observability_manager == mock_obs
        mock_obs_manager.assert_called_once()
    
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_prometheus_metrics_export(self, mock_obs_manager):
        """Test Prometheus format metrics export."""
        mock_obs = Mock()
        mock_obs.metrics_registry.export_prometheus_format.return_value = """
# HELP chunking_operations_total Total chunking operations
# TYPE chunking_operations_total counter
chunking_operations_total{method="hybrid"} 1250

# HELP system_cpu_percent CPU usage percentage
# TYPE system_cpu_percent gauge
system_cpu_percent 65.5
"""
        mock_obs_manager.return_value = mock_obs
        
        endpoint = MetricsEndpoint()
        response, status_code = endpoint.prometheus_metrics()
        
        assert status_code == 200
        assert "chunking_operations_total" in response
        assert "system_cpu_percent" in response
        assert "1250" in response
        assert "65.5" in response
    
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_json_metrics_export(self, mock_obs_manager):
        """Test JSON format metrics export."""
        mock_obs = Mock()
        mock_obs.export_all_data.return_value = {
            "metrics": [
                {
                    "name": "requests_total",
                    "value": 100,
                    "type": "counter",
                    "labels": {"endpoint": "/health"}
                },
                {
                    "name": "cpu_usage",
                    "value": 75.5,
                    "type": "gauge",
                    "unit": "percent"
                }
            ],
            "timestamp": "2024-01-15T10:30:00Z"
        }
        mock_obs_manager.return_value = mock_obs
        
        endpoint = MetricsEndpoint()
        response, status_code = endpoint.json_metrics()
        
        assert status_code == 200
        assert "metrics" in response
        assert len(response["metrics"]) == 2
        assert response["metrics"][0]["name"] == "requests_total"
        assert response["metrics"][1]["value"] == 75.5
    
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_metrics_filtering(self, mock_obs_manager):
        """Test metrics filtering by name pattern."""
        mock_obs = Mock()
        mock_obs.metrics_registry.get_metrics_by_name.return_value = [
            Mock(name="cpu_usage", value=75.0, metric_type=MetricType.GAUGE)
        ]
        mock_obs_manager.return_value = mock_obs
        
        endpoint = MetricsEndpoint()
        response, status_code = endpoint.json_metrics(filter_pattern="cpu_*")
        
        assert status_code == 200
        mock_obs.metrics_registry.get_metrics_by_name.assert_called()


class TestSystemStatusEndpoint:
    """Test SystemStatusEndpoint for system information."""
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_system_status_initialization(self, mock_obs_manager, mock_system_monitor):
        """Test SystemStatusEndpoint initialization."""
        mock_monitor = Mock()
        mock_system_monitor.return_value = mock_monitor
        mock_obs = Mock()
        mock_obs_manager.return_value = mock_obs
        
        endpoint = SystemStatusEndpoint()
        
        assert endpoint.system_monitor == mock_monitor
        assert endpoint.observability_manager == mock_obs
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.ObservabilityManager')
    @patch('src.api.health_endpoints.platform')
    @patch('src.api.health_endpoints.psutil')
    def test_system_info(self, mock_psutil, mock_platform, mock_obs_manager, mock_system_monitor):
        """Test system information endpoint."""
        # Setup mocks
        mock_platform.system.return_value = "Linux"
        mock_platform.release.return_value = "5.4.0"
        mock_platform.machine.return_value = "x86_64"
        mock_platform.python_version.return_value = "3.11.0"
        
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value = Mock(total=16000000000)
        mock_psutil.disk_usage.return_value = Mock(total=500000000000)
        
        mock_monitor = Mock()
        mock_system_monitor.return_value = mock_monitor
        mock_obs = Mock()
        mock_obs_manager.return_value = mock_obs
        
        endpoint = SystemStatusEndpoint()
        response, status_code = endpoint.system_info()
        
        assert status_code == 200
        assert response["system"]["os"] == "Linux"
        assert response["system"]["kernel"] == "5.4.0"
        assert response["system"]["architecture"] == "x86_64"
        assert response["system"]["python_version"] == "3.11.0"
        assert response["hardware"]["cpu_cores"] == 8
        assert response["hardware"]["total_memory_gb"] == 16.0
        assert response["hardware"]["total_disk_gb"] == 500.0
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_performance_metrics(self, mock_obs_manager, mock_system_monitor):
        """Test performance metrics endpoint."""
        mock_monitor = Mock()
        mock_monitor.get_system_metrics.return_value = {
            "cpu_percent": 65.5,
            "memory_percent": 78.2,
            "disk_percent": 45.0,
            "load_average": [1.2, 1.5, 1.8]
        }
        mock_system_monitor.return_value = mock_monitor
        
        mock_obs = Mock()
        mock_obs_manager.return_value = mock_obs
        
        endpoint = SystemStatusEndpoint()
        response, status_code = endpoint.performance_metrics()
        
        assert status_code == 200
        assert response["cpu_percent"] == 65.5
        assert response["memory_percent"] == 78.2
        assert response["disk_percent"] == 45.0
        assert response["load_average"] == [1.2, 1.5, 1.8]


class TestEndpointRouter:
    """Test EndpointRouter for request routing."""
    
    def test_endpoint_router_initialization(self):
        """Test EndpointRouter initialization."""
        router = EndpointRouter()
        
        assert isinstance(router.health_endpoint, HealthEndpoint)
        assert isinstance(router.metrics_endpoint, MetricsEndpoint)
        assert isinstance(router.system_endpoint, SystemStatusEndpoint)
    
    def test_route_health_endpoints(self):
        """Test routing health-related requests."""
        router = EndpointRouter()
        
        with patch.object(router.health_endpoint, 'health_check') as mock_health:
            mock_health.return_value = ({"status": "healthy"}, 200)
            
            response, status = router.route_request("GET", "/health")
            
            assert status == 200
            assert response["status"] == "healthy"
            mock_health.assert_called_once_with(component=None)
    
    def test_route_detailed_health(self):
        """Test routing detailed health requests."""
        router = EndpointRouter()
        
        with patch.object(router.health_endpoint, 'detailed_health') as mock_detailed:
            mock_detailed.return_value = ({"overall_status": "healthy"}, 200)
            
            response, status = router.route_request("GET", "/health/detailed")
            
            assert status == 200
            mock_detailed.assert_called_once()
    
    def test_route_metrics_endpoints(self):
        """Test routing metrics requests."""
        router = EndpointRouter()
        
        with patch.object(router.metrics_endpoint, 'prometheus_metrics') as mock_prometheus:
            mock_prometheus.return_value = ("# metrics data", 200)
            
            response, status = router.route_request("GET", "/metrics")
            
            assert status == 200
            mock_prometheus.assert_called_once()
    
    def test_route_system_endpoints(self):
        """Test routing system status requests."""
        router = EndpointRouter()
        
        with patch.object(router.system_endpoint, 'system_info') as mock_system:
            mock_system.return_value = ({"system": {"os": "Linux"}}, 200)
            
            response, status = router.route_request("GET", "/system/info")
            
            assert status == 200
            mock_system.assert_called_once()
    
    def test_route_not_found(self):
        """Test routing unknown endpoints."""
        router = EndpointRouter()
        
        response, status = router.route_request("GET", "/unknown")
        
        assert status == 404
        assert "error" in response
        assert response["error"] == "Endpoint not found"
    
    def test_route_method_not_allowed(self):
        """Test routing with unsupported HTTP methods."""
        router = EndpointRouter()
        
        response, status = router.route_request("POST", "/health")
        
        assert status == 405
        assert "error" in response
        assert response["error"] == "Method not allowed"


class TestFlaskIntegration:
    """Test Flask framework integration."""
    
    @patch('src.api.health_endpoints.Flask')
    @patch('src.api.health_endpoints.Blueprint')
    def test_create_flask_blueprint(self, mock_blueprint_class, mock_flask):
        """Test creating Flask blueprint."""
        mock_blueprint = Mock()
        mock_blueprint_class.return_value = mock_blueprint
        
        blueprint = create_flask_blueprint()
        
        assert blueprint == mock_blueprint
        mock_blueprint_class.assert_called_once_with(
            'health_monitoring', __name__, url_prefix='/monitoring'
        )
    
    @patch('src.api.health_endpoints.Flask')
    @patch('src.api.health_endpoints.Blueprint')
    @patch('src.api.health_endpoints.EndpointRouter')
    def test_flask_route_handling(self, mock_router_class, mock_blueprint_class, mock_flask):
        """Test Flask route handling."""
        mock_router = Mock()
        mock_router.route_request.return_value = ({"status": "healthy"}, 200)
        mock_router_class.return_value = mock_router
        
        mock_blueprint = Mock()
        mock_blueprint_class.return_value = mock_blueprint
        
        # Create blueprint
        blueprint = create_flask_blueprint()
        
        # Verify router was configured
        mock_router_class.assert_called_once()


class TestFastAPIIntegration:
    """Test FastAPI framework integration."""
    
    @patch('src.api.health_endpoints.APIRouter')
    def test_create_fastapi_router(self, mock_router_class):
        """Test creating FastAPI router."""
        mock_router = Mock()
        mock_router_class.return_value = mock_router
        
        router = create_fastapi_router()
        
        assert router == mock_router
        mock_router_class.assert_called_once_with(prefix="/monitoring", tags=["monitoring"])
    
    @patch('src.api.health_endpoints.APIRouter')
    @patch('src.api.health_endpoints.EndpointRouter')
    def test_fastapi_route_handling(self, mock_endpoint_router_class, mock_router_class):
        """Test FastAPI route handling."""
        mock_endpoint_router = Mock()
        mock_endpoint_router.route_request.return_value = ({"status": "healthy"}, 200)
        mock_endpoint_router_class.return_value = mock_endpoint_router
        
        mock_router = Mock()
        mock_router_class.return_value = mock_router
        
        # Create router
        router = create_fastapi_router()
        
        # Verify endpoint router was configured
        mock_endpoint_router_class.assert_called_once()


class TestStandaloneServer:
    """Test standalone server functionality."""
    
    @patch('src.api.health_endpoints.Flask')
    @patch('src.api.health_endpoints.create_flask_blueprint')
    def test_standalone_server_configuration(self, mock_create_blueprint, mock_flask_class):
        """Test standalone server configuration."""
        mock_app = Mock()
        mock_flask_class.return_value = mock_app
        
        mock_blueprint = Mock()
        mock_create_blueprint.return_value = mock_blueprint
        
        # Test server creation (without actually running)
        with patch('src.api.health_endpoints.threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance
            
            # This would normally start the server, but we're mocking it
            # run_standalone_server(host="localhost", port=8000, debug=False)
            
            # Verify Flask app was configured
            mock_flask_class.assert_called_once_with(__name__)


class TestErrorHandling:
    """Test error handling in endpoints."""
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_health_check_with_exception(self, mock_obs_manager, mock_system_monitor):
        """Test health check with system monitor exception."""
        mock_monitor = Mock()
        mock_monitor.health_checker.get_overall_health.side_effect = Exception("System error")
        mock_system_monitor.return_value = mock_monitor
        
        mock_obs = Mock()
        mock_obs_manager.return_value = mock_obs
        
        endpoint = HealthEndpoint()
        response, status_code = endpoint.health_check()
        
        assert status_code == 500
        assert response["status"] == "error"
        assert "System error" in response["message"]
    
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_metrics_export_with_exception(self, mock_obs_manager):
        """Test metrics export with observability manager exception."""
        mock_obs = Mock()
        mock_obs.metrics_registry.export_prometheus_format.side_effect = Exception("Metrics error")
        mock_obs_manager.return_value = mock_obs
        
        endpoint = MetricsEndpoint()
        response, status_code = endpoint.prometheus_metrics()
        
        assert status_code == 500
        assert "error" in response
        assert "Metrics error" in response["error"]


class TestResponseFormats:
    """Test response format validation."""
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_health_response_structure(self, mock_obs_manager, mock_system_monitor):
        """Test health response structure compliance."""
        mock_monitor = Mock()
        mock_monitor.health_checker.get_overall_health.return_value = HealthCheckResult(
            "system", HealthStatus.HEALTHY, "All good"
        )
        mock_system_monitor.return_value = mock_monitor
        
        mock_obs = Mock()
        mock_obs_manager.return_value = mock_obs
        
        endpoint = HealthEndpoint()
        response, status_code = endpoint.health_check()
        
        # Verify required fields
        required_fields = ["status", "message", "timestamp", "uptime"]
        for field in required_fields:
            assert field in response
        
        # Verify status is valid
        assert response["status"] in ["healthy", "degraded", "unhealthy"]
        
        # Verify timestamp format
        assert isinstance(response["timestamp"], str)
        datetime.fromisoformat(response["timestamp"].replace('Z', '+00:00'))
    
    @patch('src.api.health_endpoints.ObservabilityManager')
    def test_metrics_prometheus_format(self, mock_obs_manager):
        """Test Prometheus metrics format compliance."""
        mock_obs = Mock()
        prometheus_output = """# HELP test_metric A test metric
# TYPE test_metric gauge
test_metric{label="value"} 42.0
"""
        mock_obs.metrics_registry.export_prometheus_format.return_value = prometheus_output
        mock_obs_manager.return_value = mock_obs
        
        endpoint = MetricsEndpoint()
        response, status_code = endpoint.prometheus_metrics()
        
        assert status_code == 200
        assert isinstance(response, str)
        assert "# HELP" in response
        assert "# TYPE" in response
        assert "test_metric" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])