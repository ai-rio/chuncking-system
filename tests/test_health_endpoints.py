
import pytest
from unittest.mock import Mock, patch
from src.api.health_endpoints import (
    HealthEndpoint,
    MetricsEndpoint,
    SystemStatusEndpoint,
    EndpointRouter
)
from src.utils.observability import HealthStatus, HealthCheckResult

@pytest.fixture
def mock_system_monitor():
    monitor = Mock()
    monitor.health_checker.run_check.return_value = HealthCheckResult(
        component="test",
        status=HealthStatus.HEALTHY,
        message="OK"
    )
    monitor.health_checker.get_overall_health.return_value = HealthCheckResult(
        component="overall",
        status=HealthStatus.HEALTHY,
        message="OK"
    )
    monitor.health_checker.run_all_checks.return_value = {
        "test": HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
            message="OK"
        )
    }
    monitor.get_system_status.return_value = {
        "health": {"overall_healthy": True},
        "metrics_count": 10,
        "active_alerts": 0
    }
    return monitor

@pytest.fixture
def mock_observability_manager():
    manager = Mock()
    manager.export_prometheus_metrics.return_value = "metrics"
    manager.export_all_data.return_value = {"metrics": {"metrics": []}}
    manager.metrics_registry.get_metric_summary.return_value = {"summary": "data"}
    return manager

@pytest.fixture
def health_endpoint(mock_system_monitor, mock_observability_manager):
    return HealthEndpoint(mock_system_monitor, mock_observability_manager)

@pytest.fixture
def metrics_endpoint(mock_system_monitor, mock_observability_manager):
    return MetricsEndpoint(mock_system_monitor, mock_observability_manager)

@pytest.fixture
def system_status_endpoint(mock_system_monitor, mock_observability_manager):
    return SystemStatusEndpoint(mock_system_monitor, mock_observability_manager)

@pytest.fixture
def endpoint_router(mock_system_monitor):
    return EndpointRouter(mock_system_monitor)

def test_health_check(health_endpoint):
    response, status_code = health_endpoint.health_check()
    assert status_code == 200
    assert response["status"] == "healthy"

def test_health_check_component(health_endpoint):
    response, status_code = health_endpoint.health_check(component="test")
    assert status_code == 200
    assert response["component"] == "test"

def test_detailed_health(health_endpoint):
    response, status_code = health_endpoint.detailed_health()
    assert status_code == 200
    assert "overall_status" in response

def test_readiness_check(health_endpoint):
    response, status_code = health_endpoint.readiness_check()
    assert status_code in [200, 503]

def test_liveness_check(health_endpoint):
    response, status_code = health_endpoint.liveness_check()
    assert status_code == 200
    assert response["alive"] is True

def test_prometheus_metrics(metrics_endpoint):
    response, status_code = metrics_endpoint.prometheus_metrics()
    assert status_code == 200
    assert "metrics" in response

def test_json_metrics(metrics_endpoint):
    response, status_code = metrics_endpoint.json_metrics()
    assert status_code == 200
    assert "metrics" in response

def test_metric_details(metrics_endpoint):
    response, status_code = metrics_endpoint.metric_details("test_metric")
    assert status_code == 200
    assert "summary" in response

def test_system_info(system_status_endpoint):
    response, status_code = system_status_endpoint.system_info()
    assert status_code == 200
    assert "system" in response

def test_performance_stats(system_status_endpoint):
    response, status_code = system_status_endpoint.performance_stats()
    assert status_code == 200
    assert "cpu_percent" in response

def test_endpoint_router(endpoint_router):
    response, status_code = endpoint_router.handle_request("GET", "/health")
    assert status_code == 200
