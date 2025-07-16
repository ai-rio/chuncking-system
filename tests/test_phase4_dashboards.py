"""
Phase 4 Tests: Dashboard Configuration and Monitoring Setup

This module contains comprehensive tests for Phase 4 dashboard generation,
configuration validation, and monitoring infrastructure setup including
Grafana dashboards, Prometheus configurations, and alerting rules.

Test Coverage:
- Grafana dashboard JSON structure validation
- Prometheus configuration file validation
- Alert rules syntax and logic validation
- Dashboard panel configuration testing
- Metrics query validation
- Dashboard template generation
"""

import pytest
import json
import yaml
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
from pathlib import Path

from src.utils.observability import (
    DashboardGenerator,
    MetricsRegistry,
    HealthRegistry,
    MetricType,
    HealthStatus,
    HealthCheckResult
)


class TestDashboardGenerator:
    """Test DashboardGenerator for monitoring configuration generation."""
    
    def test_dashboard_generator_initialization(self):
        """Test DashboardGenerator initialization."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        assert generator.metrics_registry == metrics_registry
        assert generator.health_registry == health_registry
    
    def test_generate_grafana_dashboard_structure(self):
        """Test Grafana dashboard structure generation."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        dashboard_config = generator.generate_grafana_dashboard()
        
        # Verify top-level structure
        assert "dashboard" in dashboard_config
        dashboard = dashboard_config["dashboard"]
        
        # Verify required dashboard fields
        required_fields = ["id", "title", "tags", "style", "timezone", "editable", 
                          "graphTooltip", "time", "timepicker", "refresh", 
                          "schemaVersion", "version", "panels"]
        for field in required_fields:
            assert field in dashboard
        
        # Verify dashboard metadata
        assert dashboard["title"] == "Document Chunking System - Phase 4 Observability"
        assert "chunking" in dashboard["tags"]
        assert "monitoring" in dashboard["tags"]
        assert "phase4" in dashboard["tags"]
        assert "observability" in dashboard["tags"]
        
        # Verify time settings
        assert dashboard["time"]["from"] == "now-1h"
        assert dashboard["time"]["to"] == "now"
        assert dashboard["refresh"] == "5s"
    
    def test_grafana_dashboard_panels_generation(self):
        """Test Grafana dashboard panels generation."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        # Add some metrics to influence panel generation
        metrics_registry.record_metric("chunking_operations_total", 100, MetricType.COUNTER)
        metrics_registry.record_metric("system_cpu_percent", 75, MetricType.GAUGE)
        metrics_registry.record_metric("response_time_ms", 150, MetricType.HISTOGRAM)
        
        dashboard_config = generator.generate_grafana_dashboard()
        panels = dashboard_config["dashboard"]["panels"]
        
        # Verify minimum number of panels
        assert len(panels) >= 5
        
        # Verify essential panels exist
        panel_titles = [panel["title"] for panel in panels]
        essential_panels = [
            "System Health Status",
            "CPU Usage",
            "Memory Usage",
            "Chunking Operations Rate",
            "Processing Duration"
        ]
        
        for essential_panel in essential_panels:
            assert essential_panel in panel_titles
    
    def test_grafana_panel_configuration(self):
        """Test individual Grafana panel configuration."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        dashboard_config = generator.generate_grafana_dashboard()
        panels = dashboard_config["dashboard"]["panels"]
        
        # Find and test health status panel
        health_panel = next((p for p in panels if p["title"] == "System Health Status"), None)
        assert health_panel is not None
        
        # Verify panel structure
        assert health_panel["type"] == "stat"
        assert "gridPos" in health_panel
        assert "targets" in health_panel
        assert "fieldConfig" in health_panel
        assert "options" in health_panel
        
        # Verify grid position
        grid_pos = health_panel["gridPos"]
        assert "h" in grid_pos and "w" in grid_pos and "x" in grid_pos and "y" in grid_pos
        
        # Verify targets (Prometheus queries)
        targets = health_panel["targets"]
        assert len(targets) >= 1
        for target in targets:
            assert "expr" in target
            assert "refId" in target
        
        # Verify field configuration
        field_config = health_panel["fieldConfig"]
        assert "defaults" in field_config
        defaults = field_config["defaults"]
        assert "color" in defaults
        assert "thresholds" in defaults
    
    def test_prometheus_query_generation(self):
        """Test Prometheus query generation for panels."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        # Add metrics that should generate specific queries
        metrics_registry.record_metric("chunking_operations_total", 100, MetricType.COUNTER)
        metrics_registry.record_metric("chunking_errors_total", 5, MetricType.COUNTER)
        metrics_registry.record_metric("chunking_duration_ms", 850, MetricType.HISTOGRAM)
        
        dashboard_config = generator.generate_grafana_dashboard()
        panels = dashboard_config["dashboard"]["panels"]
        
        # Collect all Prometheus queries
        all_queries = []
        for panel in panels:
            if "targets" in panel:
                for target in panel["targets"]:
                    if "expr" in target:
                        all_queries.append(target["expr"])
        
        # Verify essential queries exist
        query_strings = " ".join(all_queries)
        
        # Should have rate queries for counters
        assert "rate(" in query_strings or "increase(" in query_strings
        
        # Should have quantile queries for histograms
        assert "quantile(" in query_strings or "histogram_quantile(" in query_strings
        
        # Should reference expected metrics
        assert "chunking_operations_total" in query_strings or "chunking_system" in query_strings


class TestPrometheusConfiguration:
    """Test Prometheus configuration generation."""
    
    def test_generate_prometheus_config_structure(self):
        """Test Prometheus configuration structure."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        prometheus_config = generator.generate_prometheus_config()
        
        # Verify top-level structure
        required_sections = ["global", "rule_files", "alerting", "scrape_configs"]
        for section in required_sections:
            assert section in prometheus_config
        
        # Verify global configuration
        global_config = prometheus_config["global"]
        assert "scrape_interval" in global_config
        assert "evaluation_interval" in global_config
        assert global_config["scrape_interval"] == "15s"
        assert global_config["evaluation_interval"] == "15s"
    
    def test_prometheus_scrape_configurations(self):
        """Test Prometheus scrape configurations."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        prometheus_config = generator.generate_prometheus_config()
        scrape_configs = prometheus_config["scrape_configs"]
        
        # Verify main application scrape config exists
        app_scrape = next((sc for sc in scrape_configs 
                          if sc["job_name"] == "chunking-system"), None)
        assert app_scrape is not None
        
        # Verify scrape config structure
        assert "static_configs" in app_scrape
        assert "metrics_path" in app_scrape
        assert "scrape_interval" in app_scrape
        
        # Verify target configuration
        static_configs = app_scrape["static_configs"]
        assert len(static_configs) >= 1
        assert "targets" in static_configs[0]
        
        # Verify metrics path
        assert app_scrape["metrics_path"] == "/metrics"
        
        # Check for health endpoint scraping
        health_scrape = next((sc for sc in scrape_configs 
                             if "health" in sc["job_name"]), None)
        if health_scrape:
            assert health_scrape["metrics_path"] in ["/health", "/health/detailed"]
    
    def test_prometheus_rule_files_configuration(self):
        """Test Prometheus rule files configuration."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        prometheus_config = generator.generate_prometheus_config()
        
        # Verify rule files are configured
        assert "rule_files" in prometheus_config
        rule_files = prometheus_config["rule_files"]
        assert len(rule_files) >= 1
        assert "prometheus-alerts.yml" in rule_files
    
    def test_prometheus_alerting_configuration(self):
        """Test Prometheus alerting configuration."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        prometheus_config = generator.generate_prometheus_config()
        
        # Verify alerting configuration
        assert "alerting" in prometheus_config
        alerting_config = prometheus_config["alerting"]
        
        assert "alertmanagers" in alerting_config
        alertmanagers = alerting_config["alertmanagers"]
        assert len(alertmanagers) >= 1
        
        # Verify alertmanager configuration
        alertmanager = alertmanagers[0]
        assert "static_configs" in alertmanager
        assert "targets" in alertmanager["static_configs"][0]


class TestAlertRulesGeneration:
    """Test alert rules generation and validation."""
    
    def test_generate_alert_rules_structure(self):
        """Test alert rules structure generation."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        alert_rules = generator.generate_alert_rules()
        
        # Verify top-level structure
        assert "groups" in alert_rules
        groups = alert_rules["groups"]
        assert len(groups) >= 3  # Multiple alert groups
        
        # Verify each group has required fields
        for group in groups:
            assert "name" in group
            assert "rules" in group
            assert len(group["rules"]) >= 1
    
    def test_critical_alert_rules(self):
        """Test critical alert rules generation."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        alert_rules = generator.generate_alert_rules()
        groups = alert_rules["groups"]
        
        # Find critical alerts group
        critical_group = next((g for g in groups 
                              if g["name"] == "chunking_system_alerts"), None)
        assert critical_group is not None
        
        rules = critical_group["rules"]
        rule_names = [rule["alert"] for rule in rules if "alert" in rule]
        
        # Verify essential critical alerts
        essential_alerts = [
            "SystemHealthDown",
            "HighErrorRate",
            "ProcessingLatencyHigh"
        ]
        
        for essential_alert in essential_alerts:
            assert essential_alert in rule_names
    
    def test_alert_rule_structure(self):
        """Test individual alert rule structure."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        alert_rules = generator.generate_alert_rules()
        groups = alert_rules["groups"]
        
        # Find a specific alert rule to validate
        for group in groups:
            for rule in group["rules"]:
                if "alert" in rule:
                    # Verify required alert fields
                    assert "alert" in rule
                    assert "expr" in rule
                    assert "for" in rule
                    assert "labels" in rule
                    assert "annotations" in rule
                    
                    # Verify labels structure
                    labels = rule["labels"]
                    assert "severity" in labels
                    assert labels["severity"] in ["critical", "warning", "info"]
                    
                    # Verify annotations structure
                    annotations = rule["annotations"]
                    assert "summary" in annotations
                    assert "description" in annotations
                    
                    # Found and validated at least one rule
                    break
            else:
                continue
            break
    
    def test_sla_alert_rules(self):
        """Test SLA-specific alert rules."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        alert_rules = generator.generate_alert_rules()
        groups = alert_rules["groups"]
        
        # Find SLA alerts group
        sla_group = next((g for g in groups 
                         if g["name"] == "chunking_system_sla"), None)
        assert sla_group is not None
        
        rules = sla_group["rules"]
        rule_names = [rule["alert"] for rule in rules if "alert" in rule]
        
        # Verify SLA-specific alerts
        sla_alerts = [
            "SLAErrorRateBreach",
            "SLALatencyBreach", 
            "SLAAvailabilityBreach"
        ]
        
        for sla_alert in sla_alerts:
            assert sla_alert in rule_names
    
    def test_recording_rules(self):
        """Test recording rules generation."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        alert_rules = generator.generate_alert_rules()
        groups = alert_rules["groups"]
        
        # Find recording rules group
        recording_group = next((g for g in groups 
                               if "recording" in g["name"]), None)
        assert recording_group is not None
        
        rules = recording_group["rules"]
        recording_rules = [rule for rule in rules if "record" in rule]
        
        # Verify recording rules exist
        assert len(recording_rules) >= 3
        
        # Verify recording rule structure
        for rule in recording_rules:
            assert "record" in rule
            assert "expr" in rule
            assert ":" in rule["record"]  # Recording rule naming convention


class TestDashboardValidation:
    """Test dashboard configuration validation."""
    
    def test_grafana_json_validity(self):
        """Test that generated Grafana JSON is valid."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        dashboard_config = generator.generate_grafana_dashboard()
        
        # Verify JSON is valid by serializing and parsing
        json_string = json.dumps(dashboard_config)
        parsed_config = json.loads(json_string)
        
        assert parsed_config == dashboard_config
    
    def test_prometheus_yaml_validity(self):
        """Test that generated Prometheus YAML is valid."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        prometheus_config = generator.generate_prometheus_config()
        alert_rules = generator.generate_alert_rules()
        
        # Verify YAML is valid by serializing and parsing
        prometheus_yaml = yaml.dump(prometheus_config)
        parsed_prometheus = yaml.safe_load(prometheus_yaml)
        assert parsed_prometheus == prometheus_config
        
        alert_rules_yaml = yaml.dump(alert_rules)
        parsed_alerts = yaml.safe_load(alert_rules_yaml)
        assert parsed_alerts == alert_rules
    
    def test_panel_target_query_syntax(self):
        """Test that panel target queries have valid syntax."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        dashboard_config = generator.generate_grafana_dashboard()
        panels = dashboard_config["dashboard"]["panels"]
        
        # Collect all queries and validate basic syntax
        for panel in panels:
            if "targets" in panel:
                for target in panel["targets"]:
                    if "expr" in target:
                        query = target["expr"]
                        
                        # Basic syntax validation
                        assert len(query.strip()) > 0
                        assert not query.startswith(" ")
                        assert not query.endswith(" ")
                        
                        # Check for common Prometheus functions
                        if "rate(" in query:
                            assert "[" in query and "]" in query  # Time range
                        
                        if "quantile(" in query:
                            assert "," in query  # Quantile value
    
    def test_alert_expression_syntax(self):
        """Test that alert expressions have valid syntax."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        alert_rules = generator.generate_alert_rules()
        groups = alert_rules["groups"]
        
        # Validate alert expressions (skip recording rules)
        for group in groups:
            for rule in group["rules"]:
                if "expr" in rule and "alert" in rule:  # Only check alert rules, not recording rules
                    expr = rule["expr"]
                    
                    # Basic syntax validation
                    assert len(expr.strip()) > 0
                    assert not expr.startswith(" ")
                    assert not expr.endswith(" ")
                    
                    # Check for comparison operators in alerts
                    comparison_ops = [">", "<", ">=", "<=", "==", "!="]
                    assert any(op in expr for op in comparison_ops)


class TestDashboardDataIntegration:
    """Test dashboard generation with real data."""
    
    def test_dashboard_with_comprehensive_metrics(self):
        """Test dashboard generation with comprehensive metrics data."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        # Add comprehensive metrics
        metrics_data = [
            ("chunking_operations_total", 2500, MetricType.COUNTER),
            ("chunking_errors_total", 25, MetricType.COUNTER),
            ("chunking_duration_ms", 850, MetricType.HISTOGRAM),
            ("chunk_quality_score", 87.5, MetricType.GAUGE),
            ("system_cpu_percent", 65.0, MetricType.GAUGE),
            ("system_memory_percent", 72.0, MetricType.GAUGE),
            ("system_disk_percent", 45.0, MetricType.GAUGE),
            ("cache_hit_rate", 89.5, MetricType.GAUGE),
            ("cache_miss_rate", 10.5, MetricType.GAUGE),
            ("active_connections", 45, MetricType.GAUGE),
            ("processing_queue_size", 12, MetricType.GAUGE)
        ]
        
        for name, value, metric_type in metrics_data:
            metrics_registry.record_metric(name, value, metric_type)
        
        # Add health checks
        def system_health():
            return HealthCheckResult("system", HealthStatus.HEALTHY, "Operational")
        
        def database_health():
            return HealthCheckResult("database", HealthStatus.HEALTHY, "Connected")
        
        def cache_health():
            return HealthCheckResult("cache", HealthStatus.DEGRADED, "High latency")
        
        health_registry.register_health_check("system", system_health)
        health_registry.register_health_check("database", database_health)
        health_registry.register_health_check("cache", cache_health)
        
        # Generate dashboard
        dashboard_config = generator.generate_grafana_dashboard()
        
        # Verify dashboard contains relevant panels for the metrics
        panels = dashboard_config["dashboard"]["panels"]
        panel_titles = [panel["title"] for panel in panels]
        
        # Should have panels for key metrics
        expected_panel_keywords = ["CPU", "Memory", "Operations", "Duration", "Health"]
        for keyword in expected_panel_keywords:
            assert any(keyword in title for title in panel_titles)
    
    def test_dashboard_with_health_checks(self):
        """Test dashboard generation with health check data."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        # Add multiple health checks with different statuses
        health_checks = [
            ("api_gateway", HealthStatus.HEALTHY, "API Gateway operational"),
            ("document_processor", HealthStatus.HEALTHY, "Processing service ready"),
            ("quality_evaluator", HealthStatus.HEALTHY, "Evaluation service online"),
            ("cache_service", HealthStatus.DEGRADED, "High cache miss rate"),
            ("database", HealthStatus.HEALTHY, "Database connections stable"),
            ("file_storage", HealthStatus.HEALTHY, "Storage system operational")
        ]
        
        for component, status, message in health_checks:
            def health_check(comp=component, stat=status, msg=message):
                return HealthCheckResult(comp, stat, msg)
            health_registry.register_health_check(component, health_check)
        
        # Generate dashboard
        dashboard_config = generator.generate_grafana_dashboard()
        
        # Verify health-related panels exist
        panels = dashboard_config["dashboard"]["panels"]
        health_panels = [p for p in panels if "health" in p["title"].lower()]
        
        assert len(health_panels) >= 1
        
        # Check for component health details panel
        component_panel = next((p for p in panels 
                               if "component" in p["title"].lower()), None)
        if component_panel:
            assert component_panel["type"] == "table"


class TestConfigurationFiles:
    """Test configuration file generation and validation."""
    
    def test_save_grafana_dashboard_file(self):
        """Test saving Grafana dashboard to file."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        dashboard_config = generator.generate_grafana_dashboard()
        
        # Test that the config can be serialized to JSON
        json_output = json.dumps(dashboard_config, indent=2)
        
        # Verify JSON is properly formatted
        assert len(json_output) > 1000  # Should be substantial
        assert '"dashboard"' in json_output
        assert '"title"' in json_output
        assert '"panels"' in json_output
        
        # Verify it can be parsed back
        parsed_config = json.loads(json_output)
        assert parsed_config["dashboard"]["title"] == "Document Chunking System - Phase 4 Observability"
    
    def test_save_prometheus_config_file(self):
        """Test saving Prometheus configuration to file."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        prometheus_config = generator.generate_prometheus_config()
        
        # Test that the config can be serialized to YAML
        yaml_output = yaml.dump(prometheus_config, default_flow_style=False)
        
        # Verify YAML is properly formatted
        assert len(yaml_output) > 500  # Should be substantial
        assert "global:" in yaml_output
        assert "scrape_configs:" in yaml_output
        assert "alerting:" in yaml_output
        
        # Verify it can be parsed back
        parsed_config = yaml.safe_load(yaml_output)
        assert parsed_config["global"]["scrape_interval"] == "15s"
    
    def test_save_alert_rules_file(self):
        """Test saving alert rules to file."""
        metrics_registry = MetricsRegistry()
        health_registry = HealthRegistry()
        generator = DashboardGenerator(metrics_registry, health_registry)
        
        alert_rules = generator.generate_alert_rules()
        
        # Test that the rules can be serialized to YAML
        yaml_output = yaml.dump(alert_rules, default_flow_style=False)
        
        # Verify YAML is properly formatted
        assert len(yaml_output) > 1000  # Should be substantial
        assert "groups:" in yaml_output
        assert "alert:" in yaml_output
        assert "expr:" in yaml_output
        
        # Verify it can be parsed back
        parsed_rules = yaml.safe_load(yaml_output)
        assert len(parsed_rules["groups"]) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])