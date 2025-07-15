"""
Phase 4 Test Data Generator

Utility module for generating realistic test data for Phase 4 observability tests.
Provides functions to create sample metrics, health check results, and 
configuration data for testing.
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.utils.observability import (
    MetricType,
    HealthStatus,
    HealthCheckResult,
    CustomMetric
)


class TestDataGenerator:
    """Generator for Phase 4 test data."""
    
    @staticmethod
    def generate_sample_metrics(count: int = 100, 
                               time_range_hours: int = 24) -> List[Dict[str, Any]]:
        """Generate sample metrics data."""
        metrics = []
        start_time = datetime.now() - timedelta(hours=time_range_hours)
        
        metric_templates = [
            {
                "name": "chunking_operations_total",
                "type": "counter",
                "unit": "operations",
                "base_value": 1000,
                "variation": 200
            },
            {
                "name": "chunking_errors_total",
                "type": "counter", 
                "unit": "errors",
                "base_value": 10,
                "variation": 5
            },
            {
                "name": "chunking_duration_ms",
                "type": "histogram",
                "unit": "milliseconds",
                "base_value": 800,
                "variation": 300
            },
            {
                "name": "chunk_quality_score",
                "type": "gauge",
                "unit": "score",
                "base_value": 85,
                "variation": 10
            },
            {
                "name": "system_cpu_percent",
                "type": "gauge",
                "unit": "percent",
                "base_value": 60,
                "variation": 20
            },
            {
                "name": "system_memory_percent",
                "type": "gauge",
                "unit": "percent",
                "base_value": 70,
                "variation": 15
            },
            {
                "name": "cache_hit_rate",
                "type": "gauge",
                "unit": "percent",
                "base_value": 85,
                "variation": 10
            }
        ]
        
        for i in range(count):
            template = random.choice(metric_templates)
            timestamp = start_time + timedelta(
                seconds=random.randint(0, int(time_range_hours * 3600))
            )
            
            value = template["base_value"] + random.gauss(0, template["variation"] / 3)
            if template["unit"] == "percent":
                value = max(0, min(100, value))  # Clamp percentages
            elif template["type"] == "counter":
                value = max(0, int(value))  # Counters can't be negative
            
            metric = {
                "name": template["name"],
                "value": round(value, 2),
                "type": template["type"],
                "unit": template["unit"],
                "labels": TestDataGenerator._generate_labels(template["name"]),
                "timestamp": timestamp.isoformat() + "Z"
            }
            
            metrics.append(metric)
        
        return sorted(metrics, key=lambda x: x["timestamp"])
    
    @staticmethod
    def _generate_labels(metric_name: str) -> Dict[str, str]:
        """Generate realistic labels for metrics."""
        common_labels = {
            "environment": random.choice(["test", "staging", "production"]),
            "region": random.choice(["us-west-2", "us-east-1", "eu-west-1"]),
            "version": random.choice(["4.0.0", "4.0.1", "4.1.0"])
        }
        
        specific_labels = {}
        
        if "chunking" in metric_name:
            specific_labels.update({
                "method": random.choice(["hybrid", "recursive", "header"]),
                "document_type": random.choice(["markdown", "text", "pdf"])
            })
        
        if "system" in metric_name:
            specific_labels.update({
                "host": f"host-{random.randint(1, 10)}",
                "measurement": random.choice(["1min", "5min", "current"])
            })
        
        if "cache" in metric_name:
            specific_labels.update({
                "cache_type": random.choice(["memory", "redis", "disk"]),
                "ttl": random.choice(["3600", "7200", "86400"])
            })
        
        return {**common_labels, **specific_labels}
    
    @staticmethod
    def generate_health_check_results(include_unhealthy: bool = True) -> Dict[str, Dict]:
        """Generate sample health check results."""
        components = [
            "chunking_service",
            "quality_evaluator", 
            "cache_system",
            "database",
            "file_storage",
            "api_gateway",
            "load_balancer",
            "monitoring_service"
        ]
        
        results = {}
        
        for component in components:
            if include_unhealthy and random.random() < 0.2:  # 20% chance unhealthy
                status = random.choice([HealthStatus.UNHEALTHY, HealthStatus.DEGRADED])
                message = TestDataGenerator._generate_unhealthy_message(component)
                response_time = random.uniform(100, 500)  # Slower when unhealthy
            else:
                status = HealthStatus.HEALTHY
                message = f"{component.replace('_', ' ').title()} operational"
                response_time = random.uniform(5, 50)
            
            results[component] = {
                "status": status.value,
                "message": message,
                "response_time_ms": round(response_time, 1),
                "timestamp": datetime.now().isoformat() + "Z",
                "details": TestDataGenerator._generate_health_details(component, status)
            }
        
        return results
    
    @staticmethod
    def _generate_unhealthy_message(component: str) -> str:
        """Generate realistic unhealthy messages."""
        messages = {
            "chunking_service": "High processing latency detected",
            "quality_evaluator": "Evaluation timeout threshold exceeded",
            "cache_system": "Cache hit rate below threshold",
            "database": "Connection pool exhausted",
            "file_storage": "Disk usage above 90%",
            "api_gateway": "High error rate on upstream services",
            "load_balancer": "Backend health check failing",
            "monitoring_service": "Metrics collection delayed"
        }
        return messages.get(component, f"{component} experiencing issues")
    
    @staticmethod
    def _generate_health_details(component: str, status: HealthStatus) -> Dict[str, Any]:
        """Generate component-specific health details."""
        if component == "database":
            return {
                "active_connections": random.randint(10, 50),
                "max_connections": 100,
                "query_response_time_ms": random.uniform(5, 30),
                "connection_pool_utilization": random.uniform(0.1, 0.8)
            }
        
        elif component == "cache_system":
            hit_rate = random.uniform(60, 95) if status == HealthStatus.HEALTHY else random.uniform(30, 60)
            return {
                "hit_rate": round(hit_rate, 1),
                "miss_rate": round(100 - hit_rate, 1),
                "cache_size_mb": random.randint(256, 1024),
                "evictions_per_hour": random.randint(50, 200)
            }
        
        elif component == "file_storage":
            return {
                "disk_usage_percent": random.uniform(40, 80),
                "available_space_gb": random.randint(100, 1000),
                "io_operations_per_sec": random.randint(500, 2000)
            }
        
        else:
            return {
                "uptime_percent": random.uniform(99.0, 99.99),
                "requests_per_second": random.randint(50, 500),
                "error_rate": random.uniform(0.001, 0.01)
            }
    
    @staticmethod
    def generate_prometheus_metrics_output(metrics: List[Dict]) -> str:
        """Generate Prometheus format metrics output."""
        output_lines = []
        
        # Group metrics by name
        metrics_by_name = {}
        for metric in metrics:
            name = metric["name"]
            if name not in metrics_by_name:
                metrics_by_name[name] = []
            metrics_by_name[name].append(metric)
        
        for name, metric_list in metrics_by_name.items():
            # Add HELP and TYPE comments
            output_lines.append(f"# HELP {name} {name.replace('_', ' ').title()}")
            
            metric_type = metric_list[0]["type"]
            prom_type = "gauge" if metric_type == "gauge" else "counter"
            output_lines.append(f"# TYPE {name} {prom_type}")
            
            # Add metric lines
            for metric in metric_list:
                labels_str = ""
                if metric.get("labels"):
                    label_pairs = [f'{k}="{v}"' for k, v in metric["labels"].items()]
                    labels_str = "{" + ",".join(label_pairs) + "}"
                
                output_lines.append(f"{name}{labels_str} {metric['value']}")
        
        return "\n".join(output_lines)
    
    @staticmethod
    def generate_grafana_dashboard_config(metrics: List[Dict]) -> Dict[str, Any]:
        """Generate Grafana dashboard configuration."""
        # Extract unique metric names
        metric_names = list(set(m["name"] for m in metrics))
        
        panels = []
        panel_id = 1
        y_pos = 0
        
        for i, metric_name in enumerate(metric_names[:6]):  # Limit to 6 panels
            panel = {
                "id": panel_id,
                "title": metric_name.replace("_", " ").title(),
                "type": "graph" if "rate" in metric_name or "duration" in metric_name else "stat",
                "gridPos": {
                    "h": 6,
                    "w": 8,
                    "x": (i % 3) * 8,
                    "y": y_pos
                },
                "targets": [
                    {
                        "expr": metric_name,
                        "legendFormat": metric_name,
                        "refId": "A"
                    }
                ]
            }
            
            panels.append(panel)
            panel_id += 1
            
            if (i + 1) % 3 == 0:
                y_pos += 6
        
        return {
            "dashboard": {
                "id": None,
                "title": "Test Dashboard - Phase 4 Observability",
                "tags": ["test", "phase4", "observability"],
                "style": "dark",
                "timezone": "browser",
                "panels": panels,
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
    
    @staticmethod
    def generate_load_test_scenario() -> Dict[str, Any]:
        """Generate a load test scenario with high-volume data."""
        return {
            "duration_seconds": 300,  # 5 minutes
            "operations_per_second": 100,
            "metrics_to_generate": [
                "chunking_operations_total",
                "chunking_duration_ms", 
                "system_cpu_percent",
                "system_memory_percent"
            ],
            "health_checks": [
                "chunking_service",
                "cache_system",
                "database"
            ],
            "expected_outcomes": {
                "max_response_time_ms": 100,
                "max_memory_increase_mb": 50,
                "min_success_rate": 0.95
            }
        }


def load_sample_data() -> Dict[str, Any]:
    """Load all sample data for tests."""
    generator = TestDataGenerator()
    
    return {
        "metrics": generator.generate_sample_metrics(50),
        "health_checks": generator.generate_health_check_results(),
        "prometheus_output": generator.generate_prometheus_metrics_output(
            generator.generate_sample_metrics(20)
        ),
        "grafana_config": generator.generate_grafana_dashboard_config(
            generator.generate_sample_metrics(10)
        ),
        "load_test_scenario": generator.generate_load_test_scenario()
    }


if __name__ == "__main__":
    # Generate and save sample data
    data = load_sample_data()
    
    with open("generated_test_data.json", "w") as f:
        json.dump(data, f, indent=2, default=str)
    
    print("Generated test data saved to generated_test_data.json")