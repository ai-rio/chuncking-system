#!/usr/bin/env python3

from src.utils.monitoring import AlertManager, Alert, MetricPoint

# Create AlertManager
manager = AlertManager()

# Define the rule function
def memory_rule(context):
    print(f"Rule called with context: {context}")
    metrics = context.get("metrics", [])
    print(f"Extracted metrics: {metrics}")
    memory_metrics = [m for m in metrics if m.name == "memory_usage"]
    print(f"Memory metrics found: {memory_metrics}")
    if memory_metrics and memory_metrics[-1].value > 80:
        print("Creating alert...")
        alert = Alert(
            id="memory_high",
            severity="warning",
            title="High Memory Usage",
            component="memory",
            message="High memory usage"
        )
        print(f"Created alert: {alert}")
        return alert
    print("No alert created")
    return None

# Add the rule
manager.add_alert_rule(memory_rule)
print(f"Alert rules: {manager.alert_rules}")

# Create test metrics
test_metrics = [
    MetricPoint("memory_usage", 85, "percent"),
    MetricPoint("cpu_usage", 50, "percent")
]
print(f"Test metrics: {test_metrics}")

# Evaluate rules
print("\nEvaluating rules...")
manager.evaluate_rules({"metrics": test_metrics})

print(f"\nAlerts after evaluation: {len(manager.alerts)}")
for alert in manager.alerts:
    print(f"Alert: {alert}")