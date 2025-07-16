#!/usr/bin/env python3
"""
Example script demonstrating the lightweight health check approach.
This shows how to use the CHUNKING_SYSTEM_LIGHTWEIGHT_HEALTH_CHECK environment variable
to speed up health checks during performance testing or in environments where
full DocumentChunker initialization isn't needed.
"""

import os
import time
from src.utils.monitoring import SystemMonitor

def demonstrate_health_check_performance():
    """Demonstrate the performance difference between full and lightweight health checks."""
    
    print("=== DocumentChunker Health Check Performance Comparison ===\n")
    
    # Test 1: Full health check (default behavior)
    print("1. Full Health Check (default)")
    os.environ.pop('CHUNKING_SYSTEM_LIGHTWEIGHT_HEALTH_CHECK', None)
    
    start_time = time.time()
    monitor = SystemMonitor()
    result = monitor.health_checker.run_check("application_status")
    end_time = time.time()
    
    print(f"   Result: {result.status}")
    print(f"   Message: {result.message}")
    print(f"   Time: {(end_time - start_time) * 1000:.2f}ms")
    print(f"   Lightweight check: {result.details.get('lightweight_check', False)}")
    print()
    
    # Test 2: Lightweight health check
    print("2. Lightweight Health Check (import-only)")
    os.environ['CHUNKING_SYSTEM_LIGHTWEIGHT_HEALTH_CHECK'] = 'true'
    
    start_time = time.time()
    monitor = SystemMonitor()
    result = monitor.health_checker.run_check("application_status")
    end_time = time.time()
    
    print(f"   Result: {result.status}")
    print(f"   Message: {result.message}")
    print(f"   Time: {(end_time - start_time) * 1000:.2f}ms")
    print(f"   Lightweight check: {result.details.get('lightweight_check', False)}")
    print()
    
    # Clean up
    os.environ.pop('CHUNKING_SYSTEM_LIGHTWEIGHT_HEALTH_CHECK', None)
    
    print("=== Key Differences ===")
    print("- Full health check: Instantiates DocumentChunker and CacheManager")
    print("- Lightweight health check: Only verifies imports work")
    print("- Use lightweight mode for:")
    print("  * Performance testing")
    print("  * CI/CD environments")
    print("  * High-frequency health checks")
    print("  * Container readiness probes")

if __name__ == "__main__":
    demonstrate_health_check_performance()