"""
Performance monitoring and optimization utilities for the document chunking system.

This module provides tools for monitoring system performance, memory usage,
and optimizing processing workflows.
"""

import time
import psutil
import gc
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
from src.utils.logger import get_logger


@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics."""
    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_percent: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Performance monitoring utility with memory and CPU tracking.
    """
    
    def __init__(self, enable_detailed_monitoring: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            enable_detailed_monitoring: Whether to track detailed metrics
        """
        self.logger = get_logger(__name__)
        self.enable_detailed = enable_detailed_monitoring
        self.metrics: List[PerformanceMetrics] = []
        self.current_operations: Dict[str, PerformanceMetrics] = {}
        self._memory_tracker_active = False
        self._peak_memory = 0.0
        
    def start_monitoring(self, operation: str, **custom_metrics) -> str:
        """
        Start monitoring an operation.
        
        Args:
            operation: Name of the operation
            **custom_metrics: Additional metrics to track
        
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation}_{int(time.time() * 1000)}"
        
        metrics = PerformanceMetrics(
            operation=operation,
            start_time=datetime.now(),
            memory_before_mb=self._get_memory_usage_mb(),
            custom_metrics=custom_metrics
        )
        
        if self.enable_detailed:
            metrics.cpu_percent = psutil.cpu_percent()
        
        self.current_operations[operation_id] = metrics
        
        # Start memory tracking for this operation
        if self.enable_detailed and not self._memory_tracker_active:
            self._start_memory_tracking()
        
        self.logger.debug(
            "Started monitoring operation",
            operation=operation,
            operation_id=operation_id,
            memory_mb=metrics.memory_before_mb
        )
        
        return operation_id
    
    def end_monitoring(self, operation_id: str, success: bool = True, error_message: Optional[str] = None, **custom_metrics):
        """
        End monitoring an operation.
        
        Args:
            operation_id: Operation ID from start_monitoring
            success: Whether the operation succeeded
            error_message: Error message if operation failed
            **custom_metrics: Additional metrics to record
        """
        if operation_id not in self.current_operations:
            self.logger.warning("Unknown operation ID", operation_id=operation_id)
            return
        
        metrics = self.current_operations[operation_id]
        metrics.end_time = datetime.now()
        metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000
        metrics.memory_after_mb = self._get_memory_usage_mb()
        metrics.memory_peak_mb = self._peak_memory
        metrics.success = success
        metrics.error_message = error_message
        metrics.custom_metrics.update(custom_metrics)
        
        if self.enable_detailed:
            metrics.cpu_percent = psutil.cpu_percent()
        
        # Store completed metrics
        self.metrics.append(metrics)
        del self.current_operations[operation_id]
        
        # Stop memory tracking if no active operations
        if not self.current_operations and self._memory_tracker_active:
            self._stop_memory_tracking()
        
        self.logger.info(
            "Completed operation monitoring",
            operation=metrics.operation,
            duration_ms=metrics.duration_ms,
            memory_before_mb=metrics.memory_before_mb,
            memory_after_mb=metrics.memory_after_mb,
            memory_peak_mb=metrics.memory_peak_mb,
            success=success
        )
    
    @contextmanager
    def monitor_operation(self, operation: str, **custom_metrics):
        """
        Context manager for monitoring operations.
        
        Args:
            operation: Name of the operation
            **custom_metrics: Additional metrics to track
        
        Usage:
            with monitor.monitor_operation("chunking", file_size=1000):
                # perform operation
                pass
        """
        operation_id = self.start_monitoring(operation, **custom_metrics)
        success = True
        error_message = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            self.end_monitoring(operation_id, success=success, error_message=error_message)
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _start_memory_tracking(self):
        """Start background memory tracking."""
        self._memory_tracker_active = True
        self._peak_memory = self._get_memory_usage_mb()
        
        def track_memory():
            while self._memory_tracker_active:
                current_memory = self._get_memory_usage_mb()
                if current_memory > self._peak_memory:
                    self._peak_memory = current_memory
                time.sleep(0.1)  # Check every 100ms
        
        threading.Thread(target=track_memory, daemon=True).start()
    
    def _stop_memory_tracking(self):
        """Stop background memory tracking."""
        self._memory_tracker_active = False
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """
        Get statistics for a specific operation type.
        
        Args:
            operation: Operation name
        
        Returns:
            Dictionary with operation statistics
        """
        operation_metrics = [m for m in self.metrics if m.operation == operation]
        
        if not operation_metrics:
            return {}
        
        durations = [m.duration_ms for m in operation_metrics]
        memory_usage = [m.memory_after_mb - m.memory_before_mb for m in operation_metrics]
        success_count = sum(1 for m in operation_metrics if m.success)
        
        return {
            'operation': operation,
            'total_runs': len(operation_metrics),
            'success_rate': success_count / len(operation_metrics),
            'avg_duration_ms': sum(durations) / len(durations),
            'min_duration_ms': min(durations),
            'max_duration_ms': max(durations),
            'avg_memory_delta_mb': sum(memory_usage) / len(memory_usage),
            'max_memory_delta_mb': max(memory_usage),
            'total_duration_ms': sum(durations)
        }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        if not self.metrics:
            return {}
        
        total_duration = sum(m.duration_ms for m in self.metrics)
        successful_operations = sum(1 for m in self.metrics if m.success)
        
        return {
            'total_operations': len(self.metrics),
            'successful_operations': successful_operations,
            'success_rate': successful_operations / len(self.metrics),
            'total_duration_ms': total_duration,
            'avg_duration_ms': total_duration / len(self.metrics),
            'peak_memory_mb': max((m.memory_peak_mb for m in self.metrics), default=0),
            'operations_by_type': {
                op: self.get_operation_stats(op)
                for op in set(m.operation for m in self.metrics)
            }
        }
    
    def generate_performance_report(self) -> str:
        """Generate a performance report in markdown format."""
        stats = self.get_overall_stats()
        
        if not stats:
            return "# Performance Report\n\nNo performance data available."
        
        report = f"""# Performance Report
Generated: {datetime.now().isoformat()}

## Overall Statistics
- **Total Operations**: {stats['total_operations']}
- **Success Rate**: {stats['success_rate']:.2%}
- **Total Duration**: {stats['total_duration_ms']:.2f} ms
- **Average Duration**: {stats['avg_duration_ms']:.2f} ms
- **Peak Memory Usage**: {stats['peak_memory_mb']:.2f} MB

## Operations Breakdown
"""
        
        for operation, op_stats in stats['operations_by_type'].items():
            report += f"""
### {operation.title()}
- **Runs**: {op_stats['total_runs']}
- **Success Rate**: {op_stats['success_rate']:.2%}
- **Average Duration**: {op_stats['avg_duration_ms']:.2f} ms
- **Duration Range**: {op_stats['min_duration_ms']:.2f} - {op_stats['max_duration_ms']:.2f} ms
- **Average Memory Impact**: {op_stats['avg_memory_delta_mb']:.2f} MB
- **Max Memory Impact**: {op_stats['max_memory_delta_mb']:.2f} MB
"""
        
        return report
    
    def clear_metrics(self):
        """Clear stored metrics."""
        self.metrics.clear()
        self.logger.debug("Performance metrics cleared")


class MemoryOptimizer:
    """
    Memory optimization utilities.
    """
    
    def __init__(self, auto_cleanup_threshold_mb: float = 500.0):
        """
        Initialize memory optimizer.
        
        Args:
            auto_cleanup_threshold_mb: Memory threshold for automatic cleanup
        """
        self.logger = get_logger(__name__)
        self.auto_cleanup_threshold = auto_cleanup_threshold_mb
        self.cleanup_callbacks: List[Callable] = []
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback function."""
        self.cleanup_callbacks.append(callback)
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """
        Force garbage collection and return statistics.
        
        Returns:
            Dictionary with GC statistics
        """
        before_memory = self._get_memory_usage_mb()
        
        # Run garbage collection
        collected = {
            'gen0': gc.collect(0),
            'gen1': gc.collect(1), 
            'gen2': gc.collect(2)
        }
        
        after_memory = self._get_memory_usage_mb()
        freed_mb = before_memory - after_memory
        
        self.logger.info(
            "Garbage collection completed",
            collected_objects=sum(collected.values()),
            memory_freed_mb=freed_mb,
            memory_before_mb=before_memory,
            memory_after_mb=after_memory
        )
        
        return {
            **collected,
            'total_collected': sum(collected.values()),
            'memory_freed_mb': freed_mb
        }
    
    def cleanup_if_needed(self) -> bool:
        """
        Perform cleanup if memory usage exceeds threshold.
        
        Returns:
            True if cleanup was performed
        """
        current_memory = self._get_memory_usage_mb()
        
        if current_memory > self.auto_cleanup_threshold:
            self.logger.warning(
                "Memory threshold exceeded, performing cleanup",
                current_memory_mb=current_memory,
                threshold_mb=self.auto_cleanup_threshold
            )
            
            # Run cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.error("Cleanup callback failed", error=str(e))
            
            # Force garbage collection
            self.force_garbage_collection()
            
            return True
        
        return False
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    @contextmanager
    def memory_limit_context(self, max_memory_mb: float):
        """
        Context manager that monitors memory usage and raises exception if exceeded.
        
        Args:
            max_memory_mb: Maximum allowed memory usage
        """
        initial_memory = self._get_memory_usage_mb()
        
        try:
            yield
        finally:
            current_memory = self._get_memory_usage_mb()
            if current_memory > max_memory_mb:
                self.logger.error(
                    "Memory limit exceeded",
                    current_memory_mb=current_memory,
                    limit_mb=max_memory_mb,
                    initial_memory_mb=initial_memory
                )
                raise MemoryError(f"Memory usage ({current_memory:.1f} MB) exceeded limit ({max_memory_mb} MB)")


class BatchProcessor:
    """
    Optimized batch processing with performance monitoring.
    """
    
    def __init__(self, batch_size: int = 10, memory_optimizer: Optional[MemoryOptimizer] = None):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items to process in each batch
            memory_optimizer: Optional memory optimizer
        """
        self.batch_size = batch_size
        self.memory_optimizer = memory_optimizer or MemoryOptimizer()
        self.performance_monitor = PerformanceMonitor()
        self.logger = get_logger(__name__)
    
    def process_batches(
        self,
        items: List[Any],
        processor_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """
        Process items in optimized batches.
        
        Args:
            items: Items to process
            processor_func: Function to process each item
            progress_callback: Optional progress callback
        
        Returns:
            List of processed results
        """
        total_items = len(items)
        results = []
        
        with self.performance_monitor.monitor_operation("batch_processing", total_items=total_items):
            for i in range(0, total_items, self.batch_size):
                batch = items[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                total_batches = (total_items + self.batch_size - 1) // self.batch_size
                
                with self.performance_monitor.monitor_operation(
                    "batch_item_processing",
                    batch_number=batch_num,
                    batch_size=len(batch)
                ):
                    # Process batch items
                    batch_results = []
                    for item in batch:
                        try:
                            result = processor_func(item)
                            batch_results.append(result)
                        except Exception as e:
                            self.logger.error("Failed to process item", error=str(e), item=str(item)[:100])
                            batch_results.append(None)
                    
                    results.extend(batch_results)
                
                # Progress callback
                if progress_callback:
                    progress_callback(min(i + self.batch_size, total_items), total_items)
                
                # Memory cleanup after each batch
                self.memory_optimizer.cleanup_if_needed()
                
                self.logger.debug(
                    "Batch processed",
                    batch_number=batch_num,
                    total_batches=total_batches,
                    items_processed=len(batch_results),
                    memory_mb=self.memory_optimizer._get_memory_usage_mb()
                )
        
        return results
    
    def get_performance_report(self) -> str:
        """Get performance report for batch processing."""
        return self.performance_monitor.generate_performance_report()


# Global instances for convenience
default_performance_monitor = PerformanceMonitor()
default_memory_optimizer = MemoryOptimizer()
default_batch_processor = BatchProcessor()


def monitor_performance(operation: str, **custom_metrics):
    """
    Decorator for monitoring function performance.
    
    Usage:
        @monitor_performance("file_processing", file_type="markdown")
        def process_file(file_path):
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with default_performance_monitor.monitor_operation(operation, **custom_metrics):
                return func(*args, **kwargs)
        return wrapper
    return decorator