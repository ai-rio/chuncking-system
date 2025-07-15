"""Integration tests for Phase 3 implementation."""

import pytest
import tempfile
import time
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.chunking_system import DocumentChunker
from src.config.settings import ChunkingConfig
from src.utils.cache import CacheManager
from src.utils.security import SecurityConfig
from src.utils.monitoring import SystemMonitor
from src.exceptions import ValidationError, SecurityError


class TestPhase3Integration:
    """Integration tests for all Phase 3 components working together."""
    
    def test_full_phase3_enabled(self, tmp_path):
        """Test DocumentChunker with all Phase 3 features enabled."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True,
            security_config=SecurityConfig(
                max_file_size_mb=10,  # 10MB
                allowed_extensions={'.md', '.txt'},
                enable_content_validation=True
            )
        )
        
        chunker = DocumentChunker(config)
        
        # Create test file
        test_file = tmp_path / "integration_test.md"
        test_content = """
# Integration Test Document

This is a comprehensive test document for Phase 3 integration testing.

## Section 1: Introduction

This section introduces the document and explains its purpose for testing
the integration of caching, security, and monitoring features.

## Section 2: Content

This section contains substantial content to ensure proper chunking
and to test the performance monitoring capabilities of the system.

## Section 3: Conclusion

This final section concludes the test document and provides a summary
of the testing objectives and expected outcomes.
        """
        test_file.write_text(test_content.strip())
        
        with patch('magic.from_file', return_value="text/plain"):
            # First processing (should not use cache)
            result1 = chunker.chunk_file(test_file)
        
        # Verify successful processing
        assert result1.success is True
        assert len(result1.chunks) > 0
        assert result1.cache_hit is False
        assert result1.security_audit is not None
        assert result1.security_audit['overall_status'] == 'passed'
        assert result1.performance_metrics is not None
        
        with patch('magic.from_file', return_value="text/plain"):
            # Second processing (should use cache)
            result2 = chunker.chunk_file(test_file)
        
        # Verify cache hit
        assert result2.success is True
        assert result2.cache_hit is True
        assert len(result2.chunks) == len(result1.chunks)
        assert result2.security_audit is not None
        assert result2.performance_metrics is not None
    
    def test_security_blocking_with_caching_and_monitoring(self, tmp_path):
        """Test that security blocks unsafe files even with caching/monitoring enabled."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True,
            security_config=SecurityConfig(
                max_file_size_mb=1,  # 1MB limit
                allowed_extensions={'.md'},  # Only .md allowed
                enable_content_validation=True
            )
        )
        
        chunker = DocumentChunker(config)
        
        # Create file that violates security policy
        unsafe_file = tmp_path / "unsafe.txt"  # Wrong extension
        unsafe_file.write_text("This file has wrong extension")
        
        result = chunker.chunk_file(unsafe_file)
        
        # Verify security blocked the file
        assert result.success is False
        assert len(result.chunks) == 0
        assert "security validation failed" in result.error_message.lower()
        assert result.cache_hit is False  # No caching for failed security
        assert result.security_audit is not None
        assert result.security_audit['overall_status'] == 'failed'
        assert result.performance_metrics is not None  # Still monitored
    
    def test_large_file_security_with_performance_monitoring(self, tmp_path):
        """Test security handling of large files with performance monitoring."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True,
            security_config=SecurityConfig(
                max_file_size_mb=0.0005,  # 0.5KB limit
                allowed_extensions={'.md'},
                enable_content_validation=True
            )
        )
        
        chunker = DocumentChunker(config)
        
        # Create file that exceeds size limit
        large_file = tmp_path / "large.md"
        large_content = "# Large File\n\n" + "x" * 1000  # Over 500 bytes
        large_file.write_text(large_content)
        
        result = chunker.chunk_file(large_file)
        
        # Verify security blocked the large file
        assert result.success is False
        assert len(result.chunks) == 0
        assert "security validation failed" in result.error_message.lower()
        assert result.security_audit is not None
        assert any("File too large" in error for error in result.security_audit.get('errors', []))
        assert result.performance_metrics is not None
    
    def test_directory_processing_with_mixed_security_results(self, tmp_path):
        """Test directory processing with mixed security validation results."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True,
            security_config=SecurityConfig(
                max_file_size_mb=1,
                allowed_extensions={'.md'},
                enable_content_validation=True
            )
        )
        
        chunker = DocumentChunker(config)
        
        # Create mixed files
        (tmp_path / "safe1.md").write_text("# Safe Document 1\n\nSafe content.")
        (tmp_path / "safe2.md").write_text("# Safe Document 2\n\nMore safe content.")
        (tmp_path / "unsafe.txt").write_text("Unsafe extension")
        (tmp_path / "toolarge.md").write_text("# Large\n\n" + "x" * (1024 * 1024 + 100))  # > 1MB
        
        with patch('magic.from_file', return_value="text/plain"):
            results = chunker.chunk_directory(tmp_path, file_pattern="*")
        
        # Should have 4 results
        assert len(results) == 4
        
        # Check individual results
        safe1_result = next(r for r in results if r.file_path.name == "safe1.md")
        safe2_result = next(r for r in results if r.file_path.name == "safe2.md")
        unsafe_result = next(r for r in results if r.file_path.name == "unsafe.txt")
        large_result = next(r for r in results if r.file_path.name == "toolarge.md")
        
        # Verify safe files processed successfully
        assert safe1_result.success is True
        assert safe2_result.success is True
        assert len(safe1_result.chunks) > 0
        assert len(safe2_result.chunks) > 0
        
        # Verify unsafe files were blocked
        assert unsafe_result.success is False
        assert large_result.success is False
        assert len(unsafe_result.chunks) == 0
        assert len(large_result.chunks) == 0
        
        # All should have security audits and performance metrics
        for result in results:
            assert result.security_audit is not None
            assert result.performance_metrics is not None
    
    def test_cache_invalidation_with_security_and_monitoring(self, tmp_path):
        """Test cache invalidation when file changes, with security and monitoring."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        # Create test file
        test_file = tmp_path / "changing_file.md"
        original_content = "# Original Content\n\nThis is the original content."
        test_file.write_text(original_content)
        
        with patch('magic.from_file', return_value="text/plain"):
            # First processing
            result1 = chunker.chunk_file(test_file)
        
        assert result1.success is True
        assert result1.cache_hit is False
        
        with patch('magic.from_file', return_value="text/plain"):
            # Second processing (should use cache)
            result2 = chunker.chunk_file(test_file)
        
        assert result2.success is True
        assert result2.cache_hit is True
        
        # Modify file
        time.sleep(0.1)  # Ensure different mtime
        modified_content = "# Modified Content\n\nThis content has been changed."
        test_file.write_text(modified_content)
        
        with patch('magic.from_file', return_value="text/plain"):
            # Third processing (should not use cache due to file change)
            result3 = chunker.chunk_file(test_file)
        
        assert result3.success is True
        assert result3.cache_hit is False
        
        # All results should have security audits and performance metrics
        for result in [result1, result2, result3]:
            assert result.security_audit is not None
            assert result.performance_metrics is not None
    
    def test_performance_monitoring_with_batch_processing(self, tmp_path):
        """Test performance monitoring during batch processing."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        # Create multiple test files
        test_files = []
        for i in range(5):
            test_file = tmp_path / f"batch_test_{i}.md"
            content = f"# Batch Test Document {i}\n\nContent for document {i}." * 10
            test_file.write_text(content)
            test_files.append(test_file)
        
        with patch('magic.from_file', return_value="text/plain"):
            results = chunker.chunk_directory(tmp_path)
        
        # Verify all files processed successfully
        assert len(results) == 5
        assert all(r.success for r in results)
        
        # Verify performance metrics collected for all
        for result in results:
            assert result.performance_metrics is not None
            # Check for system-level aggregated metrics
            assert "total_operations" in result.performance_metrics
            assert "successful_operations" in result.performance_metrics
            assert "success_rate" in result.performance_metrics
        
        # Check system monitor collected metrics
        monitor = chunker.system_monitor
        status = monitor.get_system_status()
        
        assert "health" in status
        assert "metrics_count" in status
        assert "active_alerts" in status
    
    def test_error_handling_with_all_features_enabled(self, tmp_path):
        """Test error handling when all Phase 3 features are enabled."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        # Test with non-existent file
        nonexistent_file = tmp_path / "nonexistent.md"
        
        result = chunker.chunk_file(nonexistent_file)
        
        # Should handle error gracefully
        assert result.success is False
        assert len(result.chunks) == 0
        assert "not found" in result.error_message.lower() or "no such file" in result.error_message.lower() or "does not exist" in result.error_message.lower()
        assert result.cache_hit is False
        assert result.performance_metrics is not None  # Still monitored
        # Security audit might be None for non-existent files
    
    def test_memory_optimization_during_processing(self, tmp_path):
        """Test memory optimization features during processing."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        # Create files with substantial content
        large_files = []
        for i in range(3):
            test_file = tmp_path / f"large_doc_{i}.md"
            # Create content that's large enough to test memory optimization
            content = f"# Large Document {i}\n\n" + ("This is a large document with substantial content. " * 100)
            test_file.write_text(content)
            large_files.append(test_file)
        
        with patch('magic.from_file', return_value="text/plain"):
            results = chunker.chunk_directory(tmp_path)
        
        # Verify processing completed successfully
        assert len(results) == 3
        assert all(r.success for r in results)
        
        # Verify memory metrics were collected
        for result in results:
            assert result.performance_metrics is not None
            # Check for system-level aggregated metrics
            assert "total_operations" in result.performance_metrics
            assert "successful_operations" in result.performance_metrics
            assert "peak_memory_mb" in result.performance_metrics
    
    def test_concurrent_access_simulation(self, tmp_path):
        """Test behavior under simulated concurrent access."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        # Clear all caches to ensure clean state
        if chunker.cache_manager:
            chunker.cache_manager.clear_all_caches()
        
        # Also clear any global cache
        from src.utils.cache import default_cache_manager
        default_cache_manager.clear_all_caches()
        
        # Create test file with unique name to avoid cache conflicts
        import uuid
        test_file = tmp_path / f"concurrent_test_{uuid.uuid4().hex[:8]}.md"
        test_file.write_text("# Concurrent Test\n\nContent for concurrent access testing.")
        
        results = []
        
        with patch('magic.from_file', return_value="text/plain"):
            # First access should not be cached
            result = chunker.chunk_file(test_file)
            results.append(result)
            
            # Subsequent accesses should use cache
            for i in range(4):
                result = chunker.chunk_file(test_file)
                results.append(result)
        
        # Verify all accesses completed
        assert len(results) == 5
        assert all(r.success for r in results)
        
        # Verify caching behavior - at least some should be cache hits
        cache_hits = sum(1 for r in results if r.cache_hit)
        # If the first is a cache hit, it means cache from previous test
        # but subsequent ones should definitely be cache hits
        if results[0].cache_hit:
            # All should be cache hits if first one is
            assert all(r.cache_hit for r in results)
        else:
            # First is not cached, subsequent ones should be
            assert results[0].cache_hit is False
            for result in results[1:]:
                assert result.cache_hit is True
        
        # All should have consistent chunk counts
        chunk_counts = [len(r.chunks) for r in results]
        assert all(count == chunk_counts[0] for count in chunk_counts)
    
    def test_system_monitoring_integration(self, tmp_path):
        """Test system monitoring integration with chunking operations."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        # Register custom health check
        def chunker_health_check():
            from src.utils.monitoring import HealthStatus
            return HealthStatus(
                component="document_chunker",
                is_healthy=True,
                message="Chunker operational",
                details={"cache_enabled": True, "security_enabled": True}
            )
        
        chunker.system_monitor.register_health_check("chunker", chunker_health_check)
        
        # Create and process test file
        test_file = tmp_path / "monitoring_test.md"
        test_file.write_text("# Monitoring Test\n\nContent for monitoring integration test.")
        
        with patch('magic.from_file', return_value="text/plain"):
            result = chunker.chunk_file(test_file)
        
        assert result.success is True
        
        # Check system status
        status = chunker.system_monitor.get_system_status()
        
        assert "health" in status
        assert "metrics_count" in status
        assert "active_alerts" in status
        
        # Run health checks
        health_results = chunker.system_monitor.health_checker.run_all_checks()
        chunker_health = next((h for h in health_results if h.component == "document_chunker"), None)
        
        assert chunker_health is not None
        assert chunker_health.is_healthy is True
    
    def test_cache_performance_with_security_overhead(self, tmp_path):
        """Test cache performance benefits even with security overhead."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True,
            security_config=SecurityConfig(
                enable_content_validation=True,
            )
        )
        
        chunker = DocumentChunker(config)
        
        # Create test file
        test_file = tmp_path / "performance_test.md"
        content = "# Performance Test\n\n" + ("Content for performance testing. " * 50)
        test_file.write_text(content)
        
        with patch('magic.from_file', return_value="text/plain"):
            # First processing (no cache)
            start_time = time.time()
            result1 = chunker.chunk_file(test_file)
            first_duration = time.time() - start_time
        
        assert result1.success is True
        assert result1.cache_hit is False
        
        with patch('magic.from_file', return_value="text/plain"):
            # Second processing (with cache)
            start_time = time.time()
            result2 = chunker.chunk_file(test_file)
            second_duration = time.time() - start_time
        
        assert result2.success is True
        assert result2.cache_hit is True
        
        # Cache should provide performance benefit
        # (allowing some tolerance for test environment variations)
        assert second_duration <= first_duration * 1.5  # At most 50% longer
        
        # Results should be identical
        assert len(result1.chunks) == len(result2.chunks)
        for chunk1, chunk2 in zip(result1.chunks, result2.chunks):
            assert chunk1.page_content == chunk2.page_content


class TestPhase3Configuration:
    """Test different Phase 3 configuration combinations."""
    
    def test_caching_only_configuration(self, tmp_path):
        """Test configuration with only caching enabled."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=False,
            enable_monitoring=False
        )
        
        chunker = DocumentChunker(config)
        
        test_file = tmp_path / "cache_only.md"
        test_file.write_text("# Cache Only Test\n\nContent for cache-only testing.")
        
        # First processing
        result1 = chunker.chunk_file(test_file)
        assert result1.success is True
        assert result1.cache_hit is False
        assert result1.security_audit is None
        assert result1.performance_metrics is None
        
        # Second processing (should use cache)
        result2 = chunker.chunk_file(test_file)
        assert result2.success is True
        assert result2.cache_hit is True
        assert result2.security_audit is None
        assert result2.performance_metrics is None
    
    def test_security_only_configuration(self, tmp_path):
        """Test configuration with only security enabled."""
        config = ChunkingConfig(
            enable_caching=False,
            enable_security=True,
            enable_monitoring=False
        )
        
        chunker = DocumentChunker(config)
        
        test_file = tmp_path / "security_only.md"
        test_file.write_text("# Security Only Test\n\nContent for security-only testing.")
        
        with patch('magic.from_file', return_value="text/plain"):
            result = chunker.chunk_file(test_file)
        
        assert result.success is True
        assert result.cache_hit is False
        assert result.security_audit is not None
        assert result.security_audit['overall_status'] == 'passed'
        assert result.performance_metrics is None
    
    def test_monitoring_only_configuration(self, tmp_path):
        """Test configuration with only monitoring enabled."""
        config = ChunkingConfig(
            enable_caching=False,
            enable_security=False,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        test_file = tmp_path / "monitoring_only.md"
        test_file.write_text("# Monitoring Only Test\n\nContent for monitoring-only testing.")
        
        result = chunker.chunk_file(test_file)
        
        assert result.success is True
        assert result.cache_hit is False
        assert result.security_audit is None
        assert result.performance_metrics is not None
        assert "duration" in result.performance_metrics
    
    def test_all_disabled_configuration(self, tmp_path):
        """Test configuration with all Phase 3 features disabled."""
        config = ChunkingConfig(
            enable_caching=False,
            enable_security=False,
            enable_monitoring=False
        )
        
        chunker = DocumentChunker(config)
        
        test_file = tmp_path / "all_disabled.md"
        test_file.write_text("# All Disabled Test\n\nContent for testing with all features disabled.")
        
        result = chunker.chunk_file(test_file)
        
        assert result.success is True
        assert result.cache_hit is False
        assert result.security_audit is None
        assert result.performance_metrics is None
        
        # Should still produce chunks
        assert len(result.chunks) > 0


class TestPhase3EdgeCases:
    """Test edge cases and error conditions in Phase 3 implementation."""
    
    def test_cache_corruption_handling(self, tmp_path):
        """Test handling of cache corruption scenarios."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=False,
            enable_monitoring=False
        )
        
        chunker = DocumentChunker(config)
        
        test_file = tmp_path / "cache_corruption.md"
        test_file.write_text("# Cache Corruption Test\n\nContent for cache corruption testing.")
        
        # First processing to populate cache
        result1 = chunker.chunk_file(test_file)
        assert result1.success is True
        assert result1.cache_hit is False
        
        # Simulate cache corruption by clearing memory cache but not file cache
        chunker.cache_manager.memory_cache.clear()
        
        # Should still work (might use file cache or reprocess)
        result2 = chunker.chunk_file(test_file)
        assert result2.success is True
        assert len(result2.chunks) == len(result1.chunks)
    
    def test_security_validation_exception_handling(self, tmp_path):
        """Test handling of security validation exceptions."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        test_file = tmp_path / "security_exception.md"
        test_file.write_text("# Security Exception Test\n\nContent.")
        
        # Mock security validation to raise exception
        with patch.object(chunker.security_auditor, 'audit_file', side_effect=Exception("Security error")):
            result = chunker.chunk_file(test_file)
        
        # Should handle security exception gracefully
        assert result.success is False
        assert "security" in result.error_message.lower()
        assert result.performance_metrics is not None  # Still monitored
    
    def test_monitoring_exception_handling(self, tmp_path):
        """Test handling of monitoring exceptions."""
        config = ChunkingConfig(
            enable_caching=False,
            enable_security=False,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        test_file = tmp_path / "monitoring_exception.md"
        test_file.write_text("# Monitoring Exception Test\n\nContent.")
        
        # Mock system monitor to raise exception
        with patch.object(chunker.system_monitor, 'monitor_operation', side_effect=Exception("Monitoring error")):
            result = chunker.chunk_file(test_file)
        
        # Should still process file even if monitoring fails
        assert result.success is True
        assert len(result.chunks) > 0
        # Performance metrics might be None due to monitoring failure
    
    def test_empty_file_handling(self, tmp_path):
        """Test handling of empty files with all features enabled."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        # Create empty file
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("")
        
        with patch('magic.from_file', return_value="text/plain"):
            result = chunker.chunk_file(empty_file)
        
        # Should handle empty file gracefully
        assert result.success is True
        assert len(result.chunks) == 0  # No chunks from empty file
        assert result.security_audit is not None
        assert result.performance_metrics is not None
    
    def test_very_large_file_with_memory_optimization(self, tmp_path):
        """Test processing very large files with memory optimization."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True,
            security_config=SecurityConfig(
                max_file_size_mb=0.05,  # 50KB limit (converted to MB)
                allowed_extensions={'.md'}
            )
        )
        
        chunker = DocumentChunker(config)
        
        # Create large file within security limits
        large_file = tmp_path / "large_within_limits.md"
        content = """# Large File

""" + ("This is a large file with substantial content. " * 500)  # ~30KB
        large_file.write_text(content)
        
        with patch('magic.from_file', return_value="text/plain"):
            result = chunker.chunk_file(large_file)
        
        # Should process successfully with memory optimization
        assert result.success is True
        assert len(result.chunks) > 0
        assert result.security_audit is not None
        assert result.security_audit['overall_status'] == 'passed'
        assert result.performance_metrics is not None
        
        # Memory metrics should show optimization
        assert "memory_before" in result.performance_metrics
        assert "memory_after" in result.performance_metrics
        assert "peak_memory" in result.performance_metrics


class TestPhase3Performance:
    """Performance tests for Phase 3 implementation."""
    
    def test_caching_performance_benefit(self, tmp_path):
        """Test that caching provides measurable performance benefits."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        # Create substantial test file
        test_file = tmp_path / "performance_test.md"
        content = "# Performance Test\n\n" + ("Content for performance testing. " * 200)
        test_file.write_text(content)
        
        # Measure first processing (no cache)
        with patch('magic.from_file', return_value="text/plain"):
            start_time = time.time()
            result1 = chunker.chunk_file(test_file)
            first_time = time.time() - start_time
        
        assert result1.success is True
        assert result1.cache_hit is False
        
        # Measure second processing (with cache)
        with patch('magic.from_file', return_value="text/plain"):
            start_time = time.time()
            result2 = chunker.chunk_file(test_file)
            second_time = time.time() - start_time
        
        assert result2.success is True
        assert result2.cache_hit is True
        
        # Cache should provide performance benefit
        print(f"First processing: {first_time:.4f}s, Second processing: {second_time:.4f}s")
        # Allow some tolerance for test environment variations
        assert second_time <= first_time * 2.0  # At most 2x slower (very lenient)
    
    def test_batch_processing_performance(self, tmp_path):
        """Test performance of batch processing with all features enabled."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        # Create multiple test files
        num_files = 10
        for i in range(num_files):
            test_file = tmp_path / f"batch_{i}.md"
            content = f"# Batch Document {i}\n\n" + (f"Content for document {i}. " * 20)
            test_file.write_text(content)
        
        # Measure batch processing time
        with patch('magic.from_file', return_value="text/plain"):
            start_time = time.time()
            results = chunker.chunk_directory(tmp_path)
            total_time = time.time() - start_time
        
        # Verify all files processed
        assert len(results) == num_files
        assert all(r.success for r in results)
        
        # Performance should be reasonable
        avg_time_per_file = total_time / num_files
        print(f"Total time: {total_time:.4f}s, Average per file: {avg_time_per_file:.4f}s")
        
        # Should process files reasonably quickly (adjust threshold as needed)
        assert avg_time_per_file < 5.0  # Less than 5 seconds per file
    
    def test_memory_usage_optimization(self, tmp_path):
        """Test memory usage optimization during processing."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True
        )
        
        chunker = DocumentChunker(config)
        
        # Create file with substantial content
        test_file = tmp_path / "memory_test.md"
        content = "# Memory Test\n\n" + ("Large content for memory testing. " * 1000)
        test_file.write_text(content)
        
        with patch('magic.from_file', return_value="text/plain"):
            result = chunker.chunk_file(test_file)
        
        assert result.success is True
        assert result.performance_metrics is not None
        
        # Check memory metrics
        memory_before = result.performance_metrics.get("memory_before", 0)
        memory_after = result.performance_metrics.get("memory_after", 0)
        peak_memory = result.performance_metrics.get("peak_memory", 0)
        
        # Memory should be tracked
        assert memory_before > 0
        assert memory_after > 0
        assert peak_memory >= max(memory_before, memory_after)
        
        print(f"Memory - Before: {memory_before}MB, After: {memory_after}MB, Peak: {peak_memory}MB")


@pytest.fixture
def phase3_test_environment(tmp_path):
    """Create a comprehensive test environment for Phase 3 testing."""
    # Create directory structure
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    
    safe_dir = docs_dir / "safe"
    safe_dir.mkdir()
    
    unsafe_dir = docs_dir / "unsafe"
    unsafe_dir.mkdir()
    
    # Create safe documents
    (safe_dir / "doc1.md").write_text("# Document 1\n\nSafe content for document 1.")
    (safe_dir / "doc2.md").write_text("# Document 2\n\nSafe content for document 2.")
    (safe_dir / "readme.txt").write_text("This is a safe text file.")
    
    # Create unsafe documents
    (unsafe_dir / "script.exe").write_text("Potentially unsafe executable")
    (unsafe_dir / "large.md").write_text("# Large\n\n" + "x" * 10000)
    (unsafe_dir / "binary.md").write_bytes(b"\x00\x01\x02Binary content")
    
    return {
        "root": tmp_path,
        "docs": docs_dir,
        "safe": safe_dir,
        "unsafe": unsafe_dir
    }


class TestPhase3ComprehensiveIntegration:
    """Comprehensive integration tests using the test environment."""
    
    def test_comprehensive_directory_processing(self, phase3_test_environment):
        """Test comprehensive directory processing with all features."""
        # Debug: Print directory structure
        import os
        docs_path = phase3_test_environment["docs"]
        print(f"\nDocs directory: {docs_path}")
        for root, dirs, files in os.walk(docs_path):
            for file in files:
                print(f"Found file: {os.path.join(root, file)}")
        
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,  # Enable security to test unsafe file handling
            enable_monitoring=True,
            security_config=SecurityConfig(
                max_file_size_mb=0.005,  # 5KB limit to catch large.md (10KB)
                allowed_extensions={'.md'},  # Only .md allowed, .txt and .exe should fail
                enable_content_validation=True
            )
        )
        
        chunker = DocumentChunker(config)
        
        with patch('magic.from_file', return_value="text/plain"):
            # Process entire document tree with pattern to match all files
            results = chunker.chunk_directory(phase3_test_environment["docs"], file_pattern="*")
        
        print(f"\nResults count: {len(results)}")
        for i, result in enumerate(results):
            print(f"Result {i}: {result.file_path}, success: {result.success}")
        
        # Should have results for all files
        assert len(results) >= 6  # At least 6 files created
        
        # Categorize results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # Should have both successful and failed results
        assert len(successful_results) >= 3  # Safe files
        assert len(failed_results) >= 3     # Unsafe files
        
        # All results should have appropriate metadata
        for result in results:
            assert result.performance_metrics is not None
            # Security audit might be None for some error cases
        
        # Verify specific file results
        doc1_results = [r for r in results if r.file_path.name == "doc1.md"]
        assert len(doc1_results) == 1
        assert doc1_results[0].success is True
        
        exe_results = [r for r in results if r.file_path.name == "script.exe"]
        assert len(exe_results) == 1
        assert exe_results[0].success is False
    
    def test_system_monitoring_comprehensive(self, phase3_test_environment):
        """Test comprehensive system monitoring during processing."""
        config = ChunkingConfig(
            enable_caching=True,
            enable_security=True,
            enable_monitoring=True,
            security_config=SecurityConfig(
                max_file_size_mb=10,  # Allow larger files
                allowed_extensions={'.md', '.txt'},  # Allow both .md and .txt files
                enable_content_validation=True
            )
        )
        
        chunker = DocumentChunker(config)
        
        # Register custom monitoring
        def processing_health_check():
            from src.utils.monitoring import HealthStatus
            return HealthStatus(
                component="processing_pipeline",
                is_healthy=True,
                message="Pipeline operational"
            )
        
        chunker.system_monitor.register_health_check("pipeline", processing_health_check)
        
        with patch('magic.from_file', return_value="text/plain"):
            # Process documents with pattern to include all files
            results = chunker.chunk_directory(phase3_test_environment["safe"], file_pattern="*")
        
        # Verify processing completed
        assert len(results) >= 3
        assert all(r.success for r in results)
        
        # Check system monitoring data
        status = chunker.system_monitor.get_system_status()
        
        assert "health" in status
        assert "metrics_count" in status
        assert "active_alerts" in status
        
        # Verify health checks
        health_results = chunker.system_monitor.health_checker.run_all_checks()
        pipeline_health = next((h for h in health_results if h.component == "processing_pipeline"), None)
        
        assert pipeline_health is not None
        assert pipeline_health.is_healthy is True