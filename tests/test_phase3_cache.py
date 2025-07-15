"""Tests for Phase 3 caching implementation."""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from src.utils.cache import CacheEntry, InMemoryCache, FileCache, CacheManager
from src.chunking_system import DocumentChunker, ChunkingConfig


class TestCacheEntry:
    """Test CacheEntry dataclass."""
    
    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            created_at=now,
            expires_at=now + timedelta(seconds=3600)
        )
        
        assert entry.key == "test_key"
        assert entry.value == {"data": "test"}
        assert entry.expires_at is not None
        assert not entry.is_expired()
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        from datetime import datetime, timedelta
        
        # Create expired entry
        now = datetime.now()
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=now - timedelta(hours=1, minutes=2),  # 1 hour and 2 minutes ago
            expires_at=now - timedelta(minutes=2)  # Expired 2 minutes ago
        )
        
        assert entry.is_expired()
    
    def test_cache_entry_no_expiration(self):
        """Test cache entry with no expiration."""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=now - timedelta(hours=2),  # 2 hours ago
            expires_at=None  # No expiration
        )
        
        assert not entry.is_expired()


class TestInMemoryCache:
    """Test InMemoryCache implementation."""
    
    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = InMemoryCache(max_size=100, default_ttl_seconds=3600)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = InMemoryCache(max_size=100, default_ttl_seconds=1)  # 1 second TTL
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = InMemoryCache(max_size=2, default_ttl_seconds=3600)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_cache_update_access_time(self):
        """Test that accessing an item updates its access time."""
        cache = InMemoryCache(max_size=2, default_ttl_seconds=3600)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to update its access time
        cache.get("key1")
        
        # Add key3, which should evict key2 (least recently used)
        cache.put("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Should still be there
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") == "value3"
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = InMemoryCache(max_size=100, default_ttl_seconds=3600)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert len(cache.cache) == 2
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = InMemoryCache(max_size=100, default_ttl_seconds=3600)
        
        # Initial stats
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['entries'] == 0
        
        # Add some data and test
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['entries'] == 1


class TestFileCache:
    """Test FileCache implementation."""
    
    def test_file_cache_set_and_get(self, tmp_path):
        """Test basic file cache operations."""
        cache = FileCache(cache_dir=tmp_path, max_size_mb=100)
        
        test_data = {"key": "value", "number": 42}
        cache.put("test_key", test_data)
        
        retrieved_data = cache.get("test_key")
        assert retrieved_data == test_data
    
    def test_file_cache_expiration(self, tmp_path):
        """Test file cache expiration."""
        cache = FileCache(cache_dir=tmp_path, max_size_mb=100)
        
        cache.put("test_key", "test_value", ttl_seconds=1)  # 1 second TTL
        assert cache.get("test_key") == "test_value"
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("test_key") is None
    
    def test_file_cache_clear(self, tmp_path):
        """Test file cache clearing."""
        cache = FileCache(cache_dir=tmp_path, max_size_mb=100)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Check files exist
        assert len(list(tmp_path.glob("*.cache"))) == 2
        
        cache.clear()
        
        # Check files are removed
        assert len(list(tmp_path.glob("*.cache"))) == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestCacheManager:
    """Test CacheManager implementation."""
    
    def test_cache_manager_initialization(self, tmp_path):
        """Test cache manager initialization."""
        manager = CacheManager(cache_dir=tmp_path)
        
        assert manager.memory_cache is not None
        assert manager.file_cache is not None
    
    def test_cache_manager_memory_operations(self, tmp_path):
        """Test cache manager memory operations."""
        manager = CacheManager(cache_dir=tmp_path)
        
        # Test memory cache
        manager.memory_cache.put("mem_key", "mem_value")
        assert manager.memory_cache.get("mem_key") == "mem_value"
        
        # Should not be in file cache
        assert manager.file_cache.get("mem_key") is None
    
    def test_cache_manager_file_operations(self, tmp_path):
        """Test cache manager file operations."""
        manager = CacheManager(cache_dir=tmp_path)
        
        # Test file cache
        test_data = {"large": "data" * 1000}
        manager.file_cache.put("file_key", test_data)
        
        # Should be in file cache
        assert manager.file_cache.get("file_key") == test_data
    
    def test_cache_manager_file_result_caching(self, tmp_path):
         """Test cache manager file result caching."""
         manager = CacheManager(cache_dir=tmp_path)
         
         # Create test file
         test_file = tmp_path / "test.txt"
         test_file.write_text("test content")
         
         # Test file result caching
         call_count = 0
         
         @manager.cache_file_result(test_file, "read_operation")
         def expensive_function():
             nonlocal call_count
             call_count += 1
             return f"processed: {test_file.read_text()}"
         
         # First call
         result1 = expensive_function()
         assert "processed: test content" in result1
         assert call_count == 1
         
         # Second call with same file (should use cache)
         result2 = expensive_function()
         assert result2 == result1
         assert call_count == 1  # Should not increment
    
    def test_cache_clear_all(self, tmp_path):
        """Test clearing all caches."""
        manager = CacheManager(cache_dir=tmp_path)
        
        # Add data to both caches
        manager.memory_cache.put("mem_key", "mem_value")
        manager.file_cache.put("file_key", "file_value")
        
        # Verify data exists
        assert manager.memory_cache.get("mem_key") == "mem_value"
        assert manager.file_cache.get("file_key") == "file_value"
        
        # Clear all caches
        manager.clear_all_caches()
        
        # Verify data is cleared
        assert manager.memory_cache.get("mem_key") is None
        assert manager.file_cache.get("file_key") is None


class TestDocumentChunkerCaching:
    """Test caching integration in DocumentChunker."""
    
    def test_chunker_with_caching_enabled(self, tmp_path):
        """Test DocumentChunker with caching enabled."""
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            enable_caching=True,
            enable_security=False,
            enable_monitoring=False
        )
        
        chunker = DocumentChunker(config)
        
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Document\n\nThis is a test document for caching.")
        
        # First chunking (should cache result)
        result1 = chunker.chunk_file(test_file)
        assert not result1.cache_hit
        assert len(result1.chunks) > 0
        
        # Second chunking (should use cache)
        result2 = chunker.chunk_file(test_file)
        assert result2.cache_hit
        assert len(result2.chunks) == len(result1.chunks)
    
    def test_chunker_with_caching_disabled(self, tmp_path):
        """Test DocumentChunker with caching disabled."""
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            enable_caching=False,
            enable_security=False,
            enable_monitoring=False
        )
        
        chunker = DocumentChunker(config)
        
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Document\n\nThis is a test document.")
        
        # Both calls should not use cache
        result1 = chunker.chunk_file(test_file)
        result2 = chunker.chunk_file(test_file)
        
        assert not result1.cache_hit
        assert not result2.cache_hit
    
    def test_cache_invalidation_on_file_change(self, tmp_path):
        """Test that cache is invalidated when file changes."""
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            enable_caching=True,
            enable_security=False,
            enable_monitoring=False
        )
        
        chunker = DocumentChunker(config)
        
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Original Content\n\nOriginal text.")
        
        # First chunking
        result1 = chunker.chunk_file(test_file)
        assert not result1.cache_hit
        
        # Second chunking (should use cache)
        result2 = chunker.chunk_file(test_file)
        assert result2.cache_hit
        
        # Modify file
        time.sleep(0.1)  # Ensure different mtime
        test_file.write_text("# Modified Content\n\nModified text.")
        
        # Third chunking (should not use cache due to file change)
        result3 = chunker.chunk_file(test_file)
        assert not result3.cache_hit
    
    def test_cache_clear(self, tmp_path):
        """Test cache clearing functionality."""
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            enable_caching=True,
            enable_security=False,
            enable_monitoring=False
        )
        
        chunker = DocumentChunker(config)
        
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Document\n\nTest content.")
        
        # Chunk file to populate cache
        result1 = chunker.chunk_file(test_file)
        assert not result1.cache_hit
        
        # Verify cache hit
        result2 = chunker.chunk_file(test_file)
        assert result2.cache_hit
        
        # Clear cache
        chunker.clear_cache()
        
        # Should not use cache after clearing
        result3 = chunker.chunk_file(test_file)
        assert not result3.cache_hit


@pytest.fixture
def sample_cache_data():
    """Sample data for cache testing."""
    return {
        "chunks": [
            {"content": "First chunk", "metadata": {"index": 0}},
            {"content": "Second chunk", "metadata": {"index": 1}}
        ],
        "metadata": {
            "source_file": "test.md",
            "chunk_count": 2,
            "processing_time": 123.45
        }
    }


class TestCachePerformance:
    """Test cache performance characteristics."""
    
    def test_memory_cache_performance(self, sample_cache_data):
        """Test memory cache performance with large datasets."""
        cache = InMemoryCache(max_size=1000, default_ttl_seconds=3600)
        
        # Measure put performance
        start_time = time.time()
        for i in range(100):
            cache.put(f"key_{i}", sample_cache_data)
        set_time = time.time() - start_time
        
        # Measure get performance
        start_time = time.time()
        for i in range(100):
            cache.get(f"key_{i}")
        get_time = time.time() - start_time
        
        # Performance assertions (adjust thresholds as needed)
        assert set_time < 1.0  # Should complete in under 1 second
        assert get_time < 0.5  # Should complete in under 0.5 seconds
    
    def test_file_cache_performance(self, tmp_path, sample_cache_data):
        """Test file cache performance."""
        cache = FileCache(cache_dir=tmp_path, max_size_mb=100)
        
        # Measure put performance
        start_time = time.time()
        for i in range(10):  # Fewer iterations for file I/O
            cache.put(f"key_{i}", sample_cache_data)
        set_time = time.time() - start_time
        
        # Measure get performance
        start_time = time.time()
        for i in range(10):
            cache.get(f"key_{i}")
        get_time = time.time() - start_time
        
        # Performance assertions (more lenient for file I/O)
        assert set_time < 5.0  # Should complete in under 5 seconds
        assert get_time < 2.0  # Should complete in under 2 seconds