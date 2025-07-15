"""Caching utilities for the document chunking system.

This module provides intelligent caching mechanisms to improve performance
by avoiding redundant computations and file operations.
"""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import wraps
from src.utils.logger import get_logger
from src.exceptions import ChunkingError

T = TypeVar('T')


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        if self.metadata is None:
            self.metadata = {}
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def touch(self):
        """Update access information."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class InMemoryCache(Generic[T]):
    """High-performance in-memory cache with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 3600):
        """
        Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl_seconds: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.logger = get_logger(__name__)
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        if key not in self.cache:
            self._misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if entry.is_expired():
            del self.cache[key]
            self._misses += 1
            return None
        
        # Update access info
        entry.touch()
        self._hits += 1
        
        return entry.value
    
    def put(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> None:
        """Put value in cache."""
        ttl = ttl_seconds or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
        
        # Calculate size estimate
        try:
            size_bytes = len(pickle.dumps(value))
        except Exception:
            size_bytes = 0
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            expires_at=expires_at,
            size_bytes=size_bytes
        )
        
        self.cache[key] = entry
        
        # Evict if necessary
        self._evict_if_needed()
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache is full using LRU strategy."""
        while len(self.cache) > self.max_size:
            # Find least recently used entry
            lru_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].last_accessed
            )
            del self.cache[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        
        return {
            'entries': len(self.cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'total_size_bytes': total_size,
            'avg_size_bytes': total_size / len(self.cache) if self.cache else 0
        }


class FileCache:
    """Persistent file-based cache for larger objects."""
    
    def __init__(self, cache_dir: Union[str, Path], max_size_mb: int = 100):
        """
        Initialize file cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.logger = get_logger(__name__)
        self.index_file = self.cache_dir / "cache_index.json"
        self._load_index()
    
    def _load_index(self) -> None:
        """Load cache index from disk."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            else:
                self.index = {}
        except Exception as e:
            self.logger.warning(f"Failed to load cache index: {e}")
            self.index = {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        if key not in self.index:
            return None
        
        entry_info = self.index[key]
        cache_path = self._get_cache_path(key)
        
        # Check if file exists
        if not cache_path.exists():
            del self.index[key]
            self._save_index()
            return None
        
        # Check expiration
        if entry_info.get('expires_at'):
            expires_at = datetime.fromisoformat(entry_info['expires_at'])
            if datetime.now() > expires_at:
                self.delete(key)
                return None
        
        try:
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
            
            # Update access info
            entry_info['access_count'] = entry_info.get('access_count', 0) + 1
            entry_info['last_accessed'] = datetime.now().isoformat()
            self._save_index()
            
            return value
        except Exception as e:
            self.logger.error(f"Failed to load cached value: {e}")
            self.delete(key)
            return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Put value in file cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            file_size = cache_path.stat().st_size
            expires_at = None
            if ttl_seconds and ttl_seconds > 0:
                expires_at = (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat()
            
            self.index[key] = {
                'created_at': datetime.now().isoformat(),
                'expires_at': expires_at,
                'size_bytes': file_size,
                'access_count': 0,
                'last_accessed': datetime.now().isoformat()
            }
            
            self._save_index()
            self._cleanup_if_needed()
            
        except Exception as e:
            self.logger.error(f"Failed to cache value: {e}")
            if cache_path.exists():
                cache_path.unlink()
    
    def delete(self, key: str) -> bool:
        """Delete entry from file cache."""
        if key not in self.index:
            return False
        
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
        
        del self.index[key]
        self._save_index()
        return True
    
    def clear(self) -> None:
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
        self.index.clear()
        self._save_index()
    
    def _cleanup_if_needed(self) -> None:
        """Clean up cache if size limit exceeded."""
        total_size = sum(info['size_bytes'] for info in self.index.values())
        
        if total_size > self.max_size_bytes:
            # Sort by last accessed time and remove oldest
            sorted_keys = sorted(
                self.index.keys(),
                key=lambda k: self.index[k]['last_accessed']
            )
            
            for key in sorted_keys:
                if total_size <= self.max_size_bytes:
                    break
                
                total_size -= self.index[key]['size_bytes']
                self.delete(key)


class CacheManager:
    """Unified cache manager with multiple cache backends."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for file cache (optional)
        """
        self.memory_cache = InMemoryCache(max_size=500, default_ttl_seconds=1800)
        
        if cache_dir:
            self.file_cache = FileCache(cache_dir, max_size_mb=50)
        else:
            self.file_cache = None
        
        self.logger = get_logger(__name__)
    
    def get_file_hash(self, file_path: Union[str, Path]) -> str:
        """Generate hash for file content."""
        file_path = Path(file_path)
        
        # Include file size and modification time in hash
        stat = file_path.stat()
        content = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    def cache_file_result(self, file_path: Union[str, Path], operation: str, ttl_seconds: int = 3600):
        """Decorator for caching file-based operations."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                file_hash = self.get_file_hash(file_path)
                cache_key = f"{operation}:{file_hash}:{hash(str(args) + str(sorted(kwargs.items())))}"
                
                # Try memory cache first
                result = self.memory_cache.get(cache_key)
                if result is not None:
                    self.logger.debug(f"Memory cache hit for {operation}")
                    return result
                
                # Try file cache if available
                if self.file_cache:
                    result = self.file_cache.get(cache_key)
                    if result is not None:
                        self.logger.debug(f"File cache hit for {operation}")
                        # Store in memory cache for faster access
                        self.memory_cache.put(cache_key, result, ttl_seconds=min(ttl_seconds, 1800))
                        return result
                
                # Cache miss - execute function
                self.logger.debug(f"Cache miss for {operation}, executing function")
                result = func(*args, **kwargs)
                
                # Store in caches
                self.memory_cache.put(cache_key, result, ttl_seconds=min(ttl_seconds, 1800))
                if self.file_cache:
                    self.file_cache.put(cache_key, result, ttl_seconds=ttl_seconds)
                
                return result
            return wrapper
        return decorator
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'memory_cache': self.memory_cache.get_stats()
        }
        
        if self.file_cache:
            file_stats = {
                'entries': len(self.file_cache.index),
                'total_size_bytes': sum(info['size_bytes'] for info in self.file_cache.index.values()),
                'max_size_bytes': self.file_cache.max_size_bytes
            }
            stats['file_cache'] = file_stats
        
        return stats
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        if self.file_cache:
            self.file_cache.clear()
        self.logger.info("All caches cleared")


# Global cache manager instance
default_cache_manager = CacheManager()


def cached_operation(operation: str, ttl_seconds: int = 3600):
    """Simple decorator for caching operation results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{operation}:{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try cache first
            result = default_cache_manager.memory_cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            default_cache_manager.memory_cache.put(cache_key, result, ttl_seconds=ttl_seconds)
            
            return result
        return wrapper
    return decorator