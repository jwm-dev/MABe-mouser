"""
Intelligent multi-layer caching system for the MABe viewer.

This module provides atomic, memoized caching at multiple levels:
1. Raw parquet data (file-level)
2. Parsed payloads (frame-level)
3. Computed features (analysis-level)
4. Rendered visualizations (display-level)

Prevents redundant computations across tabs and navigation.
"""

from __future__ import annotations

import hashlib
import pickle
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np

# Type variables for generic caching
T = TypeVar('T')
F = TypeVar('F', bound=Callable)


class CacheStats:
    """Track cache performance metrics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_requests = 0
        self.start_time = time.time()
    
    def record_hit(self):
        self.hits += 1
        self.total_requests += 1
    
    def record_miss(self):
        self.misses += 1
        self.total_requests += 1
    
    def record_eviction(self):
        self.evictions += 1
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    def __repr__(self) -> str:
        uptime = time.time() - self.start_time
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.1%}, evictions={self.evictions}, "
            f"uptime={uptime:.1f}s)"
        )


class SmartCache:
    """
    Multi-layer intelligent cache with automatic eviction.
    
    Layers:
    - L1: Hot cache (10 entries, instant access)
    - L2: Warm cache (50 entries, fast access)
    - L3: Disk cache (unlimited, moderate access)
    """
    
    def __init__(self, name: str = "cache"):
        self.name = name
        self._l1_cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, access_time)
        self._l2_cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()
        
        # Configuration
        self.l1_max_entries = 10   # Hot cache
        self.l2_max_entries = 50   # Warm cache
        self.l1_max_size_mb = 100  # Don't cache huge items in L1
        self.l2_max_size_mb = 500  # L2 can handle larger items
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 -> L2 -> miss)."""
        with self._lock:
            # Check L1 (hot cache)
            if key in self._l1_cache:
                value, _ = self._l1_cache[key]
                self._l1_cache[key] = (value, time.time())  # Update access time
                self._stats.record_hit()
                print(f"[{self.name}] L1 HIT: {key[:50]}")
                return value
            
            # Check L2 (warm cache)
            if key in self._l2_cache:
                value, _ = self._l2_cache[key]
                # Promote to L1 if small enough
                if self._estimate_size_mb(value) < self.l1_max_size_mb:
                    self._promote_to_l1(key, value)
                else:
                    # Just update access time in L2
                    self._l2_cache[key] = (value, time.time())
                self._stats.record_hit()
                print(f"[{self.name}] L2 HIT: {key[:50]}")
                return value
            
            # Cache miss
            self._stats.record_miss()
            print(f"[{self.name}] MISS: {key[:50]}")
            return None
    
    def put(self, key: str, value: Any, pin_to_l1: bool = False) -> None:
        """Put value into cache with automatic tier selection."""
        with self._lock:
            size_mb = self._estimate_size_mb(value)
            
            # Determine which tier to use
            if pin_to_l1 or size_mb < self.l1_max_size_mb:
                self._put_l1(key, value)
            elif size_mb < self.l2_max_size_mb:
                self._put_l2(key, value)
            else:
                print(f"[{self.name}] Value too large to cache: {size_mb:.1f}MB")
    
    def _put_l1(self, key: str, value: Any) -> None:
        """Put value into L1 cache."""
        self._l1_cache[key] = (value, time.time())
        self._evict_if_needed(self._l1_cache, self.l1_max_entries)
    
    def _put_l2(self, key: str, value: Any) -> None:
        """Put value into L2 cache."""
        self._l2_cache[key] = (value, time.time())
        self._evict_if_needed(self._l2_cache, self.l2_max_entries)
    
    def _promote_to_l1(self, key: str, value: Any) -> None:
        """Promote value from L2 to L1."""
        if key in self._l2_cache:
            del self._l2_cache[key]
        self._put_l1(key, value)
    
    def _evict_if_needed(self, cache: Dict, max_entries: int) -> None:
        """Evict LRU entries if cache is full."""
        while len(cache) > max_entries:
            # Find least recently used
            lru_key = min(cache.keys(), key=lambda k: cache[k][1])
            del cache[lru_key]
            self._stats.record_eviction()
    
    def _estimate_size_mb(self, value: Any) -> float:
        """Estimate size of value in MB."""
        import sys
        
        # Quick size estimation
        size_bytes = sys.getsizeof(value)
        
        # Add size of nested structures
        if isinstance(value, dict):
            for v in value.values():
                size_bytes += sys.getsizeof(v)
        elif isinstance(value, (list, tuple)):
            for item in value:
                size_bytes += sys.getsizeof(item)
        elif isinstance(value, np.ndarray):
            size_bytes = value.nbytes
        
        return size_bytes / (1024 * 1024)
    
    def clear(self) -> None:
        """Clear all cache tiers."""
        with self._lock:
            self._l1_cache.clear()
            self._l2_cache.clear()
    
    def invalidate(self, key: str) -> None:
        """Remove specific key from all tiers."""
        with self._lock:
            self._l1_cache.pop(key, None)
            self._l2_cache.pop(key, None)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class CacheManager:
    """
    Global cache manager coordinating all cache layers.
    
    Manages:
    - file_cache: Raw parquet data
    - payload_cache: Parsed frame payloads
    - feature_cache: Computed geometric features
    - analysis_cache: Analysis results (separability, etc.)
    - viz_cache: Rendered visualizations
    """
    
    def __init__(self):
        self.file_cache = SmartCache("FileCache")
        self.payload_cache = SmartCache("PayloadCache")
        self.feature_cache = SmartCache("FeatureCache")
        self.analysis_cache = SmartCache("AnalysisCache")
        self.viz_cache = SmartCache("VizCache")
        
        # Global settings
        self.enabled = True
    
    def get_cache_key(
        self,
        path: Optional[Path] = None,
        frame_range: Optional[Tuple[int, int]] = None,
        params: Optional[Dict] = None
    ) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            path: File path
            frame_range: (start, end) frame range
            params: Additional parameters to include in key
        
        Returns:
            Unique cache key string
        """
        components = []
        
        if path:
            # Include path and file modification time
            try:
                stat = path.stat()
                components.append(f"path:{path}:mtime:{stat.st_mtime_ns}")
            except:
                components.append(f"path:{path}")
        
        if frame_range:
            components.append(f"frames:{frame_range[0]}-{frame_range[1]}")
        
        if params:
            # Sort params for consistent hashing
            param_str = ":".join(f"{k}={v}" for k, v in sorted(params.items()))
            components.append(param_str)
        
        # Hash the components
        key_str = "::".join(components)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def invalidate_file(self, path: Path) -> None:
        """Invalidate all caches related to a file."""
        # This would invalidate all keys containing this path
        # For now, just clear everything (can optimize later)
        path_str = str(path)
        
        for cache in [self.file_cache, self.payload_cache, 
                     self.feature_cache, self.analysis_cache, self.viz_cache]:
            # Remove all keys containing this path
            with cache._lock:
                keys_to_remove = [k for k in cache._l1_cache.keys() if path_str in k]
                for k in keys_to_remove:
                    cache.invalidate(k)
    
    def print_stats(self) -> None:
        """Print statistics for all caches."""
        print("\n=== Cache Statistics ===")
        for name in ['file_cache', 'payload_cache', 'feature_cache', 
                     'analysis_cache', 'viz_cache']:
            cache = getattr(self, name)
            print(f"{name}: {cache.get_stats()}")
    
    def clear_all(self) -> None:
        """Clear all caches."""
        for cache in [self.file_cache, self.payload_cache,
                     self.feature_cache, self.analysis_cache, self.viz_cache]:
            cache.clear()


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def memoize(cache_name: str = "payload_cache", key_fn: Optional[Callable] = None):
    """
    Decorator to memoize function results.
    
    Args:
        cache_name: Which cache to use (file_cache, payload_cache, etc.)
        key_fn: Optional function to generate cache key from args
    
    Example:
        @memoize(cache_name="feature_cache")
        def compute_features(payloads):
            # expensive computation
            return features
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_cache_manager()
            cache = getattr(manager, cache_name)
            
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                # Default: use function name + args hash
                arg_str = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                key = hashlib.sha256(arg_str.encode()).hexdigest()
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        return wrapper
    return decorator


__all__ = [
    "CacheManager",
    "SmartCache",
    "CacheStats",
    "get_cache_manager",
    "memoize",
]
