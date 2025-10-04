"""Smart prefetching system for predictive file loading."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from .priority_executor import TaskPriority


class PrefetchManager:
    """Manages intelligent prefetching of adjacent files."""
    
    def __init__(
        self,
        load_fn: Callable[[Path], Dict[str, Any]],
        cache_fn: Callable[[Path, Dict[str, Any]], None],
        executor_submit_fn: Callable,
    ):
        """
        Initialize prefetch manager.
        
        Args:
            load_fn: Function to load a file (returns data dict)
            cache_fn: Function to cache loaded data
            executor_submit_fn: Function to submit tasks (should support priority)
        """
        self.load_fn = load_fn
        self.cache_fn = cache_fn
        self.executor_submit = executor_submit_fn
        
        self._prefetch_futures: Dict[Path, Any] = {}
        self._prefetch_lock = threading.Lock()
        self._current_file: Optional[Path] = None
        self._file_list: List[Path] = []
        self._prefetch_distance = 2  # Prefetch 2 files ahead/behind
        self._enabled = True
    
    def set_file_list(self, files: List[Path]) -> None:
        """Update the list of files for navigation prediction."""
        with self._prefetch_lock:
            self._file_list = files
    
    def set_current_file(self, path: Path) -> None:
        """
        Update current file and trigger prefetching of adjacent files.
        
        Args:
            path: Current file being viewed
        """
        if not self._enabled:
            return
        
        with self._prefetch_lock:
            self._current_file = path
            
            # Cancel any prefetches that are now too far away
            self._cancel_distant_prefetches(path)
            
            # Trigger prefetching for adjacent files
            self._prefetch_adjacent(path)
    
    def _cancel_distant_prefetches(self, current_path: Path) -> None:
        """Cancel prefetch tasks for files too far from current position."""
        if not self._file_list or current_path not in self._file_list:
            return
        
        current_idx = self._file_list.index(current_path)
        
        # Cancel futures for files outside prefetch distance
        for prefetch_path, future in list(self._prefetch_futures.items()):
            if prefetch_path not in self._file_list:
                future.cancel()
                del self._prefetch_futures[prefetch_path]
                continue
            
            prefetch_idx = self._file_list.index(prefetch_path)
            distance = abs(prefetch_idx - current_idx)
            
            if distance > self._prefetch_distance:
                future.cancel()
                del self._prefetch_futures[prefetch_path]
    
    def _prefetch_adjacent(self, current_path: Path) -> None:
        """Prefetch files adjacent to current position."""
        if not self._file_list or current_path not in self._file_list:
            return
        
        current_idx = self._file_list.index(current_path)
        
        # Determine files to prefetch
        to_prefetch = []
        
        # Prefetch forward (higher priority - users typically go forward)
        for i in range(1, self._prefetch_distance + 1):
            next_idx = current_idx + i
            if next_idx < len(self._file_list):
                to_prefetch.append((self._file_list[next_idx], i))
        
        # Prefetch backward (lower priority)
        for i in range(1, self._prefetch_distance + 1):
            prev_idx = current_idx - i
            if prev_idx >= 0:
                to_prefetch.append((self._file_list[prev_idx], i + self._prefetch_distance))
        
        # Submit prefetch tasks
        for path, distance in to_prefetch:
            if path not in self._prefetch_futures:
                self._submit_prefetch(path, distance)
    
    def _submit_prefetch(self, path: Path, distance: int) -> None:
        """Submit a prefetch task for a file."""
        def prefetch_worker():
            """Worker function to load and cache file."""
            try:
                # Load the file
                data = self.load_fn(path)
                
                # Cache the result
                if data is not None:
                    self.cache_fn(path, data)
                
                return data
            except Exception as e:
                print(f"[Prefetch] Failed to prefetch {path.name}: {e}")
                return None
            finally:
                # Remove from tracking when complete
                with self._prefetch_lock:
                    self._prefetch_futures.pop(path, None)
        
        # Submit with LOW priority (don't interfere with user operations)
        future = self.executor_submit(prefetch_worker, priority=TaskPriority.LOW)
        self._prefetch_futures[path] = future
    
    def cancel_all(self) -> None:
        """Cancel all pending prefetch operations."""
        with self._prefetch_lock:
            for future in self._prefetch_futures.values():
                future.cancel()
            self._prefetch_futures.clear()
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable prefetching."""
        self._enabled = enabled
        if not enabled:
            self.cancel_all()
    
    def set_prefetch_distance(self, distance: int) -> None:
        """Set how many files ahead/behind to prefetch."""
        self._prefetch_distance = max(1, min(distance, 5))  # Limit to 1-5


__all__ = ["PrefetchManager"]
