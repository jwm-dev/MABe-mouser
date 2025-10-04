"""Priority-based thread pool executor for task scheduling with GPU support."""

from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from enum import IntEnum
from typing import Any, Callable, Optional

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_GPU = cp.cuda.is_available()
    GPU_DEVICE_COUNT = cp.cuda.runtime.getDeviceCount() if HAS_GPU else 0
except (ImportError, Exception):
    HAS_GPU = False
    GPU_DEVICE_COUNT = 0
    cp = None

# Get CPU count
CPU_COUNT = os.cpu_count() or 4


class TaskPriority(IntEnum):
    """Task priority levels (lower value = higher priority)."""
    CRITICAL = 0    # User-initiated file loads
    HIGH = 1        # Analysis computations for current file
    NORMAL = 2      # Cache operations
    LOW = 3         # Background operations (dataset stats, prefetch)
    BACKGROUND = 4  # Lowest priority (cleanup, etc.)


class PriorityTask:
    """Wrapper for a task with priority."""
    
    def __init__(self, priority: TaskPriority, fn: Callable, args: tuple, kwargs: dict):
        self.priority = priority
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.future: Future = Future()
        self.cancelled = False
    
    def __lt__(self, other: 'PriorityTask') -> bool:
        """Compare by priority for queue ordering."""
        return self.priority < other.priority
    
    def run(self) -> None:
        """Execute the task."""
        if self.cancelled or self.future.cancelled():
            return
        
        try:
            result = self.fn(*self.args, **self.kwargs)
            if not self.cancelled:
                self.future.set_result(result)
        except Exception as exc:
            if not self.cancelled:
                self.future.set_exception(exc)
    
    def cancel(self) -> bool:
        """Cancel the task."""
        self.cancelled = True
        return self.future.cancel()


class PriorityThreadPoolExecutor:
    """Thread pool executor with priority-based task scheduling."""
    
    def __init__(self, max_workers: int = 4, thread_name_prefix: str = ""):
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._workers: list[threading.Thread] = []
        self._shutdown = False
        self._lock = threading.Lock()
        
        # Start worker threads
        for i in range(max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"{thread_name_prefix}-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while not self._shutdown:
            try:
                # Get highest priority task (blocks with timeout)
                task = self._task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Check if we should shutdown
            if self._shutdown:
                break
            
            # Execute the task
            if task is not None:  # Defensive check
                task.run()
            self._task_queue.task_done()
    
    def submit(
        self,
        fn: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs
    ) -> Future:
        """Submit a task with priority."""
        if self._shutdown:
            raise RuntimeError("Executor is shutdown")
        
        task = PriorityTask(priority, fn, args, kwargs)
        self._task_queue.put(task)
        return task.future
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        
        # No need to put shutdown signals in priority queue
        # Workers will exit when they see _shutdown flag
        
        if wait:
            # Wait for queue to be empty
            self._task_queue.join()
            # Workers will exit on their own when they check _shutdown
            for worker in self._workers:
                worker.join(timeout=1.0)


class SmartExecutorManager:
    """Manages multiple executors with intelligent task routing and GPU support."""
    
    def __init__(self):
        # Calculate optimal thread counts based on CPU cores
        io_workers = min(CPU_COUNT * 2, 16)  # More for I/O-bound tasks
        cache_workers = min(CPU_COUNT, 8)     # Medium for cache ops
        analysis_workers = max(CPU_COUNT - 2, 2)  # Most cores for analysis
        compute_workers = max(CPU_COUNT // 2, 2)  # For CPU-intensive parallel tasks
        
        print(f"[ExecutorManager] Initializing with {CPU_COUNT} CPU cores")
        print(f"[ExecutorManager] IO workers: {io_workers}, Cache: {cache_workers}, Analysis: {analysis_workers}")
        if HAS_GPU:
            print(f"[ExecutorManager] GPU acceleration available: {GPU_DEVICE_COUNT} device(s)")
        
        # IO executor for file operations (priority-based)
        self.io_executor = PriorityThreadPoolExecutor(
            max_workers=io_workers,
            thread_name_prefix="smart-io"
        )
        
        # Cache executor for caching operations (priority-based)
        self.cache_executor = PriorityThreadPoolExecutor(
            max_workers=cache_workers,
            thread_name_prefix="smart-cache"
        )
        
        # Analysis executor for CPU-intensive computations
        self.analysis_executor = PriorityThreadPoolExecutor(
            max_workers=analysis_workers,
            thread_name_prefix="smart-analysis"
        )
        
        # Process pool for truly parallel CPU-bound work
        self.process_pool = ProcessPoolExecutor(
            max_workers=compute_workers
        )
        
        # GPU availability flag
        self.has_gpu = HAS_GPU
        self.gpu_device_count = GPU_DEVICE_COUNT
    
    def submit_load(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit a file load operation (CRITICAL priority)."""
        return self.io_executor.submit(fn, *args, priority=TaskPriority.CRITICAL, **kwargs)
    
    def submit_analysis(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit an analysis computation (HIGH priority)."""
        return self.analysis_executor.submit(fn, *args, priority=TaskPriority.HIGH, **kwargs)
    
    def submit_cache(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit a cache operation (NORMAL priority)."""
        return self.cache_executor.submit(fn, *args, priority=TaskPriority.NORMAL, **kwargs)
    
    def submit_prefetch(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit a prefetch operation (LOW priority)."""
        return self.io_executor.submit(fn, *args, priority=TaskPriority.LOW, **kwargs)
    
    def submit_background(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit a background operation (BACKGROUND priority)."""
        return self.cache_executor.submit(fn, *args, priority=TaskPriority.BACKGROUND, **kwargs)
    
    def submit_parallel_compute(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit a CPU-bound task to process pool for true parallelism."""
        return self.process_pool.submit(fn, *args, **kwargs)
    
    def get_numpy_backend(self):
        """Get numpy or cupy backend for array operations."""
        if self.has_gpu and cp is not None:
            return cp
        import numpy as np
        return np
    
    def to_numpy(self, arr):
        """Convert array to numpy (from cupy if needed)."""
        if self.has_gpu and cp is not None and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return arr
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown all executors."""
        self.io_executor.shutdown(wait=wait)
        self.cache_executor.shutdown(wait=wait)
        self.analysis_executor.shutdown(wait=wait)
        self.process_pool.shutdown(wait=wait)


__all__ = ["TaskPriority", "PriorityThreadPoolExecutor", "SmartExecutorManager", "HAS_GPU", "CPU_COUNT"]
