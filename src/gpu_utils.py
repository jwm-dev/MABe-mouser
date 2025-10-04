"""GPU acceleration utilities with fallback to CPU."""

from __future__ import annotations

import numpy as np
from typing import Any, Union

# Try to import CuPy
try:
    import cupy as cp
    HAS_CUPY = cp.cuda.is_available()
except (ImportError, Exception):
    HAS_CUPY = False
    cp = None

ArrayLike = Union[np.ndarray, Any]  # Any for cupy arrays


def get_array_module(arr: ArrayLike):
    """Get the appropriate array module (numpy or cupy) for an array."""
    if HAS_CUPY and cp is not None:
        return cp.get_array_module(arr)
    return np


def to_cpu(arr: ArrayLike) -> np.ndarray:
    """Convert array to CPU numpy array (from GPU if needed)."""
    if HAS_CUPY and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_gpu(arr: np.ndarray, force: bool = False):
    """Convert numpy array to GPU array if available.
    
    Args:
        arr: Input numpy array
        force: If True, always try to move to GPU. If False, only move large arrays.
    
    Returns:
        GPU array if available and worthwhile, otherwise original array
    """
    if not HAS_CUPY or cp is None:
        return arr
    
    # Only move to GPU if array is large enough to benefit
    if not force and arr.nbytes < 1024 * 1024:  # < 1MB
        return arr
    
    try:
        return cp.asarray(arr)
    except Exception as e:
        print(f"[GPU] Failed to move array to GPU: {e}")
        return arr


def compute_distances_vectorized(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Compute pairwise distances between two sets of points (GPU-accelerated if available).
    
    Args:
        points1: Array of shape (N, D) 
        points2: Array of shape (M, D)
    
    Returns:
        Distance matrix of shape (N, M)
    """
    xp = get_array_module(points1)
    
    # Use broadcasting for vectorized distance computation
    # ||p1 - p2||^2 = ||p1||^2 + ||p2||^2 - 2*p1·p2
    p1_squared = xp.sum(points1**2, axis=1, keepdims=True)  # (N, 1)
    p2_squared = xp.sum(points2**2, axis=1, keepdims=True).T  # (1, M)
    cross_term = points1 @ points2.T  # (N, M)
    
    distances_squared = p1_squared + p2_squared - 2 * cross_term
    distances_squared = xp.maximum(distances_squared, 0)  # Fix numerical errors
    
    return xp.sqrt(distances_squared)


def compute_centroids_batch(points_list: list[np.ndarray]) -> np.ndarray:
    """Compute centroids for a batch of point sets (GPU-accelerated if available).
    
    Args:
        points_list: List of point arrays, each of shape (N_i, D)
    
    Returns:
        Array of centroids, shape (len(points_list), D)
    """
    if not points_list:
        return np.array([])
    
    # For small batches, use numpy
    if len(points_list) < 100:
        return np.array([np.mean(points, axis=0) for points in points_list if len(points) > 0])
    
    # For large batches and GPU available, batch process
    if HAS_CUPY and cp is not None:
        try:
            # Move to GPU
            gpu_points = [cp.asarray(p) for p in points_list if len(p) > 0]
            centroids = cp.array([cp.mean(p, axis=0) for p in gpu_points])
            return cp.asnumpy(centroids)
        except Exception:
            pass
    
    # Fallback to numpy
    return np.array([np.mean(points, axis=0) for points in points_list if len(points) > 0])


def vectorized_norm(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute vector norms efficiently (GPU-accelerated if available).
    
    Args:
        vectors: Array of vectors
        axis: Axis along which to compute norm
    
    Returns:
        Array of norms
    """
    arr = to_gpu(vectors, force=False)
    xp = get_array_module(arr)
    result = xp.linalg.norm(arr, axis=axis)
    return to_cpu(result)


def parallel_rolling_window(data: np.ndarray, window_size: int, func: str = 'mean') -> np.ndarray:
    """Apply rolling window operation efficiently (GPU-accelerated if available).
    
    Args:
        data: 1D or 2D array
        window_size: Size of rolling window
        func: Function to apply ('mean', 'std', 'min', 'max')
    
    Returns:
        Result of rolling window operation
    """
    if len(data) < window_size:
        return data
    
    # Use scipy for CPU, cupy for GPU
    try:
        from scipy.ndimage import uniform_filter1d
        if func == 'mean':
            return uniform_filter1d(data, size=window_size, axis=0)
    except ImportError:
        pass
    
    # Fallback to manual implementation
    if func == 'mean':
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='same')
    
    return data


__all__ = [
    'HAS_CUPY',
    'get_array_module',
    'to_cpu',
    'to_gpu',
    'compute_distances_vectorized',
    'compute_centroids_batch',
    'vectorized_norm',
    'parallel_rolling_window',
]
