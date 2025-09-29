"""Optional third-party dependencies shared across the pose viewer package."""

from __future__ import annotations

try:  # pragma: no cover - optional export dependency
    import imageio  # type: ignore[import]
except Exception:  # pragma: no cover - imageio unavailable
    imageio = None

try:  # pragma: no cover - runtime dependency check
    import pyarrow.parquet as pq  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - pyarrow not available
    try:  # pragma: no cover - fastparquet fallback
        import fastparquet as pq  # type: ignore[import]
    except Exception:  # pragma: no cover - fastparquet unavailable
        pq = None

try:  # pragma: no cover - optional GPU acceleration
    import cudf  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - cudf unavailable
    cudf = None

try:  # pragma: no cover - optional Qt icon helper
    import qtawesome  # type: ignore[import]
except Exception:  # pragma: no cover - qtawesome unavailable
    qtawesome = None


__all__ = ["imageio", "pq", "cudf", "qtawesome"]
