"""Core Qt + Vispy application logic for the pose viewer."""

from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import time

import numpy as np
from PyQt6 import QtCore

from .cache_utils import PoseViewerCacheMixin
from .exporter import PoseViewerExportMixin
from .file_loader import PoseViewerFileMixin
from .hover import PoseViewerHoverMixin
from .optional_dependencies import cudf, imageio, pq
from .playback import PoseViewerPlaybackMixin
from .plotting import initialise_viewer_theme
from .status import PoseViewerStatusMixin
from .ui import PoseViewerUIMixin


class PoseViewerApp(
    PoseViewerStatusMixin,
    PoseViewerExportMixin,
    PoseViewerPlaybackMixin,
    PoseViewerFileMixin,
    PoseViewerCacheMixin,
    PoseViewerHoverMixin,
    PoseViewerUIMixin,
):
    """Qt + Vispy app for browsing parquet tracking files."""

    def __init__(
        self,
        root: Any,
        parquet_files: Sequence[Path],
        trail_length: int = 14,
        use_gpu: bool = False,
    ) -> None:
        if not parquet_files:
            raise ValueError("No parquet files provided.")

        self.root = root
        self.root.setWindowTitle("MABe Mouser")

        self.parquet_files: List[Path] = list(parquet_files)
        self.current_file_index: int = 0
        self.current_data: Optional[Dict[str, object]] = None
        self.trail_length = max(3, trail_length)
        self.trail_visual_length = max(5, min(10, self.trail_length))
        self.trail_cache: Dict[str, List[np.ndarray]] = {}
        self.mouse_colors: Dict[str, Tuple[float, float, float]] = {}
        self.use_gpu = bool(use_gpu and cudf is not None)
        self.tail_histories: Dict[str, deque[np.ndarray]] = {}

        self.playing: bool = False
        self.playback_speed_multiplier: float = 1.0
        self.playback_base_fps: float = 30.0
        self.animation_timer = QtCore.QTimer(self.root)
        self.animation_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.animation_timer.timeout.connect(self._animation_step)
        self._playback_last_time = time.perf_counter()
        self._playback_frame_position = 0.0
        self._playback_time = 0.0
        self._frame_times = np.zeros(0, dtype=np.float64)
        self._frame_total_duration = 0.0

        self.data_cache: Dict[Path, Dict[str, object]] = {}
        self.loading_thread = None

        self.slider_active: bool = False
        self._initialise_ui_variables()
        self._initialise_status_dispatcher()

        self.cache_enabled = True
        package_dir = Path(__file__).resolve().parent
        self.cache_dir = self._resolve_cache_directory()
        for candidate in (self.cache_dir, package_dir / "__pycache__"):
            try:
                candidate.mkdir(parents=True, exist_ok=True)
            except Exception:
                continue
            else:
                self.cache_dir = candidate
                break
        else:
            self.cache_enabled = False
            self.cache_dir = package_dir

        initialise_viewer_theme()

        self._build_ui()
        self._init_playback_state()
        self._register_bindings()
        self._connect_scene_hover(self._handle_hover_payload)
        self._load_current_file()

    def _resolve_cache_directory(self) -> Path:
        env_override = os.environ.get("MABE_POSE_CACHE_DIR")
        if env_override:
            return Path(env_override)

        home = Path.home()
        if sys.platform.startswith("win"):
            base = os.environ.get("LOCALAPPDATA")
            base_path = Path(base) if base else home / "AppData" / "Local"
            return base_path / "MABe" / "PoseCache"
        if sys.platform == "darwin":
            return home / "Library" / "Caches" / "mabe-pose-viewer"

        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            return Path(xdg_cache) / "mabe-pose-viewer"
        return home / ".cache" / "mabe-pose-viewer"


__all__ = ["PoseViewerApp", "cudf", "imageio", "pq"]
