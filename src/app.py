"""Core Qt + Vispy application logic for the pose viewer."""

from __future__ import annotations

import os
import sys
import threading
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
        parquet_files: Optional[Sequence[Path]] = None,
        *,
        discovery_root: Optional[Path] = None,
        max_files: Optional[int] = None,
        use_gpu: bool = False,
    ) -> None:
        self._startup_debug_log_enabled = bool(os.environ.get("POSE_VIEWER_DEBUG_STARTUP"))
        self._startup_start_time = time.perf_counter()
        if self._startup_debug_log_enabled:
            print("[startup-debug] PoseViewerApp.__init__ begin")
        initial_files: List[Path] = [Path(p) for p in parquet_files] if parquet_files else []
        discovery_root_path: Optional[Path] = None
        if discovery_root is not None:
            discovery_root_path = Path(discovery_root)
            if not discovery_root_path.exists():
                raise ValueError(f"Cannot find parquet source: {discovery_root_path}")
        if not initial_files and discovery_root_path is None:
            raise ValueError("No parquet files provided and discovery root not supplied.")

        self.root = root
        self.root.setWindowTitle("MABe Mouser")

        self.parquet_files: List[Path] = initial_files
        self.current_file_index: int = 0
        self.current_data: Optional[Dict[str, object]] = None
        self.mouse_colors: Dict[str, Tuple[float, float, float]] = {}
        self.use_gpu = bool(use_gpu and cudf is not None)
        self._file_history: List[Path] = []
        self._history_position: int = -1
        self._history_navigation_active: bool = False

        self._discovery_root: Optional[Path] = discovery_root_path
        self._discovery_max_files: Optional[int] = max_files
        self._discovery_thread: Optional[threading.Thread] = None
        self._waiting_for_discovery: bool = False
        self._pending_initial_target: Optional[Path] = None
        self._cache_flush_pending = bool(hasattr(self, "_flush_cache_on_startup"))

        self._log_startup("initial directory seed")
        self._set_initial_active_directory()

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

        self._log_startup("cache directories prepared")

        if hasattr(self, "_load_persisted_history"):
            self._load_persisted_history()
        self._log_startup("history loaded")

        self._build_ui()
        self._log_startup("UI built")
        self._init_playback_state()
        self._log_startup("playback state initialised")
        self._register_bindings()
        self._log_startup("bindings registered")
        self._connect_scene_hover(self._handle_hover_payload)
        self._log_startup("hover connected")
        self._post_ui_initialisation()
        self._log_startup("post-ui initialisation complete")
        self._log_startup("PoseViewerApp.__init__ complete")

    def _log_startup(self, label: str) -> None:
        if getattr(self, "_startup_debug_log_enabled", False):
            elapsed = time.perf_counter() - getattr(self, "_startup_start_time", time.perf_counter())
            print(f"[startup-debug] {label}: {elapsed:.3f}s")

    def _post_ui_initialisation(self) -> None:
        if not self.parquet_files:
            apply_target = getattr(self, "_apply_pending_initial_target", None)
            if callable(apply_target):
                try:
                    applied = bool(apply_target())
                except Exception:
                    applied = False
                else:
                    if applied:
                        self._log_startup("pending history applied")
        if self.parquet_files:
            if hasattr(self, "_schedule_initial_load"):
                self._schedule_initial_load()
            else:
                self._load_current_file()
        else:
            self._enter_discovery_wait_state("Scanning for parquet files…")
        if self._discovery_root is not None:
            self._begin_parquet_discovery()
        elif not self.parquet_files:
            self._set_status("No parquet files available")
        self._start_background_cache_flush()

    def _enter_discovery_wait_state(self, message: str) -> None:
        self._waiting_for_discovery = True
        try:
            self.progressbar.setRange(0, 0)
        except Exception:
            pass
        try:
            self.progress_var.set(0.0)
        except Exception:
            pass
        self._set_status(message)

    def _exit_discovery_wait_state(self) -> None:
        self._waiting_for_discovery = False
        try:
            self.progressbar.setRange(0, 100)
        except Exception:
            pass

    def _begin_parquet_discovery(self) -> None:
        if self._discovery_root is None:
            return
        if self._discovery_thread is not None:
            return
        def _worker() -> None:
            try:
                files = self._scan_parquet_files(self._discovery_root, self._discovery_max_files)
            except Exception as exc:
                QtCore.QTimer.singleShot(0, lambda exc=exc: self._on_parquet_discovery_error(exc))
                return
            QtCore.QTimer.singleShot(0, lambda files=files: self._on_parquet_discovery_complete(files))
        self._discovery_thread = threading.Thread(target=_worker, name="pose-parquet-discovery", daemon=True)
        self._discovery_thread.start()

    @staticmethod
    def _scan_parquet_files(root: Path, max_files: Optional[int]) -> List[Path]:
        root_path = Path(root)
        if root_path.is_file():
            return [root_path] if root_path.suffix.lower() == ".parquet" else []
        if not root_path.exists():
            return []
        files = sorted(p for p in root_path.rglob("*.parquet") if p.is_file())
        if max_files is not None:
            files = files[: max(0, int(max_files))]
        return files

    def _on_parquet_discovery_error(self, exc: Exception) -> None:
        self._discovery_thread = None
        if getattr(self, "_waiting_for_discovery", False):
            self._exit_discovery_wait_state()
        self._set_status(f"Failed to scan parquet files: {exc}")
        if hasattr(self, "_show_error"):
            self._show_error("Error scanning parquet files", str(exc))

    def _on_parquet_discovery_complete(self, files: Sequence[Path]) -> None:
        self._discovery_thread = None
        if getattr(self, "_waiting_for_discovery", False):
            self._exit_discovery_wait_state()
        paths = [Path(p) for p in files]
        if not paths:
            root_desc = str(self._discovery_root) if self._discovery_root is not None else "dataset"
            message = f"No parquet files found under {root_desc}"
            self._set_status(message)
            if hasattr(self, "_show_warning"):
                self._show_warning("No parquet files found", message)
            return
        previous_path = None
        if self.parquet_files and 0 <= self.current_file_index < len(self.parquet_files):
            previous_path = self.parquet_files[self.current_file_index]
        self.parquet_files = paths
        self._set_initial_active_directory()
        if hasattr(self, "_update_file_menu"):
            self._update_file_menu()
        if previous_path is not None:
            try:
                self.current_file_index = self.parquet_files.index(previous_path)
            except ValueError:
                self.current_file_index = max(0, min(self.current_file_index, len(self.parquet_files) - 1))
        else:
            self.current_file_index = 0
        if not getattr(self, "_initial_load_scheduled", False):
            if hasattr(self, "_schedule_initial_load"):
                self._schedule_initial_load()
            else:
                self._load_current_file()
        self._start_background_cache_flush()

    def _start_background_cache_flush(self) -> None:
        if not getattr(self, "_cache_flush_pending", False):
            return
        self._cache_flush_pending = False
        flush_func = getattr(self, "_flush_cache_on_startup", None)
        if not callable(flush_func):
            return

        def _worker() -> None:
            start = time.perf_counter()
            try:
                flush_func()
            except Exception as exc:
                print(f"[PoseViewer] cache flush failed: {exc!r}")
            else:
                if getattr(self, "_startup_debug_log_enabled", False):
                    elapsed = time.perf_counter() - start
                    print(f"[startup-debug] cache flushed (async): {elapsed:.3f}s")

        threading.Thread(target=_worker, name="pose-cache-flush", daemon=True).start()

    def _set_initial_active_directory(self) -> None:
        if not self.parquet_files:
            return
        parents = {path.parent for path in self.parquet_files}
        if len(parents) <= 1:
            return
        initial_index = min(max(self.current_file_index, 0), len(self.parquet_files) - 1)
        initial_path = self.parquet_files[initial_index]
        folder = initial_path.parent
        if not folder.exists():
            return
        files = sorted(p for p in folder.glob("*.parquet") if p.is_file())
        if not files:
            return
        self.parquet_files = files
        try:
            self.current_file_index = files.index(initial_path)
        except ValueError:
            self.current_file_index = 0

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
