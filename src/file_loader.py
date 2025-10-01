"""Data loading and preprocessing mixin for the pose viewer application."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from PyQt6 import QtCore
import numpy as np
import pandas as pd

from .geometry import PoseViewerGeometryMixin
from .models import FramePayload, MouseGroup
from .optional_dependencies import cudf, pq


class PoseViewerFileMixin(PoseViewerGeometryMixin):
    _history_store_filename = "pose_viewer_history.json"
    _history_store_version = 1
    _history_max_entries = 40
    def _prompt_for_folder(self) -> None:
        directory = self._ask_directory(title="Select directory with parquet files")
        if not directory:
            return
        folder = Path(directory)
        files = sorted(folder.rglob("*.parquet"))
        if not files:
            self._show_warning("No parquet files", f"No parquet files found in {folder}")
            return
        self.parquet_files = files
        self.current_file_index = 0
        self.data_cache.clear()
        self._update_file_menu()
        self._load_current_file()

    def _go_to_previous_file(self) -> None:
        self._force_pause_playback()
        if self._navigate_history(-1):
            return
        if not self.parquet_files:
            return
        self.current_file_index = (self.current_file_index - 1) % len(self.parquet_files)
        self._load_current_file()

    def _go_to_next_file(self) -> None:
        self._force_pause_playback()
        if self._navigate_history(1):
            return
        if not self.parquet_files:
            return
        self.current_file_index = (self.current_file_index + 1) % len(self.parquet_files)
        self._load_current_file()

    def _ensure_history_state(self) -> None:
        if not isinstance(getattr(self, "_file_history", None), list):
            self._file_history = []
        if not isinstance(getattr(self, "_history_position", None), int):
            self._history_position = -1
        if not isinstance(getattr(self, "_history_navigation_active", None), bool):
            self._history_navigation_active = False

    def _history_store_path(self) -> Path:
        cache_dir = getattr(self, "cache_dir", None)
        try:
            base = Path(cache_dir) if cache_dir is not None else Path.cwd()
        except Exception:
            base = Path.cwd()
        return base / self._history_store_filename

    def _load_persisted_history(self) -> None:
        self._ensure_history_state()
        store_path = self._history_store_path()
        try:
            with store_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except FileNotFoundError:
            return
        except Exception:
            return
        recent = payload.get("recent_files") or payload.get("history")
        if not isinstance(recent, list):
            return
        max_entries = int(getattr(self, "_history_max_entries", 40))
        cleaned: List[Path] = []
        for raw in recent[-max_entries:]:
            if not isinstance(raw, str):
                continue
            candidate = Path(raw)
            if candidate.exists() and candidate.is_file():
                cleaned.append(candidate)
        if not cleaned:
            return
        self._file_history = list(cleaned)
        last_index = payload.get("last_index")
        if isinstance(last_index, int):
            index = max(0, min(last_index, len(cleaned) - 1))
        else:
            index = len(cleaned) - 1
        self._history_position = index
        self._history_navigation_active = False
        target = cleaned[index]
        setattr(self, "_pending_initial_target", target)

    def _persist_file_history(self) -> None:
        self._ensure_history_state()
        try:
            store_path = self._history_store_path()
        except Exception:
            return
        history: List[Path] = self._file_history  # type: ignore[assignment]
        max_entries = int(getattr(self, "_history_max_entries", 40))
        stored = history[-max_entries:]
        base_index = len(history) - len(stored)
        entries = [str(path) for path in stored]
        last_index = -1
        if stored:
            relative_index = self._history_position - base_index
            last_index = max(0, min(relative_index, len(stored) - 1))
        payload = {
            "version": self._history_store_version,
            "recent_files": entries,
            "last_index": last_index,
        }
        try:
            store_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            with store_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception:
            pass

    def _trim_file_history(self) -> None:
        self._ensure_history_state()
        history: List[Path] = self._file_history  # type: ignore[assignment]
        max_entries = int(getattr(self, "_history_max_entries", 40))
        if len(history) <= max_entries:
            return
        excess = len(history) - max_entries
        if excess <= 0:
            return
        del history[:excess]
        self._history_position = max(0, self._history_position - excess)

    def _set_initial_file_from_path(self, target: Path) -> bool:
        folder = target.parent
        if not folder.exists():
            return False
        files = sorted(p for p in folder.glob("*.parquet") if p.is_file())
        if not files:
            return False
        try:
            resolved_target = target.resolve()
        except FileNotFoundError:
            return False
        match_index: Optional[int] = None
        for idx, candidate in enumerate(files):
            try:
                if candidate.resolve() == resolved_target:
                    match_index = idx
                    break
            except FileNotFoundError:
                continue
        if match_index is None:
            return False
        self.parquet_files = files
        self.current_file_index = match_index
        return True

    def _apply_pending_initial_target(self) -> bool:
        target = getattr(self, "_pending_initial_target", None)
        if not target:
            return False
        if not isinstance(target, Path):
            try:
                target = Path(str(target))
            except Exception:
                self._pending_initial_target = None
                return False
        success = self._set_initial_file_from_path(target)
        if success:
            self._pending_initial_target = None
        return success

    @staticmethod
    def _coerce_positive_float(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(num) or num <= 0.0:
            return None
        return num

    @staticmethod
    def _metadata_first_numeric(source: Mapping[str, Any], keys: Iterable[str]) -> Optional[float]:
        if not isinstance(source, Mapping):
            return None
        for key in keys:
            if key in source:
                num = PoseViewerFileMixin._coerce_positive_float(source[key])
                if num is not None:
                    return num
        return None

    @classmethod
    def _metadata_size_px(
        cls,
        metadata: Mapping[str, Any],
        *,
        width_keys: Sequence[str],
        height_keys: Sequence[str],
        section_keys: Sequence[str] = (),
    ) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(metadata, Mapping):
            return (None, None)
        width = cls._metadata_first_numeric(metadata, width_keys)
        height = cls._metadata_first_numeric(metadata, height_keys)
        for section_key in section_keys:
            section = metadata.get(section_key)
            if not isinstance(section, Mapping):
                continue
            width = width or cls._metadata_first_numeric(section, ("width_px", "width", "w"))
            height = height or cls._metadata_first_numeric(section, ("height_px", "height", "h"))
            if width is not None and height is not None:
                break
        return (width, height)

    @classmethod
    def _metadata_video_size_px(cls, metadata: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        return cls._metadata_size_px(
            metadata,
            width_keys=(
                "video_width_px",
                "video_width_pix",
                "video_width_pixels",
                "frame_width_px",
            ),
            height_keys=(
                "video_height_px",
                "video_height_pix",
                "video_height_pixels",
                "frame_height_px",
            ),
            section_keys=("video",),
        )

    @classmethod
    def _metadata_arena_size_px(cls, metadata: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        return cls._metadata_size_px(
            metadata,
            width_keys=(
                "arena_width_px",
                "arena_width_pix",
                "arena_pixels_width",
            ),
            height_keys=(
                "arena_height_px",
                "arena_height_pix",
                "arena_pixels_height",
            ),
            section_keys=("arena",),
        )

    @classmethod
    def _augment_tracking_metadata(cls, metadata: Dict[str, Any]) -> None:
        if not isinstance(metadata, dict):
            return
        video_w, video_h = cls._metadata_video_size_px(metadata)
        arena_w, arena_h = cls._metadata_arena_size_px(metadata)
        if video_w is not None:
            metadata.setdefault("video_width_px", float(video_w))
        if video_h is not None:
            metadata.setdefault("video_height_px", float(video_h))
        if arena_w is not None:
            metadata.setdefault("arena_width_px", float(arena_w))
        if arena_h is not None:
            metadata.setdefault("arena_height_px", float(arena_h))
        if video_w is not None and video_h is not None:
            video_section: Dict[str, Any]
            video_raw = metadata.get("video")
            if isinstance(video_raw, Mapping):
                video_section = dict(video_raw)
            else:
                video_section = {}
            video_section.setdefault("width_px", float(video_w))
            video_section.setdefault("height_px", float(video_h))
            video_section.setdefault("width", float(video_w))
            video_section.setdefault("height", float(video_h))
            video_section.setdefault("units", "pixels")
            metadata["video"] = video_section
        if arena_w is not None and arena_h is not None:
            arena_section: Dict[str, Any]
            arena_raw = metadata.get("arena")
            if isinstance(arena_raw, Mapping):
                arena_section = dict(arena_raw)
            else:
                arena_section = {}
            arena_section.setdefault("width_px", float(arena_w))
            arena_section.setdefault("height_px", float(arena_h))
            arena_section.setdefault("width", float(arena_w))
            arena_section.setdefault("height", float(arena_h))
            arena_section.setdefault("units", "pixels")
            metadata["arena"] = arena_section
        pixels_per_cm = cls._coerce_positive_float(metadata.get("pixels_per_cm") or metadata.get("pix_per_cm_approx"))
        cm_per_pixel = cls._coerce_positive_float(metadata.get("cm_per_pixel"))
        if cm_per_pixel is None and pixels_per_cm:
            cm_per_pixel = 1.0 / pixels_per_cm
        if cm_per_pixel and "video" in metadata and isinstance(metadata["video"], dict):
            metadata["video"].setdefault("cm_per_pixel", float(cm_per_pixel))

    def _navigate_history(self, step: int) -> bool:
        self._ensure_history_state()
        history: List[Path] = self._file_history  # type: ignore[assignment]
        if not history:
            return False
        current_index = self._history_position
        if current_index < 0 or current_index >= len(history):
            current_index = len(history) - 1
            target_index = current_index + step
        else:
            target_index = current_index + step
        if target_index < 0 or target_index >= len(history):
            return False
        target_path = history[target_index]
        return self._open_file_from_history(target_path, target_index)

    def _open_file_from_history(self, target: Path, target_index: int) -> bool:
        self._ensure_history_state()
        if not self._set_initial_file_from_path(target):
            return False
        self._history_navigation_active = True
        self._history_position = target_index
        self._load_current_file()
        return True

    def _record_file_history(self, path: Path) -> None:
        self._ensure_history_state()
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            return
        history: List[Path] = self._file_history  # type: ignore[assignment]
        if self._history_navigation_active:
            self._history_navigation_active = False
            for idx, existing in enumerate(history):
                try:
                    if existing.resolve() == resolved:
                        self._history_position = idx
                        self._file_history = history
                        self._trim_file_history()
                        self._persist_file_history()
                        return
                except FileNotFoundError:
                    continue
            # if not found, append as new entry
            history.append(resolved)
            self._history_position = len(history) - 1
            self._file_history = history
            self._trim_file_history()
            self._persist_file_history()
            return
        current_index = self._history_position
        if 0 <= current_index < len(history) - 1:
            del history[current_index + 1 :]
        if history and history[-1] == resolved:
            self._history_position = len(history) - 1
            self._file_history = history
            self._trim_file_history()
            self._persist_file_history()
            return
        history.append(resolved)
        self._history_position = len(history) - 1
        self._file_history = history
        self._trim_file_history()
        self._persist_file_history()

    def _load_current_file(self) -> None:
        if not self.parquet_files:
            print("[PoseViewer] _load_current_file skipped (no files available)")
            return
        if not (0 <= self.current_file_index < len(self.parquet_files)):
            self.current_file_index = min(max(self.current_file_index, 0), len(self.parquet_files) - 1)
        self._force_pause_playback()
        path = self.parquet_files[self.current_file_index]
        self._set_file_header_text(path)
        self._update_file_menu()
        if hasattr(self, "title_bar"):
            folder = path.parent.name or str(path.parent)
            self.title_bar.set_subtitle(folder)
        self._set_status(f"Loading {path.name}…")
        self.progressbar.setRange(0, 100)
        self.progress_var.set(0.0)
        self._queue_progress_update(0.0, f"Loading {path.name}…")
        self.mouse_colors.clear()
        self.current_data = None
        self._clear_preserved_camera()
        self._scene_begin_frame(xlim=(0.0, 1.0), ylim=(0.0, 1.0), aspect=1.0)
        self._scene_finalize_frame()
        self._redraw_scene()

        if path in self.data_cache:
            cached_payload = self.data_cache[path]
            summary_message = self._format_loaded_status(path, cached_payload)
            self._queue_progress_update(1.0, summary_message)
            self._on_file_loaded(path, cached_payload)
            return

        cached_data = self._load_cached_data(path)
        if cached_data is not None:
            self.data_cache[path] = cached_data
            summary_message = self._format_loaded_status(path, cached_data)
            self._queue_progress_update(1.0, summary_message)
            self._on_file_loaded(path, cached_data)
            return

        def worker() -> None:
            try:
                data = self._prepare_data(
                    path,
                    progress_callback=lambda fraction, message: self._queue_progress_update(fraction, message),
                )
            except Exception as exc:  # pragma: no cover - UI feedback path
                QtCore.QTimer.singleShot(0, lambda exc=exc: self._handle_load_error(path, exc))
                return

            def finalize() -> None:
                self.data_cache[path] = data
                self._on_file_loaded(path, data)

            self._invoke_on_main_thread(finalize)

            summary_message = self._format_loaded_status(path, data)
            if not getattr(self, "cache_enabled", True):
                self._queue_progress_update(1.0, summary_message)
                return

            cache_message = summary_message

            def _toggle_camera_interaction(enabled: bool) -> None:
                scene = getattr(self, "scene", None)
                setter = getattr(scene, "set_camera_interaction_enabled", None) if scene is not None else None
                if callable(setter):
                    setter(enabled, reason="caching")

            self._queue_progress_update(0.99, f"Caching {path.name}…")
            self._invoke_on_main_thread(lambda: _toggle_camera_interaction(False))
            start_time = time.perf_counter()
            cache_stored = False
            try:
                self._store_cached_data(path, data)
                cache_stored = True
            except Exception:
                cache_message = f"{summary_message} | cache skipped"
            finally:
                duration = time.perf_counter() - start_time
                if cache_stored:
                    print(f"[PoseViewer] cached {path.name} in {duration:.3f}s")
                self._invoke_on_main_thread(lambda: _toggle_camera_interaction(True))

            self._queue_progress_update(1.0, cache_message)

        def _schedule_initial_load(self) -> None:
            if not getattr(self, "parquet_files", None):
                print("[PoseViewer] _schedule_initial_load postponed (no files yet)")
                return
            if getattr(self, "_initial_load_scheduled", False):
                return
            self._initial_load_scheduled = True
            def _run_initial_load() -> None:
                apply_target = getattr(self, "_apply_pending_initial_target", None)
                applied = bool(callable(apply_target) and apply_target())
                if applied:
                    update_menu = getattr(self, "_update_file_menu", None)
                    if callable(update_menu):
                        update_menu()
                self._load_current_file()
            QtCore.QTimer.singleShot(0, _run_initial_load)

        if self.loading_thread and self.loading_thread.is_alive():
            self.loading_thread.join(timeout=0.1)
        self.loading_thread = threading.Thread(target=worker, daemon=True)
        self.loading_thread.start()

    def _handle_load_error(self, path: Path, exc: Exception) -> None:
        self._set_status(f"Failed to load {path.name}: {exc}")
        self._show_error("Error loading parquet", f"{path}\n\n{exc}")

    def _on_file_loaded(self, path: Path, data: Dict[str, object]) -> None:
        self.current_data = data
        frame_count = len(data["frames"])  # type: ignore[index]
        self.frame_slider.setMaximum(max(0, frame_count - 1))
        self.frame_var.set(0)
        self.slider_active = True
        self.frame_slider.setValue(0)
        self.slider_active = False
        self._set_speed_value(self.playback_speed_multiplier)
        self._configure_playback_timeline(path=path, data=data)
        self.progress_var.set(100.0)
        status_message = self._format_loaded_status(path, data, frame_count=frame_count)
        self._set_status(status_message)
        if data.get("has_behaviors"):
            self._set_behavior("Behavior annotations loaded; press play!")
        else:
            self._set_behavior("No behavior annotations found for this file")
        self._render_frame(0)
        self._record_file_history(path)
        scene = getattr(self, "scene", None)
        if scene is not None and hasattr(scene, "show_scale_bar_hint"):
            scene.show_scale_bar_hint()

    def _format_loaded_status(
        self,
        path: Path,
        data: Dict[str, object],
        *,
        frame_count: Optional[int] = None,
    ) -> str:
        if frame_count is None:
            try:
                frame_count = len(data["frames"])  # type: ignore[index]
            except Exception:
                frame_count = None

        status_bits = [f"Loaded {path.name}"]
        if isinstance(frame_count, int):
            status_bits.append(f"{frame_count} frames")

        metadata_obj = data.get("metadata") if isinstance(data, dict) else None
        metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
        fps_value = metadata.get("frames_per_second") if isinstance(metadata, dict) else None
        if isinstance(fps_value, (int, float)) and math.isfinite(float(fps_value)):
            status_bits.append(f"{float(fps_value):.2f} fps")

        mouse_ids_obj = data.get("mouse_ids") if isinstance(data, dict) else None
        if isinstance(mouse_ids_obj, (list, tuple, set)):
            try:
                mouse_strings = sorted(str(mouse) for mouse in mouse_ids_obj)
            except Exception:
                mouse_strings = []
            if mouse_strings:
                status_bits.append(f"mice: {', '.join(mouse_strings)}")

        return " | ".join(status_bits)

    def _prepare_data(
        self,
        path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, object]:
        columns: List[str] = ["video_frame", "mouse_id", "x", "y"]
        parquet_file = None
        parquet_backend: Optional[str] = None
        if pq is not None:
            try:
                parquet_file = pq.ParquetFile(path)
                name = getattr(pq, "__name__", "")
                parquet_backend = "pyarrow" if name.startswith("pyarrow") else "fastparquet"
                schema_names = getattr(getattr(parquet_file, "schema", None), "names", None)
                if schema_names and "bodypart" in schema_names and "bodypart" not in columns:
                    columns.append("bodypart")
            except Exception:
                parquet_file = None
                parquet_backend = None

        if self.use_gpu and cudf is not None:
            try:
                return self._prepare_data_gpu(path, columns, progress_callback)
            except Exception as exc:
                if progress_callback:
                    progress_callback(0.0, f"GPU load failed ({exc}); retrying on CPU…")

        if parquet_file is not None and parquet_backend:
            try:
                if parquet_backend == "pyarrow":
                    return self._prepare_data_pyarrow(path, parquet_file, columns, progress_callback)
                if parquet_backend == "fastparquet":
                    return self._prepare_data_fastparquet(path, parquet_file, columns, progress_callback)
            except Exception as exc:
                if progress_callback:
                    progress_callback(0.0, f"{parquet_backend} loader error ({exc}); falling back to pandas…")

        return self._prepare_data_pandas(path, columns, progress_callback)

    def _prepare_data_gpu(
        self,
        path: Path,
        columns: Sequence[str],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, object]:
        if progress_callback:
            progress_callback(0.0, f"GPU loading {path.name}…")

        required_columns = {"video_frame", "mouse_id", "x", "y"}
        gdf = cudf.read_parquet(path, columns=list(dict.fromkeys(columns)))
        missing = required_columns.difference(gdf.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        gdf = gdf.dropna(subset=list(required_columns))
        if gdf.empty:
            raise ValueError("Parquet file contains no valid rows after filtering.")

        total_rows = len(gdf)
        gdf["video_frame"] = gdf["video_frame"].astype("int32")
        gdf["mouse_id"] = gdf["mouse_id"].astype("str")
        gdf["x"] = gdf["x"].astype("float32")
        gdf["y"] = gdf["y"].astype("float32")
        if "bodypart" in gdf.columns:
            gdf["bodypart"] = gdf["bodypart"].astype("str")

        if progress_callback:
            progress_callback(0.45, f"GPU normalized {total_rows} rows; transferring…")

        df_cpu = gdf.to_pandas()
        return self._prepare_data_from_dataframe(
            df_cpu,
            path,
            progress_callback,
            start_fraction=0.45,
        )

    def _prepare_data_pyarrow(
        self,
        path: Path,
        parquet_file: Any,
        columns: Sequence[str],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, object]:
        if progress_callback:
            progress_callback(0.0, f"Reading {path.name} metadata…")

        frame_points_map, frame_labels_map, mouse_ids, stats = self._initialize_accumulators()
        metadata = parquet_file.metadata
        total_rows = metadata.num_rows if metadata is not None else None
        processed_rows = 0
        row_groups = parquet_file.num_row_groups
        start_time = time.time()

        for group_index in range(row_groups):
            table = parquet_file.read_row_group(group_index, columns=list(dict.fromkeys(columns)))
            chunk = table.to_pandas(split_blocks=True, self_destruct=True)
            processed_rows += self._accumulate_chunk(
                chunk,
                frame_points_map,
                frame_labels_map,
                mouse_ids,
                stats,
            )

            if progress_callback:
                if total_rows and total_rows > 0:
                    fraction = min(1.0, processed_rows / total_rows)
                else:
                    fraction = (group_index + 1) / max(1, row_groups)
                message = self._format_progress_message(
                    path.name,
                    processed_rows,
                    total_rows,
                    start_time,
                    fraction,
                    group_index=group_index + 1,
                    group_total=row_groups,
                )
                progress_callback(fraction, message)

        if progress_callback:
            progress_callback(0.95, f"Finalizing {path.name}…")

        result = self._finalize_payloads(
            frame_points_map,
            frame_labels_map,
            mouse_ids,
            stats,
            tracking_path=path,
        )
        if progress_callback:
            progress_callback(1.0, f"Loaded {path.name}")
        return result

    def _prepare_data_fastparquet(
        self,
        path: Path,
        parquet_file: Any,
        columns: Sequence[str],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, object]:
        if progress_callback:
            progress_callback(0.0, f"Reading {path.name} metadata…")

        frame_points_map, frame_labels_map, mouse_ids, stats = self._initialize_accumulators()
        row_groups = getattr(parquet_file, "row_groups", [])
        total_rows = sum(getattr(rg, "num_rows", 0) for rg in row_groups) if row_groups else None
        processed_rows = 0
        start_time = time.time()

        column_list = list(dict.fromkeys(columns))
        iterator = parquet_file.iter_row_groups(columns=column_list) if hasattr(parquet_file, "iter_row_groups") else []

        for index, chunk in enumerate(iterator):
            if isinstance(chunk, pd.DataFrame):
                df_chunk = chunk
            else:
                df_chunk = pd.DataFrame(chunk)
            processed_rows += self._accumulate_chunk(
                df_chunk,
                frame_points_map,
                frame_labels_map,
                mouse_ids,
                stats,
            )

            if progress_callback:
                if total_rows and total_rows > 0:
                    fraction = min(1.0, processed_rows / total_rows)
                else:
                    fraction = (index + 1) / max(1, len(row_groups)) if row_groups else 1.0
                message = self._format_progress_message(
                    path.name,
                    processed_rows,
                    total_rows,
                    start_time,
                    fraction,
                    group_index=index + 1,
                    group_total=len(row_groups) if row_groups else index + 1,
                )
                progress_callback(fraction, message)

        if processed_rows == 0:
            df_full = parquet_file.to_pandas(columns=column_list)
            processed_rows = self._accumulate_chunk(
                df_full,
                frame_points_map,
                frame_labels_map,
                mouse_ids,
                stats,
            )
            if progress_callback:
                progress_callback(0.4, f"Buffered {processed_rows} rows from {path.name}")

        if progress_callback:
            progress_callback(0.95, f"Finalizing {path.name}…")

        result = self._finalize_payloads(
            frame_points_map,
            frame_labels_map,
            mouse_ids,
            stats,
            tracking_path=path,
        )
        if progress_callback:
            progress_callback(1.0, f"Loaded {path.name}")
        return result

    def _prepare_data_pandas(
        self,
        path: Path,
        columns: Sequence[str],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, object]:
        if progress_callback:
            progress_callback(0.0, f"Reading {path.name} into pandas…")

        try:
            df = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
        except ImportError as exc:
            raise RuntimeError(
                "Reading parquet files requires either the 'pyarrow' or 'fastparquet' package. "
                "Install one of them (e.g. `pip install pyarrow`) and restart the viewer."
            ) from exc
        except (KeyError, ValueError):
            try:
                df = pd.read_parquet(path)
            except ImportError as exc:
                raise RuntimeError(
                    "Reading parquet files requires either the 'pyarrow' or 'fastparquet' package. "
                    "Install one of them (e.g. `pip install pyarrow`) and restart the viewer."
                ) from exc

        return self._prepare_data_from_dataframe(df, path, progress_callback, start_fraction=0.3)

    def _prepare_data_from_dataframe(
        self,
        df: pd.DataFrame,
        path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        start_fraction: float = 0.3,
    ) -> Dict[str, object]:
        frame_points_map, frame_labels_map, mouse_ids, stats = self._initialize_accumulators()
        filename = path.name

        required_columns = ["video_frame", "mouse_id", "x", "y"]
        df = df.copy()
        df = df.dropna(subset=required_columns)
        if df.empty:
            raise ValueError("Parquet file contains no valid rows after filtering.")

        df["video_frame"] = df["video_frame"].astype(np.int32, copy=False)
        df["mouse_id"] = df["mouse_id"].astype(str)
        df["x"] = df["x"].astype(np.float32, copy=False)
        df["y"] = df["y"].astype(np.float32, copy=False)
        if "bodypart" in df.columns:
            df["bodypart"] = df["bodypart"].astype(str)

        df.sort_values(["video_frame", "mouse_id"], inplace=True, kind="mergesort", ignore_index=True)

        total_rows = int(len(df))
        chunk_size = max(10_000, min(200_000, total_rows // 8 or total_rows))
        processed_rows = 0
        start_time = time.time()

        if progress_callback:
            initial_fraction = max(0.0, min(1.0, start_fraction))
            progress_callback(initial_fraction, f"Processing {filename}…")

        for start in range(0, total_rows, chunk_size):
            end = min(total_rows, start + chunk_size)
            chunk = df.iloc[start:end]
            processed_rows += self._accumulate_chunk(
                chunk,
                frame_points_map,
                frame_labels_map,
                mouse_ids,
                stats,
                presorted=True,
                preprocessed=True,
            )

            if progress_callback:
                relative = processed_rows / max(1, total_rows)
                scaled_fraction = start_fraction + relative * (0.95 - start_fraction)
                fraction = max(0.0, min(0.95, scaled_fraction))
                message = self._format_progress_message(
                    filename,
                    processed_rows,
                    total_rows,
                    start_time,
                    fraction,
                )
                progress_callback(fraction, message)

        if progress_callback:
            message = f"Finalizing {filename} ({processed_rows} rows)…"
            completion_fraction = min(0.95, start_fraction + 0.45)
            progress_callback(completion_fraction, message)

        result = self._finalize_payloads(
            frame_points_map,
            frame_labels_map,
            mouse_ids,
            stats,
            tracking_path=path,
        )
        if progress_callback:
            progress_callback(1.0, f"Loaded {filename}")
        return result

    @staticmethod
    def _initialize_accumulators() -> Tuple[
        Dict[int, Dict[str, List[np.ndarray]]],
        Dict[int, Dict[str, List[str]]],
        Set[str],
        Dict[str, float],
    ]:
        frame_points_map: Dict[int, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
        frame_labels_map: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        mouse_ids: Set[str] = set()
        stats = {
            "x_min": float("inf"),
            "x_max": float("-inf"),
            "y_min": float("inf"),
            "y_max": float("-inf"),
        }
        return frame_points_map, frame_labels_map, mouse_ids, stats

    def _accumulate_chunk(
        self,
        chunk: pd.DataFrame,
        frame_points_map: Dict[int, Dict[str, List[np.ndarray]]],
        frame_labels_map: Dict[int, Dict[str, List[str]]],
        mouse_ids: Set[str],
        stats: Dict[str, float],
        *,
        presorted: bool = False,
        preprocessed: bool = False,
    ) -> int:
        if chunk.empty:
            return 0

        required_columns = ["video_frame", "mouse_id", "x", "y"]

        if not preprocessed:
            chunk = chunk.dropna(subset=required_columns)
            if chunk.empty:
                return 0
            chunk = chunk.copy()
            chunk["video_frame"] = chunk["video_frame"].astype(np.int32, copy=False)
            chunk["mouse_id"] = chunk["mouse_id"].astype(str)
            chunk["x"] = chunk["x"].astype(np.float32, copy=False)
            chunk["y"] = chunk["y"].astype(np.float32, copy=False)
            if "bodypart" in chunk.columns:
                chunk["bodypart"] = chunk["bodypart"].astype(str)
        else:
            # Ensure we operate on a view suitable for numpy access without mutating the original frame
            if not presorted:
                chunk = chunk.copy()

        if not presorted:
            chunk = chunk.sort_values(["video_frame", "mouse_id"], kind="mergesort", ignore_index=True)

        frame_values = chunk["video_frame"].to_numpy(np.int32, copy=False)
        mouse_values = chunk["mouse_id"].to_numpy(dtype=object, copy=False)
        x_values = chunk["x"].to_numpy(np.float32, copy=False)
        y_values = chunk["y"].to_numpy(np.float32, copy=False)
        coords = np.column_stack((x_values, y_values))

        has_bodypart = "bodypart" in chunk.columns
        body_values = chunk["bodypart"].to_numpy(dtype=object, copy=False) if has_bodypart else None

        stats["x_min"] = min(stats["x_min"], float(coords[:, 0].min()))
        stats["x_max"] = max(stats["x_max"], float(coords[:, 0].max()))
        stats["y_min"] = min(stats["y_min"], float(coords[:, 1].min()))
        stats["y_max"] = max(stats["y_max"], float(coords[:, 1].max()))

        processed = 0
        index = 0
        total = len(chunk)

        while index < total:
            frame_number = int(frame_values[index])
            frame_point_bucket = frame_points_map[frame_number]
            frame_label_bucket = frame_labels_map[frame_number]

            while index < total and frame_values[index] == frame_number:
                mouse_identifier = str(mouse_values[index])
                mouse_ids.add(mouse_identifier)

                end = index + 1
                while end < total and frame_values[end] == frame_number and mouse_values[end] == mouse_values[index]:
                    end += 1

                segment = np.array(coords[index:end], dtype=np.float32, copy=True)
                frame_point_bucket[mouse_identifier].append(segment)

                if has_bodypart and body_values is not None:
                    labels = [str(value) for value in body_values[index:end]]
                else:
                    labels = [f"bp-{offset}" for offset in range(end - index)]
                frame_label_bucket[mouse_identifier].extend(labels)

                processed += end - index
                index = end

        return processed

    def _finalize_payloads(
        self,
        frame_points_map: Dict[int, Dict[str, List[np.ndarray]]],
        frame_labels_map: Dict[int, Dict[str, List[str]]],
        mouse_ids: Set[str],
        stats: Dict[str, float],
        tracking_path: Optional[Path] = None,
    ) -> Dict[str, object]:
        if not frame_points_map:
            raise ValueError("Parquet file contains no frames after filtering.")

        frames = np.array(sorted(frame_points_map.keys()), dtype=int)
        frame_payloads: List[FramePayload] = []

        behavior_map = self._load_behavior_annotations_for_frames(tracking_path, frames) if tracking_path else {}
        metadata_obj = self._video_metadata_for_path(tracking_path) if tracking_path else None
        metadata: Dict[str, float] = dict(metadata_obj) if isinstance(metadata_obj, dict) else {}

        for frame in frames:
            mouse_groups: Dict[str, MouseGroup] = {}
            for mouse_id, point_segments in frame_points_map[frame].items():
                points = np.vstack(point_segments).astype(np.float32, copy=False)
                labels_list = frame_labels_map[frame].get(mouse_id, [])
                if len(labels_list) < len(points):
                    labels_list = labels_list + [f"bp-{idx}" for idx in range(len(points) - len(labels_list))]
                labels = tuple(labels_list[: len(points)])
                mouse_groups[mouse_id] = MouseGroup(points=points, labels=labels)
            frame_behaviors = behavior_map.get(int(frame), {}) if behavior_map else {}
            frame_payloads.append(
                FramePayload(
                    frame_number=int(frame),
                    mouse_groups=mouse_groups,
                    behaviors=dict(frame_behaviors),
                )
            )

        x_min = stats["x_min"]
        x_max = stats["x_max"]
        y_min = stats["y_min"]
        y_max = stats["y_max"]

        if not all(math.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
            raise ValueError("Parquet file contains insufficient coordinate data after filtering.")

        pad_x_data = max(10.0, (x_max - x_min) * 0.05 if math.isfinite(x_max - x_min) else 10.0)
        pad_y_data = max(10.0, (y_max - y_min) * 0.05 if math.isfinite(y_max - y_min) else 10.0)

        def _valid(value: object) -> bool:
            return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0

        arena_width_px, arena_height_px = self._metadata_arena_size_px(metadata)
        video_width_px, video_height_px = self._metadata_video_size_px(metadata)

        if metadata:
            pixels_per_cm = metadata.get("pixels_per_cm") or metadata.get("pix_per_cm_approx")
            arena_width_cm = metadata.get("arena_width_cm")
            arena_height_cm = metadata.get("arena_height_cm")
            if arena_width_px is None and _valid(pixels_per_cm) and _valid(arena_width_cm):
                arena_width_px = float(pixels_per_cm) * float(arena_width_cm)
                metadata["arena_width_px"] = arena_width_px
            if arena_height_px is None and _valid(pixels_per_cm) and _valid(arena_height_cm):
                arena_height_px = float(pixels_per_cm) * float(arena_height_cm)
                metadata["arena_height_px"] = arena_height_px

        data_x_lower = x_min - pad_x_data
        data_x_upper = x_max + pad_x_data
        data_y_lower = y_min - pad_y_data
        data_y_upper = y_max + pad_y_data

        bounds_x_lower = data_x_lower
        bounds_x_upper = data_x_upper
        bounds_y_lower = data_y_lower
        bounds_y_upper = data_y_upper

        data_center_x = (data_x_lower + data_x_upper) * 0.5
        data_center_y = (data_y_lower + data_y_upper) * 0.5
        center_x = data_center_x
        center_y = data_center_y

        target_ratio: Optional[float] = None
        preferred_center: Optional[Tuple[float, float]] = None

        domain_x_lower: Optional[float] = None
        domain_x_upper: Optional[float] = None
        domain_y_lower: Optional[float] = None
        domain_y_upper: Optional[float] = None

        if _valid(arena_width_px) and _valid(arena_height_px):
            arena_w = float(arena_width_px)
            arena_h = float(arena_height_px)
            arena_pad_x = max(pad_x_data, arena_w * 0.03)
            arena_pad_y = max(pad_y_data, arena_h * 0.03)
            bounds_x_lower = min(bounds_x_lower, 0.0)
            bounds_x_upper = max(bounds_x_upper, arena_w + arena_pad_x)
            bounds_y_lower = min(bounds_y_lower, 0.0)
            bounds_y_upper = max(bounds_y_upper, arena_h + arena_pad_y)
            target_ratio = arena_w / max(arena_h, 1e-6)
            preferred_center = (arena_w * 0.5, arena_h * 0.5)
            domain_x_lower = 0.0
            domain_x_upper = arena_w
            domain_y_lower = 0.0
            domain_y_upper = arena_h
        elif _valid(video_width_px) and _valid(video_height_px):
            video_w = float(video_width_px)
            video_h = float(video_height_px)
            video_pad_x = max(pad_x_data, video_w * 0.03)
            video_pad_y = max(pad_y_data, video_h * 0.03)
            bounds_x_lower = min(bounds_x_lower, 0.0)
            bounds_x_upper = max(bounds_x_upper, video_w + video_pad_x)
            bounds_y_lower = min(bounds_y_lower, 0.0)
            bounds_y_upper = max(bounds_y_upper, video_h + video_pad_y)
            target_ratio = video_w / max(video_h, 1e-6)
            preferred_center = (video_w * 0.5, video_h * 0.5)
            domain_x_lower = 0.0
            domain_x_upper = video_w
            domain_y_lower = 0.0
            domain_y_upper = video_h

        half_width = max(
            center_x - bounds_x_lower,
            bounds_x_upper - center_x,
            pad_x_data,
            0.5,
        )
        half_height = max(
            center_y - bounds_y_lower,
            bounds_y_upper - center_y,
            pad_y_data,
            0.5,
        )

        if preferred_center:
            half_width = max(half_width, abs(preferred_center[0] - center_x) + pad_x_data)
            half_height = max(half_height, abs(preferred_center[1] - center_y) + pad_y_data)

        width = max(half_width * 2.0, 1.0)
        height = max(half_height * 2.0, 1.0)

        if target_ratio is None:
            arena_ratio = metadata.get("arena_aspect_ratio") if metadata else None
            video_ratio = metadata.get("video_aspect_ratio") if metadata else None
            if _valid(arena_ratio):
                target_ratio = float(arena_ratio)
            elif _valid(video_ratio):
                target_ratio = float(video_ratio)
            elif height > 1e-6:
                target_ratio = max(width / max(height, 1e-6), 1e-6)
            else:
                target_ratio = 1.0

        target_ratio = float(max(target_ratio or 1.0, 1e-6))

        current_ratio = width / height if height > 1e-6 else target_ratio
        if current_ratio < target_ratio:
            required_width = target_ratio * height
            half_width = max(half_width, required_width * 0.5)
        elif current_ratio > target_ratio:
            required_height = width / target_ratio
            half_height = max(half_height, required_height * 0.5)

        half_width = max(
            half_width,
            center_x - bounds_x_lower,
            bounds_x_upper - center_x,
        )
        half_height = max(
            half_height,
            center_y - bounds_y_lower,
            bounds_y_upper - center_y,
        )

        width = max(half_width * 2.0, 1.0)
        height = max(half_height * 2.0, 1.0)

        x_lower = center_x - half_width
        x_upper = center_x + half_width
        y_lower = center_y - half_height
        y_upper = center_y + half_height

        # If metadata-derived domains exist, expand them to include the observed data bounds so
        # that downstream geometry never clips the trajectories when arena information is smaller
        # than the actual tracking coordinates. This also keeps the aspect logic squarely focused on
        # the arena ratio while letting the view grow to fit the data envelope.
        if domain_x_lower is not None:
            domain_x_lower = float(min(domain_x_lower, data_x_lower))
        if domain_x_upper is not None:
            domain_x_upper = float(max(domain_x_upper, data_x_upper))
        if domain_y_lower is not None:
            domain_y_lower = float(min(domain_y_lower, data_y_lower))
        if domain_y_upper is not None:
            domain_y_upper = float(max(domain_y_upper, data_y_upper))

        if domain_x_lower is not None:
            domain_x_lower = float(min(domain_x_lower, x_min))
        if domain_x_upper is not None:
            domain_x_upper = float(max(domain_x_upper, x_max))
        if domain_y_lower is not None:
            domain_y_lower = float(min(domain_y_lower, max(y_min, 0.0)))
        if domain_y_upper is not None:
            domain_y_upper = float(max(domain_y_upper, y_max))

        if domain_x_lower is not None and domain_x_upper is not None:
            x_lower = float(domain_x_lower)
            x_upper = float(domain_x_upper)
            width = max(x_upper - x_lower, 1.0)
            center_x = (x_lower + x_upper) * 0.5
        if domain_y_lower is not None and domain_y_upper is not None:
            y_lower = float(domain_y_lower)
            y_upper = float(domain_y_upper)
            height = max(y_upper - y_lower, 1.0)
            center_y = (y_lower + y_upper) * 0.5

        if domain_x_lower is not None and x_lower < domain_x_lower:
            shift = domain_x_lower - x_lower
            x_lower += shift
            x_upper += shift
        if domain_y_lower is not None and y_lower < domain_y_lower:
            shift = domain_y_lower - y_lower
            y_lower += shift
            y_upper += shift

        if domain_x_upper is not None and x_upper < domain_x_upper:
            x_upper = domain_x_upper
        if domain_y_upper is not None and y_upper < domain_y_upper:
            y_upper = domain_y_upper

        center_x = (x_lower + x_upper) * 0.5
        center_y = (y_lower + y_upper) * 0.5

        xlim = (x_lower, x_upper)
        ylim = (y_lower, y_upper)

        result: Dict[str, object] = {
            "frames": frames,
            "payloads": frame_payloads,
            "xlim": xlim,
            "ylim": ylim,
            "mouse_ids": sorted(mouse_ids),
            "has_behaviors": bool(behavior_map),
            "display_aspect_ratio": target_ratio,
            "display_center": (center_x, center_y),
        }

        if domain_x_lower is not None or domain_x_upper is not None:
            result["domain_xlim"] = (
                float(domain_x_lower) if domain_x_lower is not None else None,
                float(domain_x_upper) if domain_x_upper is not None else None,
            )
        if domain_y_lower is not None or domain_y_upper is not None:
            result["domain_ylim"] = (
                float(domain_y_lower) if domain_y_lower is not None else None,
                float(domain_y_upper) if domain_y_upper is not None else None,
            )

        if metadata:
            self._augment_tracking_metadata(metadata)
            metadata.setdefault("display_aspect_ratio", target_ratio)
            metadata.setdefault("display_center_x", center_x)
            metadata.setdefault("display_center_y", center_y)
            result["metadata"] = metadata

        return result

    def _video_metadata_for_path(self, tracking_path: Path) -> Optional[Dict[str, float]]:
        dataset_flag: Optional[str] = None
        for part in tracking_path.parts:
            if part == "train_tracking":
                dataset_flag = "train"
                break
            if part == "test_tracking":
                dataset_flag = "test"
                break
        if dataset_flag is None:
            return None

        dataset_root: Optional[Path] = None
        for parent in tracking_path.parents:
            candidate = parent / f"{dataset_flag}.csv"
            if candidate.exists():
                dataset_root = parent
                break
        if dataset_root is None:
            return None

        tables: Dict[Tuple[Path, str], Dict[Tuple[str, str], Dict[str, float]]] = getattr(self, "_video_metadata_tables", {})
        key = (dataset_root.resolve(), dataset_flag)
        table = tables.get(key)
        if table is None:
            csv_path = dataset_root / f"{dataset_flag}.csv"
            table = {}
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                table = {}
            else:
                def _safe_float(value: object) -> Optional[float]:
                    try:
                        result = float(value)
                    except (TypeError, ValueError):
                        return None
                    if math.isnan(result):
                        return None
                    return result

                df["lab_id"] = df["lab_id"].astype(str)
                df["video_id"] = df["video_id"].astype(str)
                numeric_columns = [
                    "frames_per_second",
                    "video_duration_sec",
                    "pix_per_cm_approx",
                    "video_width_pix",
                    "video_height_pix",
                    "arena_width_cm",
                    "arena_height_cm",
                ]
                for column in numeric_columns:
                    if column in df.columns:
                        df[column] = pd.to_numeric(df[column], errors="coerce")
                fps_values = df.get("frames_per_second")
                duration_values = df.get("video_duration_sec")
                for idx, row in df.iterrows():
                    lab_id = str(row["lab_id"])
                    video_id = str(row["video_id"])
                    fps_val = float(fps_values.iloc[idx]) if fps_values is not None and pd.notna(fps_values.iloc[idx]) else None
                    duration_val = float(duration_values.iloc[idx]) if duration_values is not None and pd.notna(duration_values.iloc[idx]) else None
                    entry: Dict[str, float] = {}
                    if fps_val is not None:
                        entry["frames_per_second"] = fps_val
                    if duration_val is not None:
                        entry["video_duration_sec"] = duration_val
                    pix_per_cm_val = _safe_float(row.get("pix_per_cm_approx")) if "pix_per_cm_approx" in row else None
                    if pix_per_cm_val is not None:
                        entry["pixels_per_cm"] = pix_per_cm_val
                    video_width_val = _safe_float(row.get("video_width_pix")) if "video_width_pix" in row else None
                    video_height_val = _safe_float(row.get("video_height_pix")) if "video_height_pix" in row else None
                    if video_width_val is not None:
                        entry["video_width_px"] = video_width_val
                    if video_height_val is not None:
                        entry["video_height_px"] = video_height_val
                    if video_width_val is not None and video_height_val is not None and video_height_val > 0:
                        entry["video_aspect_ratio"] = video_width_val / video_height_val
                    arena_width_val = _safe_float(row.get("arena_width_cm")) if "arena_width_cm" in row else None
                    arena_height_val = _safe_float(row.get("arena_height_cm")) if "arena_height_cm" in row else None
                    if arena_width_val is not None:
                        entry["arena_width_cm"] = arena_width_val
                    if arena_height_val is not None:
                        entry["arena_height_cm"] = arena_height_val
                    if arena_width_val is not None and arena_height_val is not None and arena_height_val > 0:
                        entry["arena_aspect_ratio"] = arena_width_val / arena_height_val
                        if pix_per_cm_val is not None:
                            entry["arena_width_px"] = arena_width_val * pix_per_cm_val
                            entry["arena_height_px"] = arena_height_val * pix_per_cm_val
                    if entry:
                        table[(lab_id, video_id)] = entry
            tables[key] = table
            setattr(self, "_video_metadata_tables", tables)

        if not table:
            return None
        lab_id = tracking_path.parent.name
        video_id = tracking_path.stem
        metadata = table.get((lab_id, video_id))
        if metadata is None:
            return None
        return dict(metadata)

    def _load_behavior_annotations_for_frames(
        self,
        tracking_path: Optional[Path],
        frames: np.ndarray,
    ) -> Dict[int, Dict[str, str]]:
        if tracking_path is None or frames.size == 0:
            return {}

        annotation_path = self._infer_annotation_path(tracking_path)
        if annotation_path is None or not annotation_path.exists():
            return {}

        try:
            annotations = pd.read_parquet(annotation_path)
        except Exception:
            return {}

        if annotations.empty:
            return {}

        action_column = next((col for col in ("behavior", "action", "label", "state") if col in annotations.columns), None)
        if action_column is None:
            return {}

        start_col = "start_frame" if "start_frame" in annotations.columns else None
        stop_col = "stop_frame" if "stop_frame" in annotations.columns else None
        if start_col is None or stop_col is None:
            return {}

        agent_column = next((col for col in ("agent_id", "subject_id", "animal_id", "source_id") if col in annotations.columns), None)
        target_column = next((col for col in ("target_id", "object_id", "other_id", "partner_id") if col in annotations.columns), None)

        frames_sorted = np.unique(frames.astype(int))
        frame_min = int(frames_sorted[0])
        frame_max = int(frames_sorted[-1])

        annotations = annotations.copy()
        annotations[start_col] = annotations[start_col].astype(np.int64, copy=False)
        annotations[stop_col] = annotations[stop_col].astype(np.int64, copy=False)
        annotations = annotations[(annotations[stop_col] >= frame_min) & (annotations[start_col] <= frame_max)]
        if annotations.empty:
            return {}

        behaviors: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        for row in annotations.itertuples(index=False):
            start_frame = int(getattr(row, start_col))
            stop_frame = int(getattr(row, stop_col))
            if stop_frame < frame_min or start_frame > frame_max:
                continue

            action_value = getattr(row, action_column)
            if isinstance(action_value, str):
                action_text = action_value.replace("_", " ").strip()
            else:
                action_text = str(action_value)
            if not action_text or action_text.lower() == "nan":
                continue

            agent_id = getattr(row, agent_column) if agent_column else None
            target_id = getattr(row, target_column) if target_column else None
            agent_str = self._normalize_mouse_identifier(agent_id)
            target_str = self._normalize_mouse_identifier(target_id)

            lower_bound = np.searchsorted(frames_sorted, start_frame, side="left")
            upper_bound = np.searchsorted(frames_sorted, stop_frame, side="right")
            if lower_bound >= upper_bound:
                continue

            for frame in frames_sorted[lower_bound:upper_bound]:
                if agent_str:
                    descriptor = action_text if not target_str else f"{action_text}→{target_str}"
                    bucket = behaviors[int(frame)][agent_str]
                    if descriptor not in bucket:
                        bucket.append(descriptor)
                if target_str:
                    descriptor = action_text if not agent_str else f"{action_text}←{agent_str}"
                    bucket = behaviors[int(frame)][target_str]
                    if descriptor not in bucket:
                        bucket.append(descriptor)

        finalized: Dict[int, Dict[str, str]] = {}
        for frame, per_mouse in behaviors.items():
            finalized[frame] = {mouse: "; ".join(values) for mouse, values in per_mouse.items() if values}

        return finalized

    def _infer_annotation_path(self, tracking_path: Path) -> Optional[Path]:
        parts = tracking_path.parts
        lowered = [part.lower() for part in parts]
        mapping = {
            "train_tracking": ("train_annotation", "train_annotations"),
            "test_tracking": ("test_annotation", "test_annotations"),
        }

        for pivot, replacements in mapping.items():
            if pivot in lowered:
                idx = lowered.index(pivot)
                base = Path(*parts[:idx]) if idx > 0 else Path()
                remainder = Path(*parts[idx + 1:]) if idx + 1 < len(parts) else Path()
                for replacement in replacements:
                    candidate = base / replacement / remainder
                    if candidate.exists():
                        return candidate
                break

        fallback = tracking_path.with_name(f"{tracking_path.stem}_annotation{tracking_path.suffix}")
        if fallback.exists():
            return fallback
        return None

    @staticmethod
    def _format_progress_message(
        filename: str,
        processed_rows: int,
        total_rows: Optional[int],
        start_time: float,
        fraction: float,
        *,
        group_index: Optional[int] = None,
        group_total: Optional[int] = None,
    ) -> str:
        fraction = max(0.0, min(1.0, fraction))
        if total_rows and total_rows > 0:
            base = f"Loading {filename}: {processed_rows}/{total_rows} rows ({fraction * 100:.1f}%)"
        elif group_index is not None and group_total is not None:
            base = f"Loading {filename}: group {group_index}/{group_total} ({fraction * 100:.1f}%)"
        else:
            base = f"Loading {filename}: {processed_rows} rows"

        if 0.0 < fraction < 1.0:
            elapsed = time.time() - start_time
            if fraction > 0:
                eta = max(0.0, elapsed * (1.0 - fraction) / max(fraction, 1e-6))
                base += f" | ETA {eta:.1f}s"
        return base


__all__ = ["PoseViewerFileMixin"]
