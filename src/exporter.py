"""Export handling mixin for the pose viewer application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PyQt6 import QtWidgets

from .models import FramePayload
from .optional_dependencies import imageio


class PoseViewerExportMixin:
    def _export_current_video(self) -> None:
        if not self.current_data:
            self._show_info("No data loaded", "Load a tracking file before exporting.")
            return

        if imageio is None:
            self._show_error(
                "Export unavailable",
                "The 'imageio' package is required to export animations. Install it and try again.",
            )
            return

        current_path = self.parquet_files[self.current_file_index] if self.parquet_files else None
        base_name = current_path.stem if current_path is not None else "export"

        filetypes = [
            ("MP4 video", "*.mp4"),
            ("GIF animation", "*.gif"),
            ("AVI video", "*.avi"),
            ("All files", "*.*"),
        ]

        filename = self._ask_save_filename(
            parent=self.root,
            title="Export animation",
            defaultextension=".mp4",
            initialfile=f"{base_name}.mp4",
            filetypes=filetypes,
        )
        if not filename:
            return

        export_path = Path(filename)
        if not export_path.suffix:
            export_path = export_path.with_suffix(".mp4")

        try:
            self._export_animation(export_path)
        except Exception as exc:  # pragma: no cover - user feedback path
            self._set_status(f"Export failed: {exc}")
            self._show_error("Export failed", f"Could not export animation:\n\n{exc}")
            return

        self._set_status(f"Exported {export_path.name}")
        self._show_info("Export complete", f"Saved animation to:\n{export_path}")

    def _export_animation(self, destination: Path) -> None:
        if imageio is None:
            raise RuntimeError("imageio is not available")
        if not self.current_data:
            raise RuntimeError("No data loaded")

        payloads: List[FramePayload] = self.current_data.get("payloads", [])  # type: ignore[assignment]
        if not payloads:
            raise RuntimeError("No frames available for export")

        ext = destination.suffix.lower()
        multiplier = float(self.speed_var.get() or 1.0)
        base_fps = float(self.current_data.get("base_fps", getattr(self, "playback_base_fps", 30.0)))
        fps = float(max(1.0, min(240.0, base_fps * multiplier)))

        writer: Optional[Any]
        needs_even_padding = False
        if ext in {".mp4", ".m4v", ".mov"}:
            needs_even_padding = True
            writer = imageio.get_writer(
                str(destination),
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
        elif ext == ".gif":
            writer = imageio.get_writer(
                str(destination),
                mode="I",
                duration=1.0 / fps,
                loop=0,
            )
        elif ext == ".avi":
            needs_even_padding = True
            writer = imageio.get_writer(
                str(destination),
                fps=fps,
                codec="png",
            )
        else:
            raise ValueError(f"Unsupported export format: {destination.suffix}")

        original_index = int(self.frame_slider.value())
        original_status = self.status_var.get()
        original_progress_range = (self.progressbar.minimum(), self.progressbar.maximum())
        original_progress_value = self.progress_var.get()
        was_playing = self.playing

        trail_cache_backup = self.trail_cache
        tail_histories_backup = self.tail_histories

        self._force_pause_playback()
        if hasattr(self, "export_action"):
            self.export_action.setEnabled(False)
        if hasattr(self, "export_button"):
            self.export_button.setEnabled(False)
        self.progressbar.setRange(0, 100)
        self.progress_var.set(0.0)
        self._set_status(f"Exporting {destination.name}…")
        self.trail_cache = {}
        self.tail_histories = {}

        try:
            with writer:
                total = len(payloads)
                for frame_idx in range(total):
                    self._render_frame(frame_idx)
                    frame_rgb = self._capture_canvas_rgb()
                    if needs_even_padding:
                        frame_rgb = self._ensure_even_dimensions(frame_rgb)
                    writer.append_data(frame_rgb)

                    progress = (frame_idx + 1) / total
                    self.progress_var.set(progress * 100.0)
                    self._set_status(f"Exporting {destination.name} ({frame_idx + 1}/{total})…")
                    QtWidgets.QApplication.processEvents()
        finally:
            self.trail_cache = trail_cache_backup
            self.tail_histories = tail_histories_backup

            self.slider_active = True
            self.frame_slider.setValue(original_index)
            self.slider_active = False
            if self.current_data:
                self._render_frame(original_index)

            self.progressbar.setRange(*original_progress_range)
            self.progress_var.set(original_progress_value)
            self._set_status(original_status)
            if hasattr(self, "export_action"):
                self.export_action.setEnabled(True)
            if hasattr(self, "export_button"):
                self.export_button.setEnabled(True)

            if was_playing:
                self.playing = True
                updater = getattr(self, "_update_play_button", None)
                if callable(updater):
                    updater()
                self._reset_playback_clock()
                self._schedule_next_frame()
            else:
                self.playing = False
                updater = getattr(self, "_update_play_button", None)
                if callable(updater):
                    updater()

    def _capture_canvas_rgb(self) -> np.ndarray:
        frame = self.scene.capture_frame()
        if frame.ndim == 3 and frame.shape[2] > 3:
            frame = frame[:, :, :3]
        return np.ascontiguousarray(frame)

    @staticmethod
    def _ensure_even_dimensions(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        pad_y = height % 2
        pad_x = width % 2
        if pad_x or pad_y:
            frame = np.pad(frame, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")
        return frame


__all__ = ["PoseViewerExportMixin"]
