"""Playback, rendering, and interaction mixin for the pose viewer application."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import time

import numpy as np

from .geometry import PoseViewerGeometryMixin
from .models import FramePayload, MouseGroup
from .plotting import SceneRect, get_palette_color


class PoseViewerPlaybackMixin(PoseViewerGeometryMixin):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cm_per_pixel: Optional[float] = None
        self._unit_label: str = "pixels"
        self._camera_rect: Optional[SceneRect] = None
        self._camera_override_rect: Optional[SceneRect] = None
        self._playback_state_initialised = False
        self._playback_anchor_time: Optional[float] = None
        self._playback_anchor_media_time: float = 0.0
        self._playback_last_time: float = 0.0

    def _init_playback_state(self) -> None:
        if getattr(self, "_playback_state_initialised", False):
            return
        if not hasattr(self, "scene"):
            return
        self._cm_per_pixel = getattr(self, "_cm_per_pixel", None)
        self._unit_label = getattr(self, "_unit_label", "pixels")
        self._camera_rect = None
        self._camera_override_rect = None
        self.scene.on_camera_change(self._handle_scene_camera_change)
        register_reset = getattr(self.scene, "on_reset_request", None)
        if callable(register_reset):
            register_reset(self._handle_reset_view_shortcut)
        self._playback_state_initialised = True

    def _handle_scene_camera_change(
        self,
        rect: SceneRect,
        is_user_override: bool,
        source: str = "system",
    ) -> None:
        print(
            f"[Playback] camera_change rect={rect} is_user_override={is_user_override} source={source}"
        )
        self._camera_rect = rect
        scene = getattr(self, "scene", None)
        scene_override = None
        scene_has_override = False
        previous_override = self._camera_override_rect
        if scene is not None:
            getter = getattr(scene, "get_camera_override_rect", None)
            if callable(getter):
                try:
                    scene_override = getter()
                except Exception as exc:
                    print(f"[Playback] camera_change override getter failed: {exc!r}")
            has_override_attr = getattr(scene, "has_user_camera_override", None)
            if isinstance(has_override_attr, bool):
                scene_has_override = has_override_attr
            elif callable(has_override_attr):  # defensive: property implemented as method
                try:
                    scene_has_override = bool(has_override_attr())
                except Exception as exc:
                    print(f"[Playback] camera_change has_override check failed: {exc!r}")

        if is_user_override:
            self._camera_override_rect = rect
        elif scene_has_override and scene_override is not None:
            # Scene reports an active user override even though the event flag was False.
            self._camera_override_rect = scene_override
            print(
                "[Playback] camera_change detected override via scene state; overriding stored rect"
            )
        elif source == "user":
            self._camera_override_rect = None
            print("[Playback] camera_change cleared stored override due to user reset")
        elif previous_override is not None and not scene_has_override:
            print(
                "[Playback] camera_change ignoring system clear; preserving stored override"
            )

        print(
            f"[Playback] camera_change stored override={self._camera_override_rect} scene_override={scene_override} scene_has_override={scene_has_override} source={source}"
        )

    def _restore_preserved_camera(self) -> None:
        rect = getattr(self, "_camera_override_rect", None)
        if rect is None:
            print("[Playback] _restore_preserved_camera skipped (no override)")
            return
        scene = getattr(self, "scene", None)
        if scene is None:
            print("[Playback] _restore_preserved_camera skipped (no scene)")
            return
        print(f"[Playback] _restore_preserved_camera applying rect={rect}")
        scene.apply_camera_rect(rect, as_user_override=True)

    def _clear_preserved_camera(self) -> None:
        print("[Playback] _clear_preserved_camera invoked")
        self._camera_rect = None
        self._camera_override_rect = None

    def _handle_reset_view_shortcut(self) -> None:
        scene = getattr(self, "scene", None)
        if scene is None:
            print("[Playback] reset_view shortcut ignored (no scene)")
            return
        print("[Playback] reset_view shortcut clearing override and resetting scene")
        self._camera_override_rect = None
        scene.reset_camera_view()

    def _resolve_cm_per_pixel(
        self,
        metadata: Mapping[str, Any],
        domain_xlim: Optional[Tuple[Optional[float], Optional[float]]],
        domain_ylim: Optional[Tuple[Optional[float], Optional[float]]],
    ) -> Optional[float]:
        def _to_positive_float(value: object) -> Optional[float]:
            if value is None:
                return None
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(num) or num <= 0.0:
                return None
            return num

        if not metadata:
            return None

        explicit_cm = _to_positive_float(metadata.get("cm_per_pixel"))
        if explicit_cm:
            return explicit_cm

        pixels_per_cm = metadata.get("pixels_per_cm") or metadata.get("pix_per_cm_approx")
        pixels_per_cm_val = _to_positive_float(pixels_per_cm)
        if pixels_per_cm_val:
            return 1.0 / pixels_per_cm_val

        pair_candidates: List[Tuple[Optional[float], Optional[float]]] = [
            (metadata.get("arena_width_px"), metadata.get("arena_width_cm")),
            (metadata.get("arena_height_px"), metadata.get("arena_height_cm")),
            (metadata.get("video_width_px"), metadata.get("video_width_cm")),
            (metadata.get("video_height_px"), metadata.get("video_height_cm")),
        ]
        for px_value, cm_value in pair_candidates:
            px = _to_positive_float(px_value)
            cm = _to_positive_float(cm_value)
            if px and cm:
                return cm / px

        arena_width_cm = _to_positive_float(metadata.get("arena_width_cm"))
        arena_height_cm = _to_positive_float(metadata.get("arena_height_cm"))
        if arena_width_cm and domain_xlim:
            left, right = domain_xlim
            px_width = _to_positive_float(
                float(right) - float(left)
                if left is not None and right is not None
                else None
            )
            if px_width:
                return arena_width_cm / px_width
        if arena_height_cm and domain_ylim:
            bottom, top = domain_ylim
            px_height = _to_positive_float(
                float(top) - float(bottom)
                if bottom is not None and top is not None
                else None
            )
            if px_height:
                return arena_height_cm / px_height
        return None

    def _points_to_scene_units(self, points: np.ndarray) -> np.ndarray:
        scale = self._cm_per_pixel
        array = points.astype(np.float32, copy=False)
        if not scale or not math.isfinite(scale) or scale <= 0.0:
            return array
        if math.isclose(scale, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            return array
        return (array.astype(np.float32, copy=True)) * float(scale)

    def _scalar_to_scene_units(self, value: float) -> float:
        scale = self._cm_per_pixel
        if scale and math.isfinite(scale) and scale > 0.0:
            return float(value) * float(scale)
        return float(value)

    def _configure_playback_timeline(self, *, path: Path, data: Dict[str, object]) -> None:
        frames_array = np.asarray(data.get("frames", []), dtype=np.int64)
        if frames_array.size == 0:
            self.playback_base_fps = 30.0
            self._frame_times = np.zeros(0, dtype=np.float64)
            self._frame_total_duration = 0.0
            self._playback_time = 0.0
            data["frame_times"] = self._frame_times
            data["base_fps"] = self.playback_base_fps
            data["video_duration"] = 0.0
            return

        meta_obj = data.get("metadata")
        metadata = meta_obj if isinstance(meta_obj, dict) else {}
        fps_candidate = metadata.get("frames_per_second") if isinstance(metadata, dict) else None
        duration_candidate = metadata.get("video_duration_sec") if isinstance(metadata, dict) else None

        fps_value = float(fps_candidate) if fps_candidate is not None else float("nan")
        duration_value = float(duration_candidate) if duration_candidate is not None else float("nan")
        if (not math.isfinite(fps_value) or fps_value <= 0.0) and math.isfinite(duration_value) and duration_value > 0.0:
            frame_span = float(frames_array[-1] - frames_array[0] + 1)
            fps_value = max(1.0, frame_span / duration_value)
        if not math.isfinite(fps_value) or fps_value <= 0.0:
            fps_value = 30.0

        frame_offsets = frames_array - int(frames_array[0])
        frame_times = frame_offsets.astype(np.float64) / max(fps_value, 1e-6)
        frame_interval = 1.0 / max(fps_value, 1e-6)
        duration_estimate = float(frame_times[-1] + frame_interval)
        if math.isfinite(duration_value) and duration_value > 0.0:
            total_duration = max(duration_estimate, duration_value)
        else:
            total_duration = duration_estimate

        self.playback_base_fps = float(max(fps_value, 1.0))
        self._frame_times = frame_times
        self._frame_total_duration = float(total_duration)
        self._playback_time = float(frame_times[0] if frame_times.size else 0.0)
        self._playback_frame_position = 0.0

        data["frame_times"] = frame_times
        data["base_fps"] = self.playback_base_fps
        data["video_duration"] = self._frame_total_duration

    def _frame_time_for_index(self, index: int) -> float:
        frame_times = getattr(self, "_frame_times", np.zeros(0, dtype=np.float64))
        if frame_times.size == 0:
            base_fps = max(1.0, getattr(self, "playback_base_fps", 30.0))
            return float(max(0, index)) / base_fps
        clamped = int(np.clip(index, 0, frame_times.size - 1))
        return float(frame_times[clamped])

    def _time_to_fractional_index(self, playback_time: float) -> float:
        frame_times = getattr(self, "_frame_times", np.zeros(0, dtype=np.float64))
        if frame_times.size == 0:
            return float(getattr(self, "_playback_frame_position", 0.0))
        value = float(max(0.0, playback_time))
        if value <= frame_times[0]:
            return 0.0
        if value >= frame_times[-1]:
            return float(frame_times.size - 1)
        indices = np.arange(frame_times.size, dtype=np.float64)
        return float(np.interp(value, frame_times, indices))

    def _toggle_play(self) -> None:
        if not self.current_data:
            return
        self.playing = not self.playing
        updater = getattr(self, "_update_play_button", None)
        if callable(updater):
            updater()
        if self.playing:
            self._playback_time = self._frame_time_for_index(int(self.frame_slider.value()))
            self._reset_playback_clock()
            self._schedule_next_frame()
        else:
            self._cancel_playback()
            self._clear_playback_anchor()

    def _schedule_next_frame(self) -> None:
        if not self.playing:
            return
        base_fps = max(1.0, getattr(self, "playback_base_fps", 30.0))
        multiplier = max(0.05, getattr(self, "playback_speed_multiplier", 1.0))
        desired_fps = max(1.0, base_fps * multiplier)
        delay_ms = max(1, int(round(1000.0 / desired_fps)))
        if not self.animation_timer.isActive():
            self.animation_timer.start(delay_ms)
        elif self.animation_timer.interval() != delay_ms:
            self.animation_timer.setInterval(delay_ms)

    def _set_playback_anchor(self) -> None:
        now = time.perf_counter()
        self._playback_anchor_time = now
        self._playback_anchor_media_time = float(getattr(self, "_playback_time", 0.0))
        self._playback_last_time = now

    def _clear_playback_anchor(self) -> None:
        self._playback_anchor_time = None

    def _reset_playback_clock(self) -> None:
        self._set_playback_anchor()
        current_time = float(getattr(self, "_playback_time", 0.0))
        frame_times = getattr(self, "_frame_times", np.zeros(0, dtype=np.float64))
        if frame_times.size > 0:
            self._playback_frame_position = float(self._time_to_fractional_index(current_time))
        else:
            existing = getattr(self, "_playback_frame_position", 0.0)
            self._playback_frame_position = float(existing)

    def _animation_step(self) -> None:
        if not self.current_data:
            return
        payloads: List[FramePayload] = self.current_data["payloads"]  # type: ignore[assignment]
        total_frames = len(payloads)
        if total_frames <= 0:
            return
        frame_times = getattr(self, "_frame_times", np.array([], dtype=np.float64))
        total_duration = float(getattr(self, "_frame_total_duration", 0.0))
        now = time.perf_counter()
        anchor_time = getattr(self, "_playback_anchor_time", None)
        anchor_media_time = getattr(self, "_playback_anchor_media_time", None)
        if anchor_time is None or not isinstance(anchor_media_time, (int, float)):
            anchor_time = now
            anchor_media_time = float(getattr(self, "_playback_time", 0.0))
            self._playback_anchor_time = anchor_time
            self._playback_anchor_media_time = anchor_media_time

        base_fps = max(1.0, getattr(self, "playback_base_fps", 30.0))
        speed = max(0.05, getattr(self, "playback_speed_multiplier", 1.0))

        if frame_times.size == 0 or total_duration <= 0.0:
            frame_interval = 1.0 / max(base_fps, 1e-6)
            elapsed_media_time = max(0.0, now - float(anchor_time)) * speed
            playback_time = float(anchor_media_time) + elapsed_media_time
            cycle_duration = frame_interval * float(max(total_frames, 1))
            if cycle_duration > 0.0:
                playback_time = math.fmod(playback_time, cycle_duration)
                if playback_time < 0.0:
                    playback_time += cycle_duration
            fractional_index = playback_time * base_fps
            if total_frames > 0:
                fractional_index %= float(total_frames)
            else:
                fractional_index = float(getattr(self, "_playback_frame_position", 0.0))
            base_index = int(np.clip(int(np.floor(fractional_index)), 0, max(total_frames - 1, 0))) if total_frames > 0 else 0
            self._playback_time = playback_time
            self._playback_frame_position = fractional_index
            self.slider_active = True
            self.frame_slider.setValue(base_index)
            self.slider_active = False
            self._render_frame_fractional(fractional_index)
        else:
            elapsed_media_time = max(0.0, now - float(anchor_time)) * speed
            playback_time = float(anchor_media_time) + elapsed_media_time
            playback_time = math.fmod(playback_time, total_duration)
            if playback_time < 0.0:
                playback_time += total_duration
            fractional_index = self._time_to_fractional_index(playback_time)
            base_index = int(np.clip(int(np.floor(fractional_index)), 0, total_frames - 1))
            self._playback_time = playback_time
            self._playback_frame_position = fractional_index
            self.slider_active = True
            self.frame_slider.setValue(base_index)
            self.slider_active = False
            self._render_frame_fractional(fractional_index)
        if self.playing:
            self._schedule_next_frame()

    def _cancel_playback(self) -> None:
        if self.animation_timer.isActive():
            self.animation_timer.stop()

    def _force_pause_playback(self) -> None:
        def _refresh_time_label() -> None:
            time_updater = getattr(self, "_update_time_label", None)
            if callable(time_updater):
                slider = getattr(self, "frame_slider", None)
                frame_value = 0
                if slider is not None:
                    try:
                        frame_value = int(slider.value())
                    except Exception:
                        frame_value = 0
                time_updater(frame_value)

        if not getattr(self, "playing", False) and not self.animation_timer.isActive():
            updater = getattr(self, "_update_play_button", None)
            if callable(updater):
                updater()
            _refresh_time_label()
            return
        self.playing = False
        self._cancel_playback()
        self._clear_playback_anchor()
        updater = getattr(self, "_update_play_button", None)
        if callable(updater):
            updater()
        _refresh_time_label()

    def _on_slider_change(self, value: str) -> None:
        if self.slider_active or not self.current_data:
            return
        index = int(float(value))
        self._playback_time = self._frame_time_for_index(index)
        self._playback_frame_position = float(index)
        self._render_frame(index)
        if self.playing:
            self._reset_playback_clock()

    def _on_speed_change(self, value: str) -> None:
        multiplier = max(0.05, float(value))
        self.playback_speed_multiplier = multiplier
        self._set_speed_value(multiplier)
        if self.playing:
            self._reset_playback_clock()
            self._schedule_next_frame()

    def _step_forward(self) -> None:
        if not self.current_data:
            return
        self._force_pause_playback()
        current = int(self.frame_slider.value())
        max_index = len(self.current_data["frames"]) - 1  # type: ignore[index]
        next_index = min(max_index, current + 1)
        self.slider_active = True
        self.frame_slider.setValue(next_index)
        self.slider_active = False
        self._render_frame(next_index)

    def _step_backward(self) -> None:
        if not self.current_data:
            return
        self._force_pause_playback()
        current = int(self.frame_slider.value())
        prev_index = max(0, current - 1)
        self.slider_active = True
        self.frame_slider.setValue(prev_index)
        self.slider_active = False
        self._render_frame(prev_index)

    def _render_frame(self, frame_index: int) -> None:
        if not self.current_data:
            return
        payloads: List[FramePayload] = self.current_data["payloads"]  # type: ignore[assignment]
        if not payloads:
            return
        frame_index = int(np.clip(frame_index, 0, len(payloads) - 1))
        frame_payload = payloads[frame_index]
        self._playback_frame_position = float(frame_index)
        self._playback_time = self._frame_time_for_index(frame_index)

        if self.frame_slider.maximum() != len(payloads) - 1:
            self.frame_slider.setRange(0, max(len(payloads) - 1, 0))

        self._render_pose_state(
            frame_payload.mouse_groups,
            frame_index_for_trails=frame_index,
            frame_label_value=frame_payload.frame_number,
            frame_title=f"Frame {frame_payload.frame_number}",
            behaviors=frame_payload.behaviors,
        )

    def _render_pose_state(
        self,
        mouse_groups: Dict[str, MouseGroup],
        *,
        frame_index_for_trails: int,
        frame_label_value: int,
        frame_title: str,
        behaviors: Mapping[str, str],
    ) -> None:
        if not self.current_data:
            return
        xlim = tuple(float(v) for v in self.current_data.get("xlim", (0.0, 1.0)))
        ylim = tuple(float(v) for v in self.current_data.get("ylim", (0.0, 1.0)))
        aspect_value = self.current_data.get("display_aspect_ratio") if isinstance(self.current_data, dict) else None
        metadata_obj = self.current_data.get("metadata") if isinstance(self.current_data, dict) else None
        metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
        video_size: Optional[Tuple[float, float]] = None
        arena_size: Optional[Tuple[float, float]] = None
        if metadata:
            vw = metadata.get("video_width_px")
            vh = metadata.get("video_height_px")
            if isinstance(vw, (int, float)) and isinstance(vh, (int, float)) and math.isfinite(float(vw)) and math.isfinite(float(vh)) and float(vw) > 0.0 and float(vh) > 0.0:
                video_size = (float(vw), float(vh))
            aw = metadata.get("arena_width_px")
            ah = metadata.get("arena_height_px")
            if isinstance(aw, (int, float)) and isinstance(ah, (int, float)) and math.isfinite(float(aw)) and math.isfinite(float(ah)) and float(aw) > 0.0 and float(ah) > 0.0:
                arena_size = (float(aw), float(ah))
        raw_domain_xlim = self.current_data.get("domain_xlim") if isinstance(self.current_data, dict) else None
        raw_domain_ylim = self.current_data.get("domain_ylim") if isinstance(self.current_data, dict) else None
        if isinstance(raw_domain_xlim, (tuple, list)) and len(raw_domain_xlim) >= 2:
            domain_xlim = (raw_domain_xlim[0], raw_domain_xlim[1])
        else:
            domain_xlim = (None, None)
        if isinstance(raw_domain_ylim, (tuple, list)) and len(raw_domain_ylim) >= 2:
            domain_ylim = (raw_domain_ylim[0], raw_domain_ylim[1])
        else:
            domain_ylim = (None, None)
        if arena_size is None:
            dx_min, dx_max = domain_xlim
            dy_min, dy_max = domain_ylim
            if all(isinstance(val, (int, float)) and math.isfinite(float(val)) for val in (dx_min, dx_max, dy_min, dy_max)):
                width_val = float(dx_max) - float(dx_min)
                height_val = float(dy_max) - float(dy_min)
                if width_val > 0.0 and height_val > 0.0:
                    arena_size = (width_val, height_val)

        metadata_mapping: Mapping[str, Any] = metadata if isinstance(metadata, Mapping) else {}
        cm_per_pixel = self._resolve_cm_per_pixel(metadata_mapping, domain_xlim, domain_ylim)
        self._cm_per_pixel = cm_per_pixel
        self._unit_label = "cm" if cm_per_pixel else "pixels"
        self.scene.set_unit_scale(cm_per_pixel=cm_per_pixel)
        self.scene.set_scene_dimensions(video_size=video_size, arena_size=arena_size)
        if isinstance(aspect_value, (int, float)) and math.isfinite(float(aspect_value)) and float(aspect_value) > 0.0:
            aspect = float(aspect_value)
        else:
            aspect = None
        override_rect = None
        scene_obj = getattr(self, "scene", None)
        scene_has_override = False
        if scene_obj is not None:
            getter = getattr(scene_obj, "get_camera_override_rect", None)
            if callable(getter):
                override_rect = getter()
            has_override_attr = getattr(scene_obj, "has_user_camera_override", None)
            if isinstance(has_override_attr, bool):
                scene_has_override = has_override_attr
            elif callable(has_override_attr):
                try:
                    scene_has_override = bool(has_override_attr())
                except Exception as exc:
                    print(f"[Playback] _render_pose_state has_override check failed: {exc!r}")
            if override_rect is None and scene_has_override:
                current_getter = getattr(scene_obj, "get_current_camera_rect", None)
                if callable(current_getter):
                    try:
                        override_rect = current_getter()
                    except Exception as exc:
                        print(f"[Playback] _render_pose_state current rect getter failed: {exc!r}")
        if override_rect is not None:
            self._camera_override_rect = override_rect
        preserve_view = scene_has_override or self._camera_override_rect is not None
        print(
            f"[Playback] _render_pose_state preserve_view={preserve_view} override_rect={self._camera_override_rect} scene_override={override_rect} scene_has_override={scene_has_override}"
        )
        self._scene_begin_frame(
            xlim=xlim,
            ylim=ylim,
            aspect=aspect,
            domain_xlim=domain_xlim,
            domain_ylim=domain_ylim,
            preserve_view=preserve_view,
        )
        if preserve_view:
            self._restore_preserved_camera()
        else:
            print("[Playback] _render_pose_state no override to restore")
        label_suffix = "cm" if cm_per_pixel else "pixels"
        self._update_canvas_labels(xlabel=f"X ({label_suffix})", ylabel=f"Y ({label_suffix})", title=frame_title)

        self._ensure_mouse_colors(mouse_groups.keys())
        self._update_trails(frame_index_for_trails)

        for mouse_id, group in mouse_groups.items():
            color = self.mouse_colors[mouse_id]
            self._draw_mouse_group(mouse_id, group, color)
            self._draw_trail(mouse_id, color)

        self._scene_finalize_frame()
        self._set_frame_value(int(frame_label_value))

        behaviors = dict(behaviors)
        behavior_lines: List[str] = []
        all_mouse_ids: Sequence[str] = self.current_data.get("mouse_ids", []) if self.current_data else []  # type: ignore[assignment]
        for mouse_id in all_mouse_ids:
            desc = behaviors.get(mouse_id)
            if desc:
                behavior_lines.append(f"Mouse {mouse_id}: {desc}")

        if behavior_lines:
            self._set_behavior("  |  ".join(behavior_lines))
        else:
            self._set_behavior("")

        self._redraw_scene()

    def _render_frame_fractional(self, frame_position: float) -> None:
        if not self.current_data:
            return
        payloads: List[FramePayload] = self.current_data["payloads"]  # type: ignore[assignment]
        total_frames = len(payloads)
        if total_frames == 0:
            return
        if self.frame_slider.maximum() != total_frames - 1:
            self.frame_slider.setRange(0, max(total_frames - 1, 0))
        position = float(frame_position)
        if total_frames > 0:
            position %= float(total_frames)
        base_index = int(np.floor(position)) if total_frames > 0 else 0
        base_index = int(np.clip(base_index, 0, total_frames - 1))
        t = position - float(base_index)
        if total_frames == 1:
            t = 0.0
            next_index = base_index
        else:
            next_index = (base_index + 1) if base_index < total_frames - 1 else base_index
            if base_index == total_frames - 1 and next_index == base_index:
                t = 0.0
        base_payload = payloads[base_index]
        if t <= 1e-4 or next_index == base_index:
            self._playback_frame_position = position
            self._render_pose_state(
                base_payload.mouse_groups,
                frame_index_for_trails=base_index,
                frame_label_value=base_payload.frame_number,
                frame_title=f"Frame {base_payload.frame_number}",
                behaviors=base_payload.behaviors,
            )
            return
        next_payload = payloads[next_index]
        interpolated_groups = self._interpolate_mouse_groups(base_payload.mouse_groups, next_payload.mouse_groups, t)
        display_value = (1.0 - t) * float(base_payload.frame_number) + t * float(next_payload.frame_number)
        frame_title = f"Frame {display_value:.2f}"
        self._playback_frame_position = position
        self._render_pose_state(
            interpolated_groups,
            frame_index_for_trails=base_index,
            frame_label_value=base_payload.frame_number,
            frame_title=frame_title,
            behaviors=base_payload.behaviors,
        )

    def _interpolate_mouse_groups(
        self,
        groups_a: Dict[str, MouseGroup],
        groups_b: Dict[str, MouseGroup],
        t: float,
    ) -> Dict[str, MouseGroup]:
        blend = float(np.clip(t, 0.0, 1.0))
        result: Dict[str, MouseGroup] = {}
        mouse_ids = sorted(set(groups_a.keys()) | set(groups_b.keys()))
        for mouse_id in mouse_ids:
            group_a = groups_a.get(mouse_id)
            group_b = groups_b.get(mouse_id)
            if group_a is None and group_b is not None:
                result[mouse_id] = group_b
                continue
            if group_b is None and group_a is not None:
                result[mouse_id] = group_a
                continue
            if group_a is None or group_b is None:
                continue
            result[mouse_id] = self._interpolate_mouse_group(group_a, group_b, blend)
        return result

    def _interpolate_mouse_group(self, group_a: MouseGroup, group_b: MouseGroup, t: float) -> MouseGroup:
        alpha = float(np.clip(t, 0.0, 1.0))
        labels_a = list(group_a.labels)
        labels_b = list(group_b.labels)
        order: List[str] = []
        seen = set()
        for label in labels_a:
            if label not in seen:
                order.append(label)
                seen.add(label)
        for label in labels_b:
            if label not in seen:
                order.append(label)
                seen.add(label)
        if not order:
            return MouseGroup(points=np.empty((0, 2), dtype=np.float32), labels=tuple())
        index_a = {label: idx for idx, label in enumerate(labels_a)}
        index_b = {label: idx for idx, label in enumerate(labels_b)}
        points: List[np.ndarray] = []
        for label in order:
            point_a = group_a.points[index_a[label]] if label in index_a else None
            point_b = group_b.points[index_b[label]] if label in index_b else None
            if point_a is not None and point_b is not None:
                interp = (1.0 - alpha) * point_a + alpha * point_b
            elif point_a is not None:
                interp = point_a
            elif point_b is not None:
                interp = point_b
            else:
                continue
            points.append(np.asarray(interp, dtype=np.float32))
        if not points:
            return MouseGroup(points=np.empty((0, 2), dtype=np.float32), labels=tuple(order))
        stacked = np.vstack(points).astype(np.float32, copy=False)
        return MouseGroup(points=stacked, labels=tuple(order))

    def _ensure_mouse_colors(self, mouse_ids: Iterable[str]) -> None:
        for idx, mouse_id in enumerate(sorted(mouse_ids)):
            if mouse_id in self.mouse_colors:
                continue
            self.mouse_colors[mouse_id] = get_palette_color(idx, palette="tab20")

    def _update_trails(self, frame_index: int) -> None:
        if not self.current_data:
            return
        payloads: List[FramePayload] = self.current_data["payloads"]  # type: ignore[assignment]
        start_index = max(0, frame_index - self.trail_length)
        history: Dict[str, List[np.ndarray]] = defaultdict(list)
        for idx in range(start_index, frame_index + 1):
            frame_payload = payloads[idx]
            for mouse_id, group in frame_payload.mouse_groups.items():
                core_points, _, tail_points, _ = self._split_tail_points(group)
                points_for_centroid = core_points if len(core_points) > 0 else group.points
                if points_for_centroid.size == 0:
                    continue
                centroid = points_for_centroid.mean(axis=0)
                history[mouse_id].append(centroid)

                tail_history = self.tail_histories.setdefault(mouse_id, deque(maxlen=20))
                if len(tail_points) > 0:
                    distances = np.linalg.norm(tail_points - centroid, axis=1)
                    tip_idx = int(np.argmax(distances))
                    tail_history.append(np.asarray(tail_points[tip_idx], dtype=np.float32))
                elif tail_history:
                    relaxed = (0.75 * tail_history[-1]) + (0.25 * centroid)
                    tail_history.append(relaxed.astype(np.float32))
        self.trail_cache = history

    def _split_tail_points(
        self,
        group: MouseGroup,
    ) -> Tuple[np.ndarray, Tuple[str, ...], np.ndarray, Tuple[str, ...]]:
        points = group.points
        if len(points) == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, tuple(), empty, tuple()

        labels_array = np.array(group.labels, dtype=object)
        tail_mask = np.zeros(len(points), dtype=bool)

        if labels_array.size:
            tail_mask = np.array(["tail" in str(label).lower() for label in labels_array], dtype=bool)

        if not tail_mask.any() and len(points) >= 4:
            centroid = points.mean(axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            median = float(np.median(distances))
            mad = float(np.median(np.abs(distances - median)))
            spread = mad if mad > 1e-6 else float(np.std(distances))
            if spread > 1e-6:
                threshold = median + 3.2 * spread
                tail_mask = distances > threshold
                if tail_mask.sum() > max(2, len(points) // 2):
                    top_indices = np.argsort(distances)[-max(2, len(points) // 4):]
                    mask = np.zeros_like(tail_mask)
                    mask[top_indices] = True
                    tail_mask = mask

        core_indices = np.where(~tail_mask)[0]
        tail_indices = np.where(tail_mask)[0]

        core_points = points[core_indices]
        core_labels = tuple(str(labels_array[i]) for i in core_indices) if labels_array.size else tuple()
        tail_points = points[tail_indices] if tail_indices.size else np.empty((0, 2), dtype=np.float32)
        tail_labels = tuple(str(labels_array[i]) for i in tail_indices) if labels_array.size else tuple()

        if not core_labels:
            core_labels = tuple(f"bp-{idx}" for idx in range(len(core_points)))
        if not tail_labels and len(tail_points) > 0:
            tail_labels = tuple(f"tail-{idx}" for idx in range(len(tail_points)))

        return core_points, core_labels, tail_points, tail_labels

    def _order_tail_sequence(
        self,
        base_point: np.ndarray,
        tail_points: np.ndarray,
        tail_labels: Tuple[str, ...],
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        if tail_points.size == 0:
            return tail_points.astype(np.float32, copy=False), tail_labels

        points = tail_points.astype(np.float32, copy=False)
        if tail_labels:
            label_list = list(tail_labels)
        else:
            label_list = [f"tail-{idx}" for idx in range(len(points))]

        remaining = list(range(len(points)))
        ordered_indices: List[int] = []

        start_idx = min(remaining, key=lambda idx: float(np.linalg.norm(points[idx] - base_point)))
        ordered_indices.append(start_idx)
        remaining.remove(start_idx)

        while remaining:
            last_point = points[ordered_indices[-1]]
            next_idx = min(remaining, key=lambda idx: float(np.linalg.norm(points[idx] - last_point)))
            ordered_indices.append(next_idx)
            remaining.remove(next_idx)

        ordered_points = points[ordered_indices]
        ordered_labels = tuple(label_list[idx] for idx in ordered_indices)
        return ordered_points, ordered_labels

    def _build_tail_polyline(
        self,
        mouse_id: str,
        base_point: np.ndarray,
        tail_points: np.ndarray,
    ) -> Optional[np.ndarray]:
        if tail_points.size == 0:
            return None

        base_point_f32 = base_point.astype(np.float32)
        points = tail_points.astype(np.float32, copy=False)
        if np.linalg.norm(points[0] - base_point_f32) < 1e-6:
            chain = points
        else:
            chain = np.vstack((base_point_f32[None, :], points))

        if len(chain) <= 2:
            return chain

        samples_per_segment = max(10, min(36, len(chain) * 8))
        return self._cubic_bezier_chain(chain, samples_per_segment)

    def _draw_whiskers(
        self,
        anchor_point: np.ndarray,
        nose_point: np.ndarray,
        base_color: Tuple[float, float, float],
    ) -> None:
        direction = nose_point - anchor_point
        norm = float(np.linalg.norm(direction))
        if norm < 1e-5:
            direction_unit = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction_unit = direction / norm

        whisker_length = min(max(norm * 0.85, 18.0), 48.0)
        whisker_angles = (-0.55, -0.3, 0.3, 0.55)
        accent_color = self._lighten_color(base_color, 0.55)
        underline_color = self._lighten_color(base_color, 0.2)
        segments: List[Tuple[np.ndarray, np.ndarray]] = []
        for angle in whisker_angles:
            oriented = self._rotate_vector(direction_unit, angle)
            tip = nose_point + oriented * whisker_length
            segments.append((nose_point.astype(np.float32), tip.astype(np.float32)))

        if segments:
            self._scene_add_whiskers(
                segments,
                primary_color=accent_color,
                secondary_color=underline_color,
            )

    def _draw_mouse_group(
        self,
        mouse_id: str,
        group: MouseGroup,
        base_color: Tuple[float, float, float],
    ) -> None:
        points = group.points
        if len(points) == 0:
            return

        core_points, core_labels, tail_points, tail_labels = self._split_tail_points(group)
        body_points = core_points if len(core_points) > 0 else points
        anchor_point = body_points.mean(axis=0) if len(body_points) > 0 else points.mean(axis=0)

        hull = self._convex_hull(core_points) if len(core_points) >= 3 else None

        nose_point: Optional[np.ndarray] = None
        if group.labels:
            for idx, label in enumerate(group.labels):
                if "nose" in str(label).lower():
                    nose_point = points[idx]
                    break

        sorted_tail_points: Optional[np.ndarray] = None
        sorted_tail_labels: Optional[Tuple[str, ...]] = None
        connection_point = anchor_point
        if len(tail_points) > 0:
            sorted_tail_points, sorted_tail_labels = self._order_tail_sequence(anchor_point, tail_points, tail_labels)
            tail_start_index = 0
            if hull is not None and len(hull) >= 3:
                connection_point, tail_start_index = self._tail_base_connection(sorted_tail_points, hull)
            elif len(body_points) > 0:
                tail_start_index = self._nearest_tail_index_to_body(sorted_tail_points, body_points)
                nearest_body = body_points
                if len(nearest_body) > 0:
                    dists = np.linalg.norm(nearest_body - sorted_tail_points[tail_start_index], axis=1)
                    body_idx = int(np.argmin(dists))
                    connection_point = nearest_body[body_idx].astype(np.float32)
            if tail_start_index > 0 and sorted_tail_points is not None and len(sorted_tail_points) > 0:
                sorted_tail_points = np.concatenate(
                    (sorted_tail_points[tail_start_index:], sorted_tail_points[:tail_start_index]),
                    axis=0,
                )
                if sorted_tail_labels is not None:
                    labels_list = list(sorted_tail_labels)
                    rotated = labels_list[tail_start_index:] + labels_list[:tail_start_index]
                    sorted_tail_labels = tuple(rotated)

        glow_points = points.astype(np.float32)
        self._scene_add_glow(glow_points, color=base_color)

        if len(body_points) > 0:
            self._scene_add_body(
                body_points.astype(np.float32),
                base_color=base_color,
                labels=core_labels,
                mouse_id=mouse_id,
                edge_color=(1.0, 1.0, 1.0),
            )

        label_color = self._lighten_color(base_color, 0.35)
        border_color = self._lighten_color(base_color, 0.15)
        self._scene_add_label(
            f"Mouse {mouse_id}",
            anchor_point.astype(np.float32),
            glow_points,
            color=label_color,
            border_color=border_color,
        )

        if nose_point is not None:
            self._draw_whiskers(anchor_point, nose_point, base_color)

        if sorted_tail_points is not None and len(sorted_tail_points) > 0:
            if sorted_tail_labels is None:
                sorted_tail_labels = tuple(f"tail-{idx}" for idx in range(len(sorted_tail_points)))
            tail_face = self._lighten_color(base_color, 0.35)
            self._scene_add_tail(
                sorted_tail_points.astype(np.float32),
                base_color=tail_face,
                labels=sorted_tail_labels,
                mouse_id=mouse_id,
                edge_color=base_color,
            )
            polyline = self._build_tail_polyline(mouse_id, connection_point, sorted_tail_points)
            if polyline is not None and len(polyline) >= 2:
                self._scene_add_tail_polyline(
                    polyline.astype(np.float32),
                    primary_color=self._lighten_color(base_color, 0.35),
                    secondary_color=base_color,
                )

        cluster_for_edges = core_points
        if len(cluster_for_edges) >= 2:
            edges = self._minimum_spanning_edges(cluster_for_edges)
            segments = [
                (
                    np.asarray([x1, y1], dtype=np.float32),
                    np.asarray([x2, y2], dtype=np.float32),
                )
                for (x1, y1), (x2, y2) in edges
            ]
            if segments:
                edge_color = self._lighten_color(base_color, 0.05)
                self._scene_add_edges(segments, color=edge_color)

        if hull is not None and len(hull) >= 3:
            fill_color = self._lighten_color(base_color, 0.45)
            self._scene_add_hull(hull.astype(np.float32), color=fill_color)

    def _draw_trail(self, mouse_id: str, base_color: Tuple[float, float, float]) -> None:
        trail_points = self.trail_cache.get(mouse_id)
        if not trail_points or len(trail_points) < 2:
            return

        segment_count = min(len(trail_points), self.trail_visual_length)
        recent_points = trail_points[-segment_count:]
        if len(recent_points) < 2:
            return

        alphas = np.linspace(0.12, 0.48, len(recent_points))
        lightened = self._lighten_color(base_color, 0.12)
        segments = []
        for (x1, y1), (x2, y2), alpha in zip(recent_points[:-1], recent_points[1:], alphas[1:]):
            start = np.asarray([x1, y1], dtype=np.float32)
            end = np.asarray([x2, y2], dtype=np.float32)
            segments.append((start, end, float(alpha * 0.7)))

        if segments:
            self._scene_add_trail(segments, color=lightened)


__all__ = ["PoseViewerPlaybackMixin"]
