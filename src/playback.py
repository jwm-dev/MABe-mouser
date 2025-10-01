"""Playback, rendering, and interaction mixin for the pose viewer application."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import time

import numpy as np

from .geometry import PoseViewerGeometryMixin
from .models import FramePayload, MouseGroup
from .plotting import SceneRect, get_palette_color
from .render import draw_mouse_group


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

    def _toggle_labels(self) -> None:
        scene = getattr(self, "scene", None)
        if scene is None:
            print("[Playback] toggle_labels shortcut ignored (no scene)")
            return
        toggle = getattr(scene, "toggle_labels_visible", None)
        if not callable(toggle):
            print("[Playback] toggle_labels shortcut ignored (no toggle method)")
            return
        toggle()

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
            frame_label_value=frame_payload.frame_number,
            frame_title=f"Frame {frame_payload.frame_number}",
            behaviors=frame_payload.behaviors,
        )

    def _render_pose_state(
        self,
        mouse_groups: Dict[str, MouseGroup],
        *,
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
        metadata_mapping: Mapping[str, Any] = metadata if isinstance(metadata, Mapping) else {}
        video_size = None
        arena_size = None
        video_dims = self._metadata_video_size_px(metadata_mapping) if metadata_mapping else (None, None)
        arena_dims = self._metadata_arena_size_px(metadata_mapping) if metadata_mapping else (None, None)
        if all(value is not None for value in video_dims):
            video_size = (float(video_dims[0]), float(video_dims[1]))  # type: ignore[index]
        if all(value is not None for value in arena_dims):
            arena_size = (float(arena_dims[0]), float(arena_dims[1]))  # type: ignore[index]
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

        cm_per_pixel = self._resolve_cm_per_pixel(metadata_mapping, domain_xlim, domain_ylim)
        self._cm_per_pixel = cm_per_pixel
        self._unit_label = "pixels"
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
        width_hint = None
        height_hint = None
        if video_size and len(video_size) >= 2:
            if math.isfinite(float(video_size[0])):
                width_hint = int(round(float(video_size[0])))
            if math.isfinite(float(video_size[1])):
                height_hint = int(round(float(video_size[1])))
        x_label_suffix = f"pixels, video width {width_hint}px" if width_hint is not None else "pixels"
        y_label_suffix = f"pixels, video height {height_hint}px" if height_hint is not None else "pixels"
        self._update_canvas_labels(
            xlabel=f"X ({x_label_suffix})",
            ylabel=f"Y ({y_label_suffix})",
            title=frame_title,
        )

        self._ensure_mouse_colors(mouse_groups.keys())

        for mouse_id, group in mouse_groups.items():
            color = self.mouse_colors[mouse_id]
            draw_mouse_group(self, mouse_id, group, color)

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


__all__ = ["PoseViewerPlaybackMixin"]
