"""Camera-related mixins for Vispy scene controllers."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Optional, Tuple, Union

import numpy as np
from vispy import scene

from .scene_types import SceneRect


class PoseSceneCameraMixin:
	"""Mixin providing camera setup, event handling and rect utilities."""

	def _initialise_camera(self, *, width: float, height: float) -> None:
		"""Initialise the interactive PanZoom camera and related state."""
		self._device_pixel_ratio = 1.0
		self._viewport_size = (float(width), float(height))
		self._qt_resize_watcher: Optional[Any] = None
		self._in_view_resize = False
		self._in_camera_update = False
		self._user_camera_override = False
		self._override_view_rect: Optional[SceneRect] = None
		self._pending_user_camera_override = False
		self._camera_change_callback: Optional[Callable[[SceneRect, bool, str], None]] = None

		camera = scene.cameras.PanZoomCamera(aspect=1.0, rect=(0.0, 0.0, 1.0, 1.0))
		camera.interactive = True
		camera.keep_aspect = True
		self.view.camera = camera
		self.view.camera.set_range(x=(0.0, 1.0), y=(0.0, 1.0), margin=0.0)
		self.view.camera.aspect = 1.0
		self.view.camera.flip = (False, False, False)

		self.view.scene.transform.changed.connect(self._on_scene_transform_change)
		self.view.camera.events.transform_change.connect(self._on_camera_transform_change)
		self.view.camera.events.mouse_wheel.connect(self._on_camera_interaction)
		self.view.camera.events.mouse_move.connect(self._on_camera_interaction)
		self.view.camera.events.mouse_press.connect(self._on_camera_interaction)
		self.view.camera.events.mouse_release.connect(self._on_camera_interaction)
		self.view.events.mouse_wheel.connect(self._on_view_mouse_wheel)
		self.view.events.mouse_press.connect(self._on_view_mouse_event)
		self.view.events.mouse_release.connect(self._on_view_mouse_event)
		self.view.events.mouse_move.connect(self._on_view_mouse_event)
		self.view.events.resize.connect(self._on_view_resize)
		self.canvas.events.resize.connect(self._on_resize)
		self.canvas.events.mouse_wheel.connect(self._on_canvas_mouse_wheel)
		self.canvas.events.mouse_move.connect(self._on_mouse_move)

	def has_user_camera_override(self) -> bool:
		value = bool(self._user_camera_override)
		print(f"[PoseScene] has_user_camera_override -> {value}")
		return value

	def on_camera_change(self, callback: Optional[Callable[[SceneRect, bool, str], None]]) -> None:
		self._camera_change_callback = callback

	def _emit_camera_change(self, *, source: str = "system") -> None:
		if self._camera_change_callback and self._current_view_rect is not None:
			callback = self._camera_change_callback
			try:
				callback(self._current_view_rect, bool(self._user_camera_override), source)
			except TypeError:
				callback(self._current_view_rect, bool(self._user_camera_override))

	def get_camera_override_rect(self) -> Optional[SceneRect]:
		return self._override_view_rect

	def get_current_camera_rect(self) -> Optional[SceneRect]:
		return self._current_view_rect

	def apply_camera_rect(self, rect: SceneRect, *, as_user_override: bool = True) -> SceneRect:
		print(f"[PoseScene] apply_camera_rect rect={rect} as_user_override={as_user_override}")
		bounds = self._scene_rect or rect
		clamped = self._clamp_rect_to_bounds(rect, bounds)
		self._pending_user_camera_override = False
		self._user_camera_override = bool(as_user_override)
		self._override_view_rect = clamped if as_user_override else None
		self._set_camera_rect(clamped, flip_x=self._x_axis_flipped, flip_y=self._y_axis_flipped)
		return clamped

	def reset_camera_view(self) -> None:
		print("[PoseScene] reset_camera_view called; clearing overrides")
		self._pending_user_camera_override = False
		self._user_camera_override = False
		self._override_view_rect = None
		self._apply_geometry()
		self.request_draw()

	def _set_camera_rect(
		self,
		rect: SceneRect,
		*,
		flip_x: bool,
		flip_y: bool,
		update_override: bool = False,
	) -> None:
		print(
			f"[PoseScene] _set_camera_rect rect={rect} flip=({flip_x},{flip_y}) update_override={update_override}"
		)
		camera = getattr(self.view, "camera", None)
		if camera is None:
			return
		center_x, center_y = rect.center
		x_min = rect.x
		y_min = rect.y
		x_max = rect.x + rect.width
		y_max = rect.y + rect.height
		self._in_camera_update = True
		try:
			camera.set_range(x=(x_min, x_max), y=(y_min, y_max), z=(0.0, 1.0), margin=0.0)
			camera.flip = (flip_x, flip_y, False)
			camera.center = (center_x, center_y, 0.0)
			try:
				camera.rect = (x_min, y_min, rect.width, rect.height)
			except AttributeError:
				pass
		finally:
			self._in_camera_update = False
		camera.aspect = float(rect.width / rect.height) if rect.height > 0.0 else None
		self._current_view_rect = rect
		if update_override:
			self._override_view_rect = rect
		self._update_axes_for_rect(rect)
		self._update_frame_border()
		try:
			transform = self.view.scene.transform
			origin = transform.map([x_min, y_min, 0.0])
			x_axis_point = transform.map([x_max, y_min, 0.0])
			y_axis_point = transform.map([x_min, y_max, 0.0])
			x_pixels = float(np.linalg.norm(x_axis_point[:2] - origin[:2]))
			y_pixels = float(np.linalg.norm(y_axis_point[:2] - origin[:2]))
			view_size = getattr(self.view, "size", None)
			print(
				f"[PoseScene] camera rect=({x_min:.1f},{x_max:.1f})×({y_min:.1f},{y_max:.1f}) size=({rect.width:.1f}×{rect.height:.1f}) flip=({flip_x},{flip_y}) px=({x_pixels:.1f}×{y_pixels:.1f}) view_size={view_size}"
			)
		except Exception:
			pass
		self._refresh_hover_cache()
		if not self._in_layout_update:
			self._update_layout_for_dimensions()
		self._emit_camera_change(source="system")
		print(
			f"[PoseScene] _set_camera_rect done current_rect={self._current_view_rect} override={self._override_view_rect}"
		)

	def _handle_native_resize(self, width: Union[int, float], height: Union[int, float], device_pixel_ratio: float) -> None:
		logical_width = max(float(width), 1.0)
		logical_height = max(float(height), 1.0)
		self._device_pixel_ratio = max(float(device_pixel_ratio), 1.0)
		self._viewport_size = (logical_width, logical_height)
		self._update_reset_button_geometry(logical_width, logical_height)
		self._update_square_layout()
		self._apply_geometry()
		self._refresh_hover_cache()

	def _on_canvas_mouse_wheel(self, event: Any) -> None:
		delta = getattr(event, "delta", None)
		print(f"[PoseScene] _on_canvas_mouse_wheel delta={delta}")

	def _forward_event_to_camera(self, event: Any) -> bool:
		camera = getattr(self.view, "camera", None)
		if camera is None:
			return False
		handler = getattr(camera, "viewbox_mouse_event", None)
		if handler is None:
			return False
		try:
			handler(event)
			self._on_camera_transform_change(event)
			return True
		except Exception as exc:  # pragma: no cover - diagnostic path
			print(f"[PoseScene] _forward_event_to_camera error={exc!r}")
			return False

	def _on_view_mouse_wheel(self, event: Any) -> None:
		delta = getattr(event, "delta", None)
		info = self._describe_mouse_event(event)
		print(f"[PoseScene] _on_view_mouse_wheel delta={delta} {info}")
		forwarded = self._forward_event_to_camera(event)
		if forwarded:
			updated = self._describe_mouse_event(event)
			print(f"[PoseScene] _on_view_mouse_wheel forwarded -> {updated}")

	def _on_view_mouse_event(self, event: Any) -> None:
		info = self._describe_mouse_event(event)
		print(f"[PoseScene] _on_view_mouse_event {info}")
		if self._handle_scale_bar_mouse_event(event):
			return
		forwarded = self._forward_event_to_camera(event)
		if forwarded:
			updated = self._describe_mouse_event(event)
			print(f"[PoseScene] _on_view_mouse_event forwarded -> {updated}")

	def _on_view_resize(self, event: Any) -> None:
		if self._in_view_resize:
			return
		self._in_view_resize = True
		try:
			self._update_square_layout()
			self._apply_geometry()
		finally:
			self._in_view_resize = False

	def _on_resize(self, event: Any) -> None:
		size = getattr(event, "size", None)
		if size is not None:
			width, height = size
			self._handle_native_resize(width, height, self._device_pixel_ratio)
		self.request_draw()

	def _on_camera_interaction(self, event: Optional[Any]) -> None:
		event_name = None
		if event is not None:
			event_name = getattr(event, "type", None) or getattr(event, "name", None)
			if event_name is None:
				event_name = type(event).__name__
		else:
			event_name = "None"
		print(f"[PoseScene] _on_camera_interaction event={event_name}")
		self._on_camera_transform_change(event)

	def _on_camera_transform_change(self, event: Optional[Any] = None) -> None:
		print("[PoseScene] _on_camera_transform_change triggered")
		if self._in_camera_update:
			print("[PoseScene] _on_camera_transform_change ignored (in_camera_update)")
			return
		camera = getattr(self.view, "camera", None)
		if camera is None:
			print("[PoseScene] _on_camera_transform_change abort: camera missing")
			return
		base_rect = self._base_view_rect or self._scene_rect
		if base_rect is None:
			print("[PoseScene] _on_camera_transform_change abort: base_rect missing")
			return

		def _extract_rect(cam: Any) -> Optional[Tuple[float, float, float, float]]:
			rect_val = getattr(cam, "rect", None)
			if rect_val is not None:
				try:
					left = getattr(rect_val, "left", None)
					bottom = getattr(rect_val, "bottom", None)
					width = getattr(rect_val, "width", None)
					height = getattr(rect_val, "height", None)
					if hasattr(rect_val, "pos"):
						pos = rect_val.pos
					else:
						pos = None
					if hasattr(rect_val, "size"):
						size = rect_val.size
					else:
						size = None
					x_val = float(left if left is not None else (pos[0] if pos is not None and len(pos) >= 2 else 0.0))
					y_val = float(bottom if bottom is not None else (pos[1] if pos is not None and len(pos) >= 2 else 0.0))
					w_val = float(width if width is not None else (size[0] if size is not None and len(size) >= 2 else 0.0))
					h_val = float(height if height is not None else (size[1] if size is not None and len(size) >= 2 else 0.0))
					if math.isfinite(w_val) and math.isfinite(h_val) and w_val > 0.0 and h_val > 0.0:
						return (x_val, y_val, w_val, h_val)
				except Exception:
					try:
						sequence = tuple(float(rect_val[idx]) for idx in range(4))
						if all(math.isfinite(v) for v in sequence[2:]):
							return sequence
					except Exception:
						pass
			try:
				state = cam.get_state() if hasattr(cam, "get_state") else None
			except Exception:
				state = None
			if isinstance(state, Mapping):
				rect_state = state.get("rect")
				if rect_state is not None:
					try:
						if len(rect_state) >= 4:
							vals = tuple(float(rect_state[i]) for i in range(4))
							if all(math.isfinite(v) for v in vals[2:]):
								return vals
					except Exception:
						pass
			try:
				ranges = cam.get_range()
			except Exception:
				ranges = None
			if isinstance(ranges, Mapping) and "x" in ranges and "y" in ranges:
				try:
					x0, x1 = ranges["x"]
					y0, y1 = ranges["y"]
					return (
						float(x0),
						float(y0),
						float(x1) - float(x0),
						float(y1) - float(y0),
					)
				except Exception:
					return None
			return None

		rect = _extract_rect(camera)
		if rect is None:
			print("[PoseScene] _on_camera_transform_change abort: camera rect unavailable")
			return
		print(f"[PoseScene] _on_camera_transform_change camera_rect={rect} base_rect={base_rect}")

		x, y, width, height = rect
		width = max(float(width), 1e-9)
		height = max(float(height), 1e-9)
		cx = float(x) + width * 0.5
		cy = float(y) + height * 0.5

		base_width = max(base_rect.width, 1e-9)
		base_height = max(base_rect.height, 1e-9)
		updated = False
		if width > base_width + 1e-6 or height > base_height + 1e-6:
			width = base_width
			height = base_height
			updated = True

		half_w = width * 0.5
		half_h = height * 0.5
		base_x0 = base_rect.x
		base_y0 = base_rect.y
		base_x1 = base_rect.x + base_rect.width
		base_y1 = base_rect.y + base_rect.height

		min_cx = base_x0 + half_w
		max_cx = base_x1 - half_w
		if max_cx >= min_cx:
			clamped_cx = min(max(cx, min_cx), max_cx)
		else:
			clamped_cx = (base_x0 + base_x1) * 0.5
		if abs(clamped_cx - cx) > 1e-6:
			updated = True
		cx = clamped_cx

		min_cy = base_y0 + half_h
		max_cy = base_y1 - half_h
		if max_cy >= min_cy:
			clamped_cy = min(max(cy, min_cy), max_cy)
		else:
			clamped_cy = (base_y0 + base_y1) * 0.5
		if abs(clamped_cy - cy) > 1e-6:
			updated = True
		cy = clamped_cy

		new_x = cx - half_w
		new_y = cy - half_h
		if abs(new_x - x) > 1e-6 or abs(new_y - y) > 1e-6:
			updated = True

		self._current_view_rect = SceneRect(new_x, new_y, width, height)
		is_override = (
			abs(new_x - base_rect.x) > 1e-6
			or abs(new_y - base_rect.y) > 1e-6
			or width < base_width - 1e-6
			or height < base_height - 1e-6
		)
		previous_override = self._override_view_rect
		if not self._in_camera_update:
			if is_override:
				self._override_view_rect = self._current_view_rect
				if previous_override != self._override_view_rect:
					print(f"[PoseScene] user override stored rect={self._override_view_rect}")
			else:
				if self._override_view_rect is not None:
					print("[PoseScene] user override cleared (reverted to base rect)")
				self._override_view_rect = None
		self._user_camera_override = self._override_view_rect is not None
		print(
			f"[PoseScene] _on_camera_transform_change new_rect={self._current_view_rect} is_override={is_override} override_now={self._override_view_rect}"
		)

		self._update_axes_for_rect(self._current_view_rect)
		if not updated:
			self.show_scale_bar_hint()
			layout_request = getattr(self, "_request_label_layout", None)
			if callable(layout_request):
				layout_request()
			else:
				self._label_layout_dirty = True
				self._layout_labels()
			self._refresh_hover_cache()
			self.request_draw()
			self._emit_camera_change(source="user")
			return

		self._in_camera_update = True
		try:
			camera.set_range(x=(new_x, new_x + width), y=(new_y, new_y + height), margin=0.0)
			camera.center = (cx, cy, 0.0)
			try:
				camera.rect = (new_x, new_y, width, height)
			except AttributeError:
				pass
		finally:
			self._in_camera_update = False
		self._update_axes_for_rect(self._current_view_rect)
		self.show_scale_bar_hint()
		layout_request = getattr(self, "_request_label_layout", None)
		if callable(layout_request):
			layout_request()
		else:
			self._label_layout_dirty = True
			self._layout_labels()
		self._refresh_hover_cache()
		self.request_draw()
		self._emit_camera_change(source="user")
		print(
			"[PoseScene] _on_camera_transform_change applied rect={self._current_view_rect} override={self._override_view_rect}"
		)


__all__ = ["PoseSceneCameraMixin"]
