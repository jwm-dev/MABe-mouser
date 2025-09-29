"""Vispy-based plotting utilities for the pose viewer application."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from vispy import app, scene
from vispy.geometry import PolygonData
from vispy.scene import visuals
from vispy.visuals.axis import MaxNLocator
from vispy.visuals.transforms import STTransform

try:  # pragma: no cover - GUI unavailable
	from PyQt6 import QtCore, QtWidgets  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - PyQt not installed
	QtCore = None
	QtWidgets = None

from .constants import UI_ACCENT, UI_BACKGROUND, UI_SURFACE, UI_TEXT_MUTED, UI_TEXT_PRIMARY
from .optional_dependencies import qtawesome


app.use_app("pyqt6")


TAB20_RGB: Tuple[Tuple[float, float, float], ...] = (
	(0.121568627, 0.466666667, 0.705882353),
	(0.682352941, 0.780392157, 0.909803922),
	(1.0, 0.498039216, 0.054901961),
	(1.0, 0.733333333, 0.470588235),
	(0.17254902, 0.62745098, 0.17254902),
	(0.596078431, 0.874509804, 0.541176471),
	(0.839215686, 0.152941176, 0.156862745),
	(1.0, 0.596078431, 0.588235294),
	(0.580392157, 0.403921569, 0.741176471),
	(0.77254902, 0.690196078, 0.835294118),
	(0.549019608, 0.337254902, 0.294117647),
	(0.768627451, 0.611764706, 0.580392157),
	(0.890196078, 0.466666667, 0.760784314),
	(0.968627451, 0.71372549, 0.823529412),
	(0.498039216, 0.498039216, 0.498039216),
	(0.780392157, 0.780392157, 0.780392157),
	(0.737254902, 0.741176471, 0.133333333),
	(0.858823529, 0.858823529, 0.552941176),
	(0.090196078, 0.745098039, 0.811764706),
	(0.619607843, 0.854901961, 0.898039216),
)


GOLDEN_ANGLE_RADIANS = math.radians(137.50776405003785)
LABEL_MAX_ATTEMPTS = 18
LABEL_RADIUS_SCALES: Tuple[float, ...] = (1.0, 1.25, 1.55, 1.9, 2.45)
LABEL_OVERLAP_PADDING = 6.0
LABEL_TEXT_LINE_HEIGHT = 1.32
LABEL_TEXT_WIDTH_FACTOR = 0.62
LABEL_TEXT_EXTRA_CHARS = 1.4


ColorInput = Union[str, Sequence[float], Sequence[int], np.ndarray]

_COLOR_NAME_MAP: Dict[str, Tuple[float, float, float]] = {
	"black": (0.0, 0.0, 0.0),
	"white": (1.0, 1.0, 1.0),
	"red": (1.0, 0.0, 0.0),
	"green": (0.0, 0.5, 0.0),
	"blue": (0.0, 0.0, 1.0),
	"yellow": (1.0, 1.0, 0.0),
	"cyan": (0.0, 1.0, 1.0),
	"magenta": (1.0, 0.0, 1.0),
	"orange": (1.0, 0.55, 0.0),
	"purple": (0.5, 0.0, 0.5),
	"gray": (0.5, 0.5, 0.5),
	"grey": (0.5, 0.5, 0.5),
}


def _normalize_channel(value: Union[float, int]) -> float:
	val = float(value)
	if val > 1.0:
		val /= 255.0
	return max(0.0, min(1.0, val))


def _parse_hex_color(text: str) -> Tuple[float, float, float, Optional[float]]:
	hex_digits = text.lstrip("#").strip()
	if len(hex_digits) in {3, 4}:
		hex_digits = "".join(ch * 2 for ch in hex_digits)
	if len(hex_digits) == 6:
		r = int(hex_digits[0:2], 16)
		g = int(hex_digits[2:4], 16)
		b = int(hex_digits[4:6], 16)
		return (
			_normalize_channel(r),
			_normalize_channel(g),
			_normalize_channel(b),
			None,
		)
	if len(hex_digits) == 8:
		r = int(hex_digits[0:2], 16)
		g = int(hex_digits[2:4], 16)
		b = int(hex_digits[4:6], 16)
		a = int(hex_digits[6:8], 16)
		return (
			_normalize_channel(r),
			_normalize_channel(g),
			_normalize_channel(b),
			_normalize_channel(a),
		)
	raise ValueError(f"Unsupported hex color '{text}'")


def to_rgb(color: ColorInput) -> Tuple[float, float, float]:
	if isinstance(color, str):
		text = color.strip().lower()
		if text in {"none", "transparent"}:
			return (0.0, 0.0, 0.0)
		if text.startswith("#"):
			r, g, b, _ = _parse_hex_color(text)
			return (r, g, b)
		if text in _COLOR_NAME_MAP:
			return _COLOR_NAME_MAP[text]
		raise ValueError(f"Unrecognised color string '{color}'")

	if isinstance(color, np.ndarray):
		flat = color.flatten()
	else:
		flat = list(color)  # type: ignore[arg-type]

	if len(flat) == 4:
		flat = flat[:3]
	if len(flat) != 3:
		raise ValueError(f"Cannot convert value '{color}' to RGB")

	r, g, b = (_normalize_channel(chan) for chan in flat)
	return (float(r), float(g), float(b))


def to_rgba(color: ColorInput, *, alpha: Optional[float] = None) -> Tuple[float, float, float, float]:
	if isinstance(color, str):
		text = color.strip().lower()
		if text in {"none", "transparent"}:
			return (0.0, 0.0, 0.0, 0.0 if alpha is None else _normalize_channel(alpha))
		if text.startswith("#"):
			r, g, b, parsed_alpha = _parse_hex_color(text)
			return (
				r,
				g,
				b,
				float(_normalize_channel(parsed_alpha if parsed_alpha is not None else (1.0 if alpha is None else alpha))),
			)
		if text in _COLOR_NAME_MAP:
			r, g, b = _COLOR_NAME_MAP[text]
			return (r, g, b, float(_normalize_channel(1.0 if alpha is None else alpha)))
		raise ValueError(f"Unrecognised color string '{color}'")

	if isinstance(color, np.ndarray):
		flat = color.flatten()
	else:
		flat = list(color)  # type: ignore[arg-type]

	if len(flat) == 4:
		r, g, b, a = flat
		return (
			float(_normalize_channel(r)),
			float(_normalize_channel(g)),
			float(_normalize_channel(b)),
			float(_normalize_channel(alpha if alpha is not None else a)),
		)
	if len(flat) == 3:
		r, g, b = flat
		return (
			float(_normalize_channel(r)),
			float(_normalize_channel(g)),
			float(_normalize_channel(b)),
			float(_normalize_channel(1.0 if alpha is None else alpha)),
		)
	raise ValueError(f"Cannot convert value '{color}' to RGBA")


def _ensure_3d(points: np.ndarray) -> np.ndarray:
	arr = np.asarray(points, dtype=np.float32)
	if arr.ndim == 1:
		arr = arr.reshape(1, -1)
	if arr.shape[1] == 2:
		zeros = np.zeros((arr.shape[0], 1), dtype=np.float32)
		arr = np.concatenate((arr.astype(np.float32, copy=False), zeros), axis=1)
	elif arr.shape[1] == 1:
		zeros = np.zeros((arr.shape[0], 2), dtype=np.float32)
		arr = np.concatenate((arr.astype(np.float32, copy=False), zeros), axis=1)
	elif arr.shape[1] > 3:
		arr = arr[:, :3]
	return arr.astype(np.float32, copy=False)


def _rgba(color: ColorInput, alpha: Optional[float] = None) -> np.ndarray:
	return np.array(to_rgba(color, alpha=alpha), dtype=np.float32)


@dataclass
class HoverDataset:
	positions: np.ndarray
	labels: Sequence[str]
	mouse_id: str
	screen_positions: Optional[np.ndarray] = None


@dataclass
class LabelDefinition:
	text: str
	anchor: np.ndarray
	points: np.ndarray
	color: Tuple[float, float, float]
	border_color: Tuple[float, float, float]


@dataclass(frozen=True)
class SceneRect:
	x: float
	y: float
	width: float
	height: float

	@staticmethod
	def from_limits(xlim: Tuple[float, float], ylim: Tuple[float, float]) -> "SceneRect":
		x0, x1 = (float(xlim[0]), float(xlim[1]))
		y0, y1 = (float(ylim[0]), float(ylim[1]))
		x_min = min(x0, x1)
		y_min = min(y0, y1)
		return SceneRect(
			x=x_min,
			y=y_min,
			width=max(abs(x1 - x0), 0.0),
			height=max(abs(y1 - y0), 0.0),
		)

	@property
	def xlim(self) -> Tuple[float, float]:
		return (self.x, self.x + self.width)

	@property
	def ylim(self) -> Tuple[float, float]:
		return (self.y, self.y + self.height)

	@property
	def center(self) -> Tuple[float, float]:
		return (self.x + self.width * 0.5, self.y + self.height * 0.5)

	@property
	def aspect(self) -> float:
		height = max(self.height, 1e-9)
		return float(self.width / height)

class PoseScene:
	def __init__(
		self,
		*,
		size: Tuple[int, int] = (820, 820),
		dpi: int = 110,
		antialias: int = 4,
		background_color: ColorInput = UI_SURFACE,
	) -> None:
		"""Initialise the Vispy scene and visuals used by the pose viewer."""
		canvas_bg = to_rgba(UI_BACKGROUND)
		view_bg = to_rgba(background_color)
		width = int(max(size[0], 1))
		height = int(max(size[1], 1))
		canvas_kwargs: Dict[str, Any] = {
			"keys": None,
			"size": (width, height),
			"dpi": dpi,
			"bgcolor": canvas_bg,
			"show": False,
			"vsync": True,
			"resizable": True,
		}
		if antialias > 0:
			canvas_kwargs["samples"] = int(max(antialias, 0))
		try:
			self.canvas = scene.SceneCanvas(**canvas_kwargs)
		except TypeError:
			canvas_kwargs.pop("samples", None)
			self.canvas = scene.SceneCanvas(**canvas_kwargs)
		self.canvas.create_native()
		if hasattr(self.canvas.native, "setMinimumSize"):
			self.canvas.native.setMinimumSize(1, 1)
		self.canvas.bgcolor = canvas_bg

		self._grid = self.canvas.central_widget.add_grid(margin=0)
		self._grid.spacing = 0

		self.view = self._grid.add_view(row=0, col=0)
		self.view.border_color = None
		self.view.bgcolor = view_bg
		self.view.padding = 0
		self.view.stretch = (1, 1)
		try:
			self.view.interactive = True
		except Exception:
			pass


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

		self.canvas.events.resize.connect(self._on_resize)
		self.canvas.events.mouse_move.connect(self._on_mouse_move)
		self.canvas.events.mouse_wheel.connect(self._on_canvas_mouse_wheel)
		self.view.events.resize.connect(self._on_view_resize)

		self._device_pixel_ratio = 1.0
		self._viewport_size = (float(width), float(height))
		self._qt_resize_watcher: Optional[Any] = None

		self._raw_xlim = (0.0, 1.0)
		self._raw_ylim = (0.0, 1.0)
		self._current_xlim = self._raw_xlim
		self._current_ylim = self._raw_ylim
		self._domain_xlim: Optional[Tuple[Optional[float], Optional[float]]] = None
		self._domain_ylim: Optional[Tuple[Optional[float], Optional[float]]] = None
		self._x_axis_flipped = False
		self._y_axis_flipped = False
		self._target_aspect_ratio: Optional[float] = None
		self._view_aspect_ratio: Optional[float] = None
		self._manual_aspect_ratio: Optional[float] = None
		self._data_aspect_ratio: Optional[float] = None
		self._data_rect: Optional[SceneRect] = None
		self._scene_rect: Optional[SceneRect] = None
		self._base_view_rect: Optional[SceneRect] = None
		self._current_view_rect: Optional[SceneRect] = None
		self._arena_rect: Optional[SceneRect] = None
		self._arena_size: Optional[Tuple[float, float]] = None
		self._video_size: Optional[Tuple[float, float]] = None
		self._hover_callback: Optional[Callable[[Optional[Dict[str, Any]]], None]] = None
		self._hover_datasets: List[HoverDataset] = []
		self._hover_threshold_px = 16.0
		self._in_layout_update = False
		self._in_view_resize = False
		self._in_camera_update = False
		self._user_camera_override = False
		self._override_view_rect: Optional[SceneRect] = None
		self._pending_user_camera_override = False
		self._camera_change_callback: Optional[Callable[[SceneRect, bool], None]] = None
		self._unit_scale = 1.0
		self._unit_label = "pixels"
		self._x_axis_label = ""
		self._y_axis_label = ""

		self._title_text = visuals.Text(
			"",
			pos=(0.5, 1.02, 0.0),
			color=UI_TEXT_PRIMARY,
			font_size=14,
			anchor_x="center",
			anchor_y="bottom",
			parent=self.view.scene,
		)
		self._title_text.visible = False

		self._hover_text = visuals.Text(
			"",
			color=_rgba(UI_ACCENT, 0.94),
			font_size=11,
			parent=self.view.scene,
			anchor_x="left",
			anchor_y="bottom",
		)
		self._hover_text.visible = False

		self._glow_markers = visuals.Markers(parent=self.view.scene)
		self._glow_markers.set_gl_state(
			blend=True,
			depth_test=False,
			blend_func=("src_alpha", "one"),
		)
		self._glow_markers.antialias = 4

		self._body_markers = visuals.Markers(parent=self.view.scene)
		self._body_markers.set_gl_state("translucent", depth_test=False)

		self._tail_markers = visuals.Markers(parent=self.view.scene)
		self._tail_markers.set_gl_state("translucent", depth_test=False)

		self._trail_lines = visuals.Line(parent=self.view.scene, connect="segments")
		self._trail_lines.set_gl_state("translucent", depth_test=False)

		self._edge_lines = visuals.Line(parent=self.view.scene, connect="segments")
		self._edge_lines.set_gl_state("translucent", depth_test=False)

		self._tail_core_lines = visuals.Line(parent=self.view.scene, connect="strip")
		self._tail_core_lines.set_gl_state("translucent", depth_test=False)

		self._tail_overlay_lines = visuals.Line(parent=self.view.scene, connect="strip")
		self._tail_overlay_lines.set_gl_state("translucent", depth_test=False)

		self._whisker_lines_primary = visuals.Line(parent=self.view.scene, connect="segments")
		self._whisker_lines_primary.set_gl_state("translucent", depth_test=False)

		self._whisker_lines_secondary = visuals.Line(parent=self.view.scene, connect="segments")
		self._whisker_lines_secondary.set_gl_state("translucent", depth_test=False)

		self._label_lines = visuals.Line(parent=self.view.scene, connect="segments")
		self._label_lines.set_gl_state("translucent", depth_test=False)

		self._frame_border = visuals.Line(parent=self.view.scene, connect="segments")
		self._frame_border.set_gl_state("translucent", depth_test=False)
		self._frame_border.visible = False

		self._grid_lines = visuals.Line(parent=self.view.scene, connect="strip")
		self._grid_lines.set_gl_state("translucent", depth_test=False)
		self._grid_lines.visible = False

		self._scale_bar_margin = 18.0
		self._scale_bar_line = visuals.Line(parent=self.canvas.scene, connect="segments")
		self._scale_bar_line.set_gl_state("translucent", depth_test=False)
		self._scale_bar_line.transform = STTransform()
		self._scale_bar_line.visible = False

		self._scale_bar_text = visuals.Text(
			"",
			color=_rgba(UI_TEXT_PRIMARY, 0.92),
			font_size=11,
			parent=self.canvas.scene,
			anchor_x="right",
			anchor_y="top",
		)
		self._scale_bar_text.transform = STTransform()
		self._scale_bar_text.visible = False
		self._scale_bar_alpha = 0.0
		self._scale_bar_idle_delay = 1.6
		self._scale_bar_fade_duration = 0.75
		self._scale_bar_line_points = np.zeros((0, 3), dtype=np.float32)
		self._scale_bar_line_connect = np.zeros((0, 2), dtype=np.uint32)
		self._scale_bar_line_width = 1.6
		self._scale_bar_geometry_ready = False
		self._scale_bar_fade_start = 0.0
		self._scale_bar_hold_timer = app.Timer(
			interval=self._scale_bar_idle_delay,
			connect=self._on_scale_bar_hold_timeout,
			start=False,
		)
		self._scale_bar_fade_timer = app.Timer(
			interval=1.0 / 60.0,
			connect=self._on_scale_bar_fade_step,
			start=False,
		)
		self._scale_bar_mode = "auto"
		self._scale_bar_hit_rect: Optional[Tuple[float, float, float, float]] = None
		self._scale_bar_click_armed = False

		self._reset_button: Optional[Any] = None
		self._reset_button_effect: Optional[Any] = None
		self._reset_button_callback: Optional[Callable[[], None]] = None
		self._reset_button_margin = 14.0
		self._initialise_reset_button()

		self._hull_polygons: List[visuals.Polygon] = []
		self._label_texts: List[visuals.Text] = []
		self._label_bounds: List[Tuple[float, float, float, float]] = []
		self._label_definitions: List[LabelDefinition] = []
		self._label_layout_dirty = False

		self._update_layout_for_dimensions()
		self.reset_frame_state()

	def reset_frame_state(self) -> None:
		self._glow_positions: List[np.ndarray] = []
		self._glow_sizes: List[np.ndarray] = []
		self._glow_colors: List[np.ndarray] = []

		self._body_positions: List[np.ndarray] = []
		self._body_sizes: List[np.ndarray] = []
		self._body_face_colors: List[np.ndarray] = []
		self._body_edge_colors: List[np.ndarray] = []
		self._body_edge_widths: List[np.ndarray] = []

		self._tail_positions: List[np.ndarray] = []
		self._tail_sizes: List[np.ndarray] = []
		self._tail_face_colors: List[np.ndarray] = []
		self._tail_edge_colors: List[np.ndarray] = []
		self._tail_edge_widths: List[np.ndarray] = []

		self._trail_segments: List[np.ndarray] = []
		self._trail_colors: List[np.ndarray] = []
		self._trail_widths: List[float] = []

		self._edge_segments: List[np.ndarray] = []
		self._edge_colors: List[np.ndarray] = []

		self._tail_core_points: List[np.ndarray] = []
		self._tail_core_colors: List[np.ndarray] = []
		self._tail_core_widths: List[float] = []

		self._tail_overlay_points: List[np.ndarray] = []
		self._tail_overlay_colors: List[np.ndarray] = []
		self._tail_overlay_widths: List[float] = []

		self._whisker_primary_segments: List[np.ndarray] = []
		self._whisker_primary_colors: List[np.ndarray] = []
		self._whisker_primary_widths: List[float] = []

		self._whisker_secondary_segments: List[np.ndarray] = []
		self._whisker_secondary_colors: List[np.ndarray] = []
		self._whisker_secondary_widths: List[float] = []

		self._label_line_segments: List[np.ndarray] = []
		self._label_line_colors: List[np.ndarray] = []
		self._label_bounds = []
		self._label_definitions = []
		self._label_layout_dirty = False

		self._hover_datasets = []

	def native_widget(self) -> Any:
		return self.canvas.native

	@property
	def unit_label(self) -> str:
		return self._unit_label

	@property
	def has_user_camera_override(self) -> bool:
		value = bool(self._user_camera_override)
		print(f"[PoseScene] has_user_camera_override -> {value}")
		return value

	def set_unit_scale(self, *, cm_per_pixel: Optional[float]) -> None:
		print(f"[PoseScene] set_unit_scale requested cm_per_pixel={cm_per_pixel}")
		current_scale = float(self._unit_scale)
		current_label = self._unit_label
		scale_val = None
		if cm_per_pixel is not None:
			try:
				scale_val = float(cm_per_pixel)
			except (TypeError, ValueError):
				scale_val = None
			if scale_val is not None and (not math.isfinite(scale_val) or scale_val <= 0.0):
				scale_val = None
		if scale_val is None:
			new_scale = 1.0
			new_label = "pixels"
		else:
			new_scale = float(scale_val)
			new_label = "cm"
		if math.isclose(new_scale, current_scale, rel_tol=1e-9, abs_tol=1e-9) and new_label == current_label:
			print("[PoseScene] set_unit_scale unchanged; skipping reset")
			self._unit_scale = current_scale
			self._unit_label = current_label
			self._update_scale_bar()
			return
		self._unit_scale = new_scale
		self._unit_label = new_label
		self._base_view_rect = None
		self._current_view_rect = None
		self._scene_rect = None
		self._user_camera_override = False
		self._override_view_rect = None
		print(f"[PoseScene] set_unit_scale applied scale={self._unit_scale} label={self._unit_label}; cleared overrides")
		self._update_scale_bar()

	def _to_scene_units_scalar(self, value: Optional[float]) -> Optional[float]:
		if value is None:
			return None
		return float(value) * self._unit_scale

	def _to_scene_units_pair(self, pair: Tuple[Optional[float], Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
		left = self._to_scene_units_scalar(pair[0])
		right = self._to_scene_units_scalar(pair[1])
		return (left, right)

	def _to_scene_units_array(self, array: np.ndarray) -> np.ndarray:
		if array.size == 0:
			return array.astype(np.float32, copy=False)
		if math.isclose(self._unit_scale, 1.0, rel_tol=1e-9, abs_tol=1e-9):
			return array.astype(np.float32, copy=False)
		scaled = array.astype(np.float32, copy=True)
		if scaled.ndim >= 1 and scaled.shape[-1] >= 1:
			scaled[..., 0] *= self._unit_scale
		if scaled.ndim >= 1 and scaled.shape[-1] >= 2:
			scaled[..., 1] *= self._unit_scale
		return scaled

	def on_hover(self, callback: Callable[[Optional[Dict[str, Any]]], None]) -> None:
		self._hover_callback = callback

	def set_limits(
		self,
		xlim: Tuple[float, float],
		ylim: Tuple[float, float],
		*,
		aspect_ratio: Optional[float] = None,
		domain_xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
		domain_ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
		preserve_view: bool = False,
	) -> None:
		print(
			f"[PoseScene] set_limits xlim={xlim} ylim={ylim} preserve_view={preserve_view} current_override={self._override_view_rect}"
		)
		if not preserve_view:
			self._user_camera_override = False
			self._override_view_rect = None
			print("[PoseScene] set_limits cleared overrides (preserve_view=False)")
		x0_raw, x1_raw = map(float, xlim)
		y0_raw, y1_raw = map(float, ylim)
		x0_opt = self._to_scene_units_scalar(x0_raw)
		x1_opt = self._to_scene_units_scalar(x1_raw)
		y0_opt = self._to_scene_units_scalar(y0_raw)
		y1_opt = self._to_scene_units_scalar(y1_raw)
		x0 = float(x0_opt if x0_opt is not None else x0_raw)
		x1 = float(x1_opt if x1_opt is not None else x1_raw)
		y0 = float(y0_opt if y0_opt is not None else y0_raw)
		y1 = float(y1_opt if y1_opt is not None else y1_raw)
		self._raw_xlim = (x0, x1)
		self._raw_ylim = (y0, y1)
		self._current_xlim = (x0, x1)
		self._current_ylim = (y0, y1)
		self._x_axis_flipped = x0 > x1
		self._y_axis_flipped = y0 > y1
		self._data_rect = SceneRect.from_limits((x0, x1), (y0, y1))
		if domain_xlim is not None:
			lower = None
			upper = None
			if len(domain_xlim) > 0 and domain_xlim[0] is not None:
				lower_opt = self._to_scene_units_scalar(float(domain_xlim[0]))
				lower = float(lower_opt) if lower_opt is not None else float(domain_xlim[0])
			if len(domain_xlim) > 1 and domain_xlim[1] is not None:
				upper_opt = self._to_scene_units_scalar(float(domain_xlim[1]))
				upper = float(upper_opt) if upper_opt is not None else float(domain_xlim[1])
			self._domain_xlim = (lower, upper) if (lower is not None or upper is not None) else None
		if domain_ylim is not None:
			lower = None
			upper = None
			if len(domain_ylim) > 0 and domain_ylim[0] is not None:
				lower_opt = self._to_scene_units_scalar(float(domain_ylim[0]))
				lower = float(lower_opt) if lower_opt is not None else float(domain_ylim[0])
			if len(domain_ylim) > 1 and domain_ylim[1] is not None:
				upper_opt = self._to_scene_units_scalar(float(domain_ylim[1]))
				upper = float(upper_opt) if upper_opt is not None else float(domain_ylim[1])
			self._domain_ylim = (lower, upper) if (lower is not None or upper is not None) else None
		width = float(abs(x1 - x0))
		height = float(abs(y1 - y0))
		if math.isfinite(width) and math.isfinite(height) and height > 1e-6:
			self._data_aspect_ratio = width / height
		else:
			self._data_aspect_ratio = None
		if aspect_ratio is not None:
			ratio = float(aspect_ratio)
			if math.isfinite(ratio) and ratio > 0.0:
				self._manual_aspect_ratio = ratio
			else:
				self._manual_aspect_ratio = None
		else:
			self._manual_aspect_ratio = None
		domain_width, domain_height = self._compute_domain_dimensions()
		self._target_aspect_ratio = self._resolve_aspect_ratio(domain_width=domain_width, domain_height=domain_height, data_rect=self._data_rect)
		self._apply_geometry()
		print(
			f"[PoseScene] set_limits completed; current_rect={self._current_view_rect} override={self._override_view_rect} user_override={self._user_camera_override}"
		)

	def set_axis_labels(self, *, xlabel: str, ylabel: str) -> None:
		self._x_axis_label = str(xlabel)
		self._y_axis_label = str(ylabel)

	def set_title(self, text: str) -> None:
		self._title_text.text = ""
		self._title_text.visible = False

	def set_scene_dimensions(
		self,
		*,
		video_size: Optional[Tuple[float, float]] = None,
		arena_size: Optional[Tuple[float, float]] = None,
	) -> None:
		changed = False
		if video_size and len(video_size) >= 2:
			vw = float(video_size[0])
			vh = float(video_size[1])
			if math.isfinite(vw) and math.isfinite(vh) and vw > 0.0 and vh > 0.0:
				converted_w = self._to_scene_units_scalar(vw)
				converted_h = self._to_scene_units_scalar(vh)
				if converted_w is None or converted_h is None:
					converted_w = float(vw)
					converted_h = float(vh)
				tuple_value = (float(converted_w), float(converted_h))
				if self._video_size != tuple_value:
					self._video_size = tuple_value
					changed = True
		if arena_size and len(arena_size) >= 2:
			aw = float(arena_size[0])
			ah = float(arena_size[1])
			if math.isfinite(aw) and math.isfinite(ah) and aw > 0.0 and ah > 0.0:
				converted_w = self._to_scene_units_scalar(aw)
				converted_h = self._to_scene_units_scalar(ah)
				if converted_w is None or converted_h is None:
					converted_w = float(aw)
					converted_h = float(ah)
				tuple_value = (float(converted_w), float(converted_h))
				if self._arena_size != tuple_value:
					self._arena_size = tuple_value
					self._arena_rect = None
					changed = True
		elif arena_size is None and self._arena_rect is not None:
			self._arena_rect = None
			self._arena_size = None
			changed = True
		if changed:
			print(f"[PoseScene] set_scene_dimensions video={self._video_size} arena={self._arena_size}")
			self._user_camera_override = False
			self._override_view_rect = None
			self._apply_geometry()
			self._update_layout_for_dimensions()

	@staticmethod
	def _size_to_pair(value: Any) -> Optional[Tuple[float, float]]:
		if value is None:
			return None
		if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 2:
			width = float(value[0])
			height = float(value[1])
			return (width, height)
		width_attr = getattr(value, "width", None)
		height_attr = getattr(value, "height", None)
		if width_attr is not None and height_attr is not None:
			try:
				return (float(width_attr), float(height_attr))
			except (TypeError, ValueError):
				return None
		return None

	@staticmethod
	def _safe_ratio(width: Optional[float], height: Optional[float]) -> Optional[float]:
		if width is None or height is None:
			return None
		try:
			w_val = float(width)
			h_val = float(height)
		except (TypeError, ValueError):
			return None
		if not math.isfinite(w_val) or not math.isfinite(h_val) or w_val <= 0.0 or h_val <= 0.0:
			return None
		return w_val / h_val

	def _compute_domain_dimensions(self) -> Tuple[Optional[float], Optional[float]]:
		domain_width: Optional[float] = None
		domain_height: Optional[float] = None
		if self._domain_xlim is not None:
			x_lower, x_upper = self._domain_xlim
			if x_lower is not None and x_upper is not None:
				try:
					width_val = float(x_upper) - float(x_lower)
				except (TypeError, ValueError):
					width_val = None
				if width_val is not None and math.isfinite(width_val) and width_val > 0.0:
					domain_width = width_val
		if self._domain_ylim is not None:
			y_lower, y_upper = self._domain_ylim
			if y_lower is not None and y_upper is not None:
				try:
					height_val = float(y_upper) - float(y_lower)
				except (TypeError, ValueError):
					height_val = None
				if height_val is not None and math.isfinite(height_val) and height_val > 0.0:
					domain_height = height_val
		return (domain_width, domain_height)

	def _resolve_aspect_ratio(
		self,
		*,
		domain_width: Optional[float] = None,
		domain_height: Optional[float] = None,
		data_rect: Optional[SceneRect] = None,
		arena_rect: Optional[SceneRect] = None,
	) -> float:
		video_ratio: Optional[float] = None
		if self._video_size is not None and len(self._video_size) >= 2:
			video_ratio = self._safe_ratio(self._video_size[0], self._video_size[1])
		domain_ratio = self._safe_ratio(domain_width, domain_height)
		arena_candidate = arena_rect if arena_rect is not None else self._arena_rect
		arena_ratio = arena_candidate.aspect if (arena_candidate is not None and arena_candidate.height > 0.0) else None
		data_ratio = self._data_aspect_ratio
		scene_ratio = data_rect.aspect if data_rect is not None and data_rect.height > 0.0 else None
		current_ratio = (
			self._target_aspect_ratio
			if (
				self._target_aspect_ratio is not None
				and math.isfinite(self._target_aspect_ratio)
				and self._target_aspect_ratio > 0.0
			)
			else None
		)
		for ratio in (
			self._manual_aspect_ratio,
			arena_ratio,
			domain_ratio,
			data_ratio,
			scene_ratio,
			current_ratio,
			video_ratio,
		):
			if ratio is not None and math.isfinite(ratio) and ratio > 0.0:
				return float(ratio)
		return 1.0

	def _effective_viewport_size(self) -> Optional[Tuple[float, float]]:
		view_size = self._size_to_pair(getattr(self.view, "size", None))
		if view_size is not None and view_size[0] > 1e-3 and view_size[1] > 1e-3:
			return (float(view_size[0]), float(view_size[1]))
		canvas_w, canvas_h = self._viewport_size
		if canvas_w > 1e-3 and canvas_h > 1e-3:
			return (float(canvas_w), float(canvas_h))
		return None

	def _update_layout_for_dimensions(self) -> None:
		if self._in_layout_update:
			return
		self._in_layout_update = True
		try:
			canvas_w, canvas_h = self._viewport_size
			canvas_w = max(float(canvas_w), 1.0)
			canvas_h = max(float(canvas_h), 1.0)
			try:
				self.view.size = (canvas_w, canvas_h)
			except Exception:
				pass
			native_widget = getattr(self.canvas, "native", None)
			if hasattr(native_widget, "updateGeometry"):
				native_widget.updateGeometry()
			self._view_aspect_ratio = self._safe_ratio(canvas_w, canvas_h)
			target = self._target_aspect_ratio if (self._target_aspect_ratio and math.isfinite(self._target_aspect_ratio) and self._target_aspect_ratio > 0.0) else (self._view_aspect_ratio or 1.0)
			self._letterbox_state = {
				"left": 0.0,
				"right": 0.0,
				"top": 0.0,
				"bottom": 0.0,
				"data_width": canvas_w,
				"data_height": canvas_h,
				"target_aspect": target,
			}
		finally:
			self._in_layout_update = False
		self._update_scale_bar()

	def _update_square_layout(self) -> None:
		"""Legacy resize hook that now proxies to the metadata-aware layout logic."""
		self._update_layout_for_dimensions()

	def _on_scene_transform_change(self, event: Optional[Any] = None) -> None:
		self._update_scale_bar()

	def _apply_geometry(self) -> None:
		print(
			"[PoseScene] _apply_geometry start"
			f" data_rect={self._data_rect} base_view={self._base_view_rect} override={self._override_view_rect}"
		)
		data_rect = self._data_rect

		dom_x_min = self._domain_xlim[0] if self._domain_xlim else None
		dom_x_max = self._domain_xlim[1] if self._domain_xlim else None
		dom_y_min = self._domain_ylim[0] if self._domain_ylim else None
		dom_y_max = self._domain_ylim[1] if self._domain_ylim else None
		arena_size = self._arena_size if self._arena_size is not None else None
		domain_rect: Optional[SceneRect] = None
		if (
			dom_x_min is not None
			and dom_x_max is not None
			and dom_y_min is not None
			and dom_y_max is not None
		):
			domain_rect = SceneRect.from_limits(
				(float(dom_x_min), float(dom_x_max)),
				(float(dom_y_min), float(dom_y_max)),
			)
		arena_rect_source = "none"
		arena_rect_candidate: Optional[SceneRect] = None
		if domain_rect is not None:
			arena_rect_candidate = domain_rect
			arena_rect_source = "domain"
		elif arena_size is not None:
			arena_w, arena_h = arena_size
			if dom_x_min is not None and dom_y_min is not None:
				arena_rect_candidate = SceneRect(float(dom_x_min), float(dom_y_min), arena_w, arena_h)
			elif data_rect is not None:
				center_x_data, center_y_data = data_rect.center
				arena_rect_candidate = SceneRect(
					center_x_data - arena_w * 0.5,
					center_y_data - arena_h * 0.5,
					arena_w,
					arena_h,
				)
			else:
				arena_rect_candidate = SceneRect(0.0, 0.0, arena_w, arena_h)
			arena_rect_source = "metadata"
		elif data_rect is not None:
			arena_rect_candidate = data_rect
			arena_rect_source = "data"
		if arena_rect_candidate is not None:
			if dom_x_min is None:
				dom_x_min = arena_rect_candidate.x
			if dom_x_max is None:
				dom_x_max = arena_rect_candidate.x + arena_rect_candidate.width
			if dom_y_min is None:
				dom_y_min = arena_rect_candidate.y
			if dom_y_max is None:
				dom_y_max = arena_rect_candidate.y + arena_rect_candidate.height

		domain_width: Optional[float] = None
		domain_height: Optional[float] = None
		if dom_x_min is not None and dom_x_max is not None:
			domain_width = max(float(dom_x_max) - float(dom_x_min), 0.0)
		if dom_y_min is not None and dom_y_max is not None:
			domain_height = max(float(dom_y_max) - float(dom_y_min), 0.0)

		target_ratio = self._resolve_aspect_ratio(
			domain_width=domain_width,
			domain_height=domain_height,
			data_rect=data_rect,
			arena_rect=arena_rect_candidate,
		)
		self._target_aspect_ratio = float(max(target_ratio, 1e-6))
		target_ratio = self._target_aspect_ratio

		if data_rect is None:
			if not self._in_layout_update:
				self._update_layout_for_dimensions()
			return

		data_width = max(data_rect.width, 1e-6)
		data_height = max(data_rect.height, 1e-6)
		arena_width = arena_rect_candidate.width if arena_rect_candidate is not None else None
		arena_height = arena_rect_candidate.height if arena_rect_candidate is not None else None

		min_width = data_width
		min_height = data_height
		if domain_width and domain_width > 0.0:
			min_width = max(min_width, domain_width)
		if domain_height and domain_height > 0.0:
			min_height = max(min_height, domain_height)
		if arena_width and arena_width > 0.0:
			min_width = max(min_width, arena_width)
		if arena_height and arena_height > 0.0:
			min_height = max(min_height, arena_height)

		width = max(min_width, min_height * target_ratio)
		height = width / target_ratio if target_ratio > 0.0 else min_height
		if height < min_height:
			height = min_height
			width = height * (target_ratio if target_ratio > 0.0 else 1.0)

		width = max(width, 1e-6)
		height = max(height, 1e-6)

		if arena_rect_candidate is not None:
			center_x, center_y = arena_rect_candidate.center
		else:
			center_x, center_y = data_rect.center
		if dom_x_min is not None and dom_x_max is not None:
			halve_w = width * 0.5
			min_cx = float(dom_x_min) + halve_w
			max_cx = float(dom_x_max) - halve_w
			if max_cx >= min_cx:
				center_x = min(max(center_x, min_cx), max_cx)
			else:
				center_x = (float(dom_x_min) + float(dom_x_max)) * 0.5
		if dom_y_min is not None and dom_y_max is not None:
			halve_h = height * 0.5
			min_cy = float(dom_y_min) + halve_h
			max_cy = float(dom_y_max) - halve_h
			if max_cy >= min_cy:
				center_y = min(max(center_y, min_cy), max_cy)
			else:
				center_y = (float(dom_y_min) + float(dom_y_max)) * 0.5
		if arena_rect_source == "domain" and arena_rect_candidate is not None:
			self._arena_rect = arena_rect_candidate
		elif arena_rect_source == "metadata" and arena_size is not None:
			arena_w, arena_h = arena_size
			self._arena_rect = SceneRect(
				center_x - arena_w * 0.5,
				center_y - arena_h * 0.5,
				arena_w,
				arena_h,
			)
		elif arena_rect_source == "data" and arena_rect_candidate is not None:
			self._arena_rect = SceneRect(
				center_x - arena_rect_candidate.width * 0.5,
				center_y - arena_rect_candidate.height * 0.5,
				arena_rect_candidate.width,
				arena_rect_candidate.height,
			)
		else:
			self._arena_rect = None

		x_min = center_x - width * 0.5
		x_max = center_x + width * 0.5
		y_min = center_y - height * 0.5
		y_max = center_y + height * 0.5

		default_rect = SceneRect(x_min, y_min, width, height)
		self._scene_rect = default_rect
		if (
			self._base_view_rect is None
			or abs(self._base_view_rect.x - default_rect.x) > 1e-6
			or abs(self._base_view_rect.y - default_rect.y) > 1e-6
			or abs(self._base_view_rect.width - default_rect.width) > 1e-6
			or abs(self._base_view_rect.height - default_rect.height) > 1e-6
		):
			self._base_view_rect = default_rect

		if self._override_view_rect is not None:
			active_rect = self._clamp_rect_to_bounds(self._override_view_rect, default_rect)
			self._override_view_rect = active_rect
		else:
			active_rect = default_rect
		self._current_view_rect = active_rect
		self._user_camera_override = self._override_view_rect is not None
		print(
			f"[PoseScene] _apply_geometry using active_rect={active_rect} user_override={self._user_camera_override}"
		)

		active_center_x, active_center_y = active_rect.center
		active_x_min = active_rect.x
		active_x_max = active_rect.x + active_rect.width
		active_y_min = active_rect.y
		active_y_max = active_rect.y + active_rect.height

		flip_x = self._x_axis_flipped
		flip_y = self._y_axis_flipped

		self._set_camera_rect(active_rect, flip_x=flip_x, flip_y=flip_y)
		print(
			f"[PoseScene] _apply_geometry completed; current_rect={self._current_view_rect} override={self._override_view_rect}"
		)

	def _update_frame_border(self) -> None:
		target_rect = self._arena_rect or self._scene_rect or self._current_view_rect
		if target_rect is None or target_rect.width <= 0.0 or target_rect.height <= 0.0:
			self._frame_border.visible = False
			self._frame_border.set_data(pos=np.zeros((0, 3), dtype=np.float32))
			return

		x_min = target_rect.x
		x_max = target_rect.x + target_rect.width
		y_min = target_rect.y
		y_max = target_rect.y + target_rect.height

		positions = np.array(
			[
				[x_min, y_max, 0.0],  # top-left
				[x_max, y_max, 0.0],  # top-right
				[x_max, y_min, 0.0],  # bottom-right
				[x_min, y_min, 0.0],  # bottom-left
			],
			dtype=np.float32,
		)
		axis_color = to_rgba(UI_TEXT_MUTED, alpha=0.8)
		color_rgba = np.tile(np.array(axis_color, dtype=np.float32), (positions.shape[0], 1))
		line_width = 1.2

		self._frame_border.visible = True
		indices = np.array(
			[
				[0, 1],
				[1, 2],
				[2, 3],
				[3, 0],
			],
			dtype=np.uint32,
		)
		self._frame_border.set_data(pos=_ensure_3d(positions), color=color_rgba, width=line_width, connect=indices)

	def _update_axes_for_rect(self, rect: Optional[SceneRect]) -> None:
		if rect is None:
			return
		x_min = rect.x
		x_max = rect.x + rect.width
		y_min = rect.y
		y_max = rect.y + rect.height
		flip_x = self._x_axis_flipped
		flip_y = self._y_axis_flipped
		x_pair = (x_max, x_min) if flip_x else (x_min, x_max)
		y_pair = (y_max, y_min) if flip_y else (y_min, y_max)
		self._current_xlim = x_pair
		self._current_ylim = y_pair
		self._update_grid_lines(rect)
		self._update_scale_bar()

	def _clamp_rect_to_bounds(self, rect: SceneRect, bounds: SceneRect) -> SceneRect:
		width = min(rect.width, bounds.width)
		height = min(rect.height, bounds.height)
		width = max(width, 1e-9)
		height = max(height, 1e-9)
		rect_center_x, rect_center_y = rect.center
		bounds_x0 = bounds.x
		bounds_y0 = bounds.y
		bounds_x1 = bounds.x + bounds.width
		bounds_y1 = bounds.y + bounds.height
		half_w = width * 0.5
		half_h = height * 0.5
		min_cx = bounds_x0 + half_w
		max_cx = bounds_x1 - half_w
		if max_cx >= min_cx:
			rect_center_x = min(max(rect_center_x, min_cx), max_cx)
		else:
			rect_center_x = (bounds_x0 + bounds_x1) * 0.5
		min_cy = bounds_y0 + half_h
		max_cy = bounds_y1 - half_h
		if max_cy >= min_cy:
			rect_center_y = min(max(rect_center_y, min_cy), max_cy)
		else:
			rect_center_y = (bounds_y0 + bounds_y1) * 0.5
		new_x = rect_center_x - half_w
		new_y = rect_center_y - half_h
		return SceneRect(new_x, new_y, width, height)

	def on_camera_change(self, callback: Optional[Callable[[SceneRect, bool, str], None]]) -> None:
		self._camera_change_callback = callback

	def _emit_camera_change(self, *, source: str = "system") -> None:
		if self._camera_change_callback and self._current_view_rect is not None:
			callback = self._camera_change_callback
			try:
				callback(self._current_view_rect, bool(self._user_camera_override), source)
			except TypeError:
				callback(self._current_view_rect, bool(self._user_camera_override))

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

	def _scene_to_canvas_transform(self) -> Optional[Any]:
		try:
			transform = self.view.scene.node_transform(self.canvas.scene)
		except Exception:
			transform = None
		if transform is None:
			transform = getattr(self.view.scene, "transform", None)
		return transform

	def _map_scene_to_canvas(
		self,
		points: np.ndarray,
		*,
		transform: Optional[Any] = None,
		divide_by_dpr: bool = False,
	) -> np.ndarray:
		if points.size == 0:
			return np.zeros((0, 2), dtype=np.float32)
		if transform is None:
			transform = self._scene_to_canvas_transform()
		if transform is None:
			return np.zeros((0, 2), dtype=np.float32)
		mapped = transform.map(_ensure_3d(points))
		coords = np.asarray(mapped[:, :2], dtype=np.float32)
		if divide_by_dpr:
			dpr = float(self._device_pixel_ratio or 1.0)
			if not math.isclose(dpr, 1.0, rel_tol=1e-6, abs_tol=1e-6):
				coords = coords / dpr
		return coords

	def _compute_tick_positions(
		self,
		start: float,
		stop: float,
		*,
		target_tick_count: int = 7,
	) -> List[float]:
		if not (math.isfinite(start) and math.isfinite(stop)):
			return []
		if abs(stop - start) < 1e-9:
			return [float(start)]
		lower = float(min(start, stop))
		upper = float(max(start, stop))
		if upper - lower < 1e-9:
			return [lower]
		locator = MaxNLocator(nbins=max(3, int(target_tick_count)))
		values = np.asarray(locator.tick_values(lower, upper), dtype=np.float64)
		mask = np.logical_and(values >= lower - 1e-9, values <= upper + 1e-9)
		selected = values[mask]
		if selected.size == 0:
			selected = np.array([lower, upper], dtype=np.float64)
		return [float(val) for val in selected]

	def _update_grid_lines(self, rect: Optional[SceneRect]) -> None:
		if rect is None or rect.width <= 0.0 or rect.height <= 0.0:
			self._grid_lines.visible = False
			self._grid_lines.set_data(pos=np.zeros((0, 3), dtype=np.float32))
			return
		grid_bounds = self._arena_rect or self._scene_rect or rect
		if grid_bounds.width <= 0.0 or grid_bounds.height <= 0.0:
			grid_bounds = rect
		try:
			scene_transform = self.view.scene.transform
			origin = scene_transform.map([rect.x, rect.y, 0.0])
			x_edge = scene_transform.map([rect.x + rect.width, rect.y, 0.0])
			y_edge = scene_transform.map([rect.x, rect.y + rect.height, 0.0])
			x_span_px = float(np.linalg.norm(x_edge[:2] - origin[:2]))
			y_span_px = float(np.linalg.norm(y_edge[:2] - origin[:2]))
		except Exception:
			x_span_px = float(self._viewport_size[0])
			y_span_px = float(self._viewport_size[1])
		x_pixels_per_unit = x_span_px / max(rect.width, 1e-9)
		y_pixels_per_unit = y_span_px / max(rect.height, 1e-9)

		def _dynamic_bins(pixels_per_unit: float, span_px: float) -> int:
			if not math.isfinite(pixels_per_unit) or pixels_per_unit <= 0.0 or not math.isfinite(span_px) or span_px <= 0.0:
				return 7
			base = max(span_px / 140.0, 5.0)
			zoom_bonus = max(0.0, math.log10(max(pixels_per_unit, 1.0))) * 5.0
			count = int(round(base + zoom_bonus))
			return int(max(5, min(24, count)))

		x_bins = _dynamic_bins(x_pixels_per_unit, x_span_px)
		y_bins = _dynamic_bins(y_pixels_per_unit, y_span_px)
		x_ticks = self._compute_tick_positions(rect.x, rect.x + rect.width, target_tick_count=x_bins)
		y_ticks = self._compute_tick_positions(rect.y, rect.y + rect.height, target_tick_count=y_bins)

		def _extend_ticks(ticks: Sequence[float], lower_bound: float, upper_bound: float) -> List[float]:
			unique = sorted({float(t) for t in ticks})
			if not unique:
				return []
			spacing: Optional[float] = None
			if len(unique) >= 2:
				diffs = [unique[i + 1] - unique[i] for i in range(len(unique) - 1)]
				diffs = [diff for diff in diffs if diff > 1e-6 and math.isfinite(diff)]
				if diffs:
					spacing = float(min(diffs))
			if spacing is None or spacing <= 0.0 or not math.isfinite(spacing):
				return unique
			next_tick = unique[-1] + spacing
			loops = 0
			while next_tick <= upper_bound + 1e-6 and loops < 2048:
				unique.append(next_tick)
				next_tick += spacing
				loops += 1
			prev_tick = unique[0] - spacing
			loops = 0
			while prev_tick >= lower_bound - 1e-6 and loops < 2048:
				unique.insert(0, prev_tick)
				prev_tick -= spacing
				loops += 1
			return unique

		x_ticks = _extend_ticks(x_ticks, grid_bounds.x, grid_bounds.x + grid_bounds.width)
		y_ticks = _extend_ticks(y_ticks, grid_bounds.y, grid_bounds.y + grid_bounds.height)
		positions: List[List[float]] = []
		connect: List[List[int]] = []
		x_min_bound = grid_bounds.x
		x_max_bound = grid_bounds.x + grid_bounds.width
		y_min_bound = grid_bounds.y
		y_max_bound = grid_bounds.y + grid_bounds.height
		def _skip_edge(tick_value: float, lower_bound: float, upper_bound: float) -> bool:
			return (
				abs(tick_value - lower_bound) <= 1e-6
				or abs(tick_value - upper_bound) <= 1e-6
			)

		for tick in x_ticks:
			tick_val = float(tick)
			if tick_val < x_min_bound - 1e-6 or tick_val > x_max_bound + 1e-6:
				continue
			if _skip_edge(tick_val, x_min_bound, x_max_bound):
				continue
			start_idx = len(positions)
			positions.append([tick_val, y_min_bound, 0.0])
			positions.append([tick_val, y_max_bound, 0.0])
			connect.append([start_idx, start_idx + 1])
		for tick in y_ticks:
			tick_val = float(tick)
			if tick_val < y_min_bound - 1e-6 or tick_val > y_max_bound + 1e-6:
				continue
			if _skip_edge(tick_val, y_min_bound, y_max_bound):
				continue
			start_idx = len(positions)
			positions.append([x_min_bound, tick_val, 0.0])
			positions.append([x_max_bound, tick_val, 0.0])
			connect.append([start_idx, start_idx + 1])
		if not positions:
			self._grid_lines.visible = False
			self._grid_lines.set_data(pos=np.zeros((0, 3), dtype=np.float32))
			return
		pos_array = _ensure_3d(np.asarray(positions, dtype=np.float32))
		color_rgba = np.tile(_rgba(UI_TEXT_MUTED, 0.28), (pos_array.shape[0], 1))
		indices = np.asarray(connect, dtype=np.uint32)
		self._grid_lines.visible = True
		self._grid_lines.set_data(pos=pos_array, color=color_rgba, width=0.85, connect=indices)

	def _update_scale_bar(self) -> None:
		rect = self._current_view_rect or self._scene_rect
		canvas_w, canvas_h = self._viewport_size
		self._update_reset_button_geometry(float(canvas_w), float(canvas_h))
		if (
			rect is None
			or rect.width <= 0.0
			or rect.height <= 0.0
			or canvas_w <= self._scale_bar_margin * 2.0
			or canvas_h <= self._scale_bar_margin * 2.0
		):
			self._scale_bar_geometry_ready = False
			self._scale_bar_hit_rect = None
			self._scale_bar_line_points = np.zeros((0, 3), dtype=np.float32)
			self._scale_bar_line_connect = np.zeros((0, 2), dtype=np.uint32)
			self._scale_bar_text.text = ""
			self._cancel_scale_bar_fade()
			self._set_scale_bar_alpha(0.0)
			return

		try:
			scene_transform = self.view.scene.transform
			start = scene_transform.map([rect.x, rect.y, 0.0])
			end = scene_transform.map([rect.x + rect.width, rect.y, 0.0])
			delta = np.asarray(end[:2] - start[:2], dtype=np.float32)
			pixel_span = float(np.linalg.norm(delta))
		except Exception:
			pixel_span = 0.0
		if not math.isfinite(pixel_span) or pixel_span <= 1e-6:
			self._scale_bar_geometry_ready = False
			self._scale_bar_hit_rect = None
			self._scale_bar_line_points = np.zeros((0, 3), dtype=np.float32)
			self._scale_bar_line_connect = np.zeros((0, 2), dtype=np.uint32)
			self._scale_bar_text.text = ""
			self._cancel_scale_bar_fade()
			self._set_scale_bar_alpha(0.0)
			return
		pixels_per_unit = pixel_span / max(rect.width, 1e-9)
		if not math.isfinite(pixels_per_unit) or pixels_per_unit <= 1e-9:
			self._scale_bar_geometry_ready = False
			self._scale_bar_hit_rect = None
			self._scale_bar_line_points = np.zeros((0, 3), dtype=np.float32)
			self._scale_bar_line_connect = np.zeros((0, 2), dtype=np.uint32)
			self._scale_bar_text.text = ""
			self._cancel_scale_bar_fade()
			self._set_scale_bar_alpha(0.0)
			return

		max_px = max(float(canvas_w) - self._scale_bar_margin * 2.0, 24.0)
		min_px = min(48.0, max_px)
		available_units = max_px / pixels_per_unit
		min_units = min_px / pixels_per_unit
		if available_units <= 0.0:
			self._scale_bar_geometry_ready = False
			self._scale_bar_hit_rect = None
			self._scale_bar_line_points = np.zeros((0, 3), dtype=np.float32)
			self._scale_bar_line_connect = np.zeros((0, 2), dtype=np.uint32)
			self._scale_bar_text.text = ""
			self._cancel_scale_bar_fade()
			self._set_scale_bar_alpha(0.0)
			return

		def _select_length(min_units_val: float, max_units_val: float) -> float:
			if not math.isfinite(max_units_val) or max_units_val <= 0.0:
				return 0.0
			lower = max(0.0, float(min_units_val))
			upper = float(max_units_val)
			if upper <= lower:
				upper = lower if lower > 0.0 else max_units_val
			exponent = int(math.floor(math.log10(upper))) if upper > 0.0 else 0
			best: Optional[float] = None
			for exp in range(exponent, exponent - 12, -1):
				base = 10.0 ** exp
				for multiplier in (5.0, 2.0, 1.0):
					candidate = multiplier * base
					if candidate > upper * 1.0001:
						continue
					if candidate < lower * 0.9999:
						continue
					best = candidate
					break
				if best is not None:
					break
			if best is None:
				best = max(min(upper, lower if lower > 0.0 else upper), 1e-6)
			return float(best)

		scale_units = _select_length(min_units, available_units)
		if scale_units <= 0.0 or not math.isfinite(scale_units):
			self._scale_bar_geometry_ready = False
			self._scale_bar_hit_rect = None
			self._scale_bar_line_points = np.zeros((0, 3), dtype=np.float32)
			self._scale_bar_line_connect = np.zeros((0, 2), dtype=np.uint32)
			self._scale_bar_text.text = ""
			self._cancel_scale_bar_fade()
			self._set_scale_bar_alpha(0.0)
			self._scale_bar_line.set_data(pos=np.zeros((0, 3), dtype=np.float32))
			return
		scale_px = max(scale_units * pixels_per_unit, 6.0)
		scale_px = min(scale_px, max_px)
		scale_units = max(scale_px / pixels_per_unit, 1e-6)
		tick_height = 8.0
		line_points = np.array(
			[
				[0.0, 0.0, 0.0],
				[0.0, -tick_height, 0.0],
				[scale_px, -tick_height, 0.0],
				[scale_px, 0.0, 0.0],
			],
			dtype=np.float32,
		)
		line_connect = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
		self._scale_bar_line_points = _ensure_3d(line_points)
		self._scale_bar_line_connect = line_connect
		self._scale_bar_line_width = 1.6
		x_start = float(max(self._scale_bar_margin, float(canvas_w) - self._scale_bar_margin - scale_px))
		y_base = float(max(self._scale_bar_margin, float(canvas_h) - self._scale_bar_margin))
		self._scale_bar_line.transform = STTransform(translate=(x_start, y_base, 0.0))

		def _format_units(value: float) -> str:
			if value >= 100.0:
				return f"{value:.0f}"
			if value >= 10.0:
				return f"{value:.1f}"
			if value >= 1.0:
				return f"{value:.2f}"
			return f"{value:.3f}"

		if self._unit_label == "cm" and self._unit_scale > 0.0:
			pixel_equiv = scale_units / self._unit_scale
			label = f"{_format_units(scale_units)} cm  (≈ {pixel_equiv:.0f} px)"
		else:
			label = f"{_format_units(scale_units)} {self._unit_label}"
		self._scale_bar_text.text = label
		text_x = float(canvas_w) - self._scale_bar_margin
		text_y = y_base - tick_height - 6.0
		self._scale_bar_text.transform = STTransform(translate=(text_x, text_y, 0.0))
		self._scale_bar_geometry_ready = True
		text_extent = max(scale_px * 0.6, 120.0)
		hit_left = max(0.0, min(x_start, text_x - text_extent) - 16.0)
		hit_right = min(float(canvas_w), float(canvas_w) - self._scale_bar_margin + 24.0)
		hit_bottom = max(0.0, text_y - 60.0)
		hit_top = min(float(canvas_h), y_base + 24.0)
		self._scale_bar_hit_rect = (hit_left, hit_bottom, hit_right, hit_top)
		self._apply_scale_bar_alpha()
		self._update_reset_button_geometry(float(canvas_w), float(canvas_h))

	def _cancel_scale_bar_fade(self) -> None:
		try:
			self._scale_bar_hold_timer.stop()
		except Exception:
			pass
		try:
			self._scale_bar_fade_timer.stop()
		except Exception:
			pass

	def _set_scale_bar_alpha(self, alpha: float) -> None:
		self._scale_bar_alpha = float(max(0.0, min(1.0, alpha)))
		self._apply_scale_bar_alpha()

	def _apply_scale_bar_alpha(self) -> None:
		if (
			not self._scale_bar_geometry_ready
			or self._scale_bar_line_points.size == 0
			or self._scale_bar_line_connect.size == 0
		):
			self._scale_bar_line.visible = False
			self._scale_bar_text.visible = False
			self._apply_reset_button_alpha(0.0)
			return
		mode = getattr(self, "_scale_bar_mode", "auto")
		alpha = float(max(0.0, min(1.0, self._scale_bar_alpha)))
		if mode == "hidden":
			alpha = 0.0
		elif mode == "locked":
			alpha = 1.0
		color_scale = 0.88 * alpha
		line_colors = np.tile(_rgba(UI_TEXT_PRIMARY, color_scale), (self._scale_bar_line_points.shape[0], 1))
		self._scale_bar_line.set_data(
			pos=self._scale_bar_line_points,
			color=line_colors,
			connect=self._scale_bar_line_connect,
			width=self._scale_bar_line_width,
		)
		self._scale_bar_line.visible = alpha > 0.0
		self._scale_bar_text.color = _rgba(UI_TEXT_PRIMARY, 0.92 * alpha)
		self._scale_bar_text.visible = alpha > 0.0
		self._apply_reset_button_alpha(alpha)
		self.request_draw()

	def on_reset_request(self, callback: Optional[Callable[[], None]]) -> None:
		self._reset_button_callback = callback

	def _initialise_reset_button(self) -> None:
		if self._reset_button is not None:
			return
		if QtWidgets is None or qtawesome is None:
			return
		native = getattr(self.canvas, "native", None)
		if not isinstance(native, QtWidgets.QWidget):
			return
		button = QtWidgets.QToolButton(native)
		button.setObjectName("PoseSceneResetButton")
		button.setAutoRaise(True)
		if QtCore is not None:
			button.setIconSize(QtCore.QSize(22, 22))
			button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
			button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
		icon = None
		if qtawesome is not None:
			for key in ("fa5s.camera", "fa.camera", "mdi.camera" ):
				try:
					icon = qtawesome.icon(key, color="#f4f7ff")
					break
				except Exception:
					continue
		if icon is not None:
			button.setIcon(icon)
		else:
			button.setText("R")
		button.setToolTip("Recenter view (R)")
		button.setStyleSheet(
			"QToolButton#PoseSceneResetButton {"
			" background-color: rgba(17, 26, 48, 170);"
			" border: 1px solid rgba(90, 114, 168, 160);"
			" border-radius: 16px;"
			" padding: 6px;"
			" }"
			" QToolButton#PoseSceneResetButton:hover {"
			" background-color: rgba(28, 40, 68, 210);"
			" }"
			" QToolButton#PoseSceneResetButton:pressed {"
			" background-color: rgba(12, 20, 38, 230);"
			" }"
		)
		button.setFixedSize(34, 34)
		effect = QtWidgets.QGraphicsOpacityEffect(button)
		effect.setOpacity(0.0)
		button.setGraphicsEffect(effect)
		button.hide()
		button.clicked.connect(self._on_reset_button_clicked)
		button.raise_()
		self._reset_button = button
		self._reset_button_effect = effect
		self._update_reset_button_geometry(*self._viewport_size)

	def _update_reset_button_geometry(self, canvas_w: float, canvas_h: float) -> None:
		button = self._reset_button
		if button is None:
			return
		if canvas_w <= 0.0 or canvas_h <= 0.0:
			return
		width = button.width() or button.sizeHint().width()
		height = button.height() or button.sizeHint().height()
		margin = float(self._reset_button_margin)
		x = int(max(margin, canvas_w - margin - width))
		y = int(max(margin, margin))
		button.move(x, y)
		button.raise_()

	def _apply_reset_button_alpha(self, alpha: float) -> None:
		button = self._reset_button
		effect = self._reset_button_effect
		if button is None or effect is None:
			return
		clamped = float(max(0.0, min(1.0, alpha)))
		effect.setOpacity(clamped)
		is_visible = clamped > 0.02
		button.setVisible(is_visible)
		button.setEnabled(clamped >= 0.35)
		if QtCore is not None:
			button.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, clamped <= 0.02)

	def _on_reset_button_clicked(self) -> None:
		button = self._reset_button
		effect = self._reset_button_effect
		if button is None or effect is None:
			return
		if effect.opacity() <= 0.02 or not button.isVisible():
			return
		self._invoke_reset_callback()

	def _invoke_reset_callback(self) -> None:
		callback = self._reset_button_callback
		if callable(callback):
			callback()
		else:
			self.reset_camera_view()

	def _schedule_scale_bar_fade(self, delay: float) -> None:
		self._cancel_scale_bar_fade()
		mode = getattr(self, "_scale_bar_mode", "auto")
		if mode == "hidden":
			self._set_scale_bar_alpha(0.0)
			return
		if mode == "locked":
			self._set_scale_bar_alpha(1.0)
			return
		delay = float(max(0.0, delay))
		if delay <= 0.0:
			self._start_scale_bar_fade()
			return
		self._scale_bar_hold_timer.interval = delay
		self._scale_bar_hold_timer.start(iterations=1)

	def _start_scale_bar_fade(self) -> None:
		mode = getattr(self, "_scale_bar_mode", "auto")
		if mode == "hidden":
			self._set_scale_bar_alpha(0.0)
			return
		if mode == "locked":
			self._set_scale_bar_alpha(1.0)
			return
		if not self._scale_bar_geometry_ready or self._scale_bar_alpha <= 0.0:
			self._set_scale_bar_alpha(0.0)
			return
		self._scale_bar_fade_start = time.perf_counter()
		self._scale_bar_fade_timer.start()

	def _on_scale_bar_hold_timeout(self, event: Optional[Any] = None) -> None:
		self._start_scale_bar_fade()

	def _on_scale_bar_fade_step(self, event: Optional[Any] = None) -> None:
		duration = float(max(self._scale_bar_fade_duration, 1e-6))
		elapsed = time.perf_counter() - self._scale_bar_fade_start
		progress = float(min(1.0, max(0.0, elapsed / duration)))
		self._set_scale_bar_alpha(1.0 - progress)
		if progress >= 1.0 - 1e-3:
			self._scale_bar_fade_timer.stop()
			self._set_scale_bar_alpha(0.0)

	def show_scale_bar_hint(self, *, hold_delay: Optional[float] = None) -> None:
		if not self._scale_bar_geometry_ready:
			self._update_scale_bar()
		delay = float(self._scale_bar_idle_delay if hold_delay is None else max(0.0, hold_delay))
		mode = getattr(self, "_scale_bar_mode", "auto")
		if mode == "hidden":
			self._cancel_scale_bar_fade()
			self._set_scale_bar_alpha(0.0)
			return
		if mode == "locked":
			self._cancel_scale_bar_fade()
			self._set_scale_bar_alpha(1.0)
			return
		self._set_scale_bar_alpha(1.0)
		self._schedule_scale_bar_fade(delay)

	def _cycle_scale_bar_mode(self) -> None:
		current = getattr(self, "_scale_bar_mode", "auto")
		if current == "auto":
			self._scale_bar_mode = "locked"
			self._cancel_scale_bar_fade()
			self._set_scale_bar_alpha(1.0)
		elif current == "locked":
			self._scale_bar_mode = "hidden"
			self._cancel_scale_bar_fade()
			self._set_scale_bar_alpha(0.0)
		else:
			self._scale_bar_mode = "auto"
			self._scale_bar_alpha = 1.0
			self.show_scale_bar_hint(hold_delay=self._scale_bar_idle_delay)
		self._scale_bar_click_armed = False

	def _scale_bar_contains_point(self, x: float, y: float) -> bool:
		rect = self._scale_bar_hit_rect
		if rect is None:
			return False
		left, bottom, right, top = rect
		return left <= x <= right and bottom <= y <= top

	@staticmethod
	def _is_primary_mouse_button(event: Any) -> bool:
		button = getattr(event, "button", None)
		if isinstance(button, str):
			name = button.lower()
			if name in {"left", "button1", "primary"}:
				return True
		if button == 1:
			return True
		name_attr = getattr(button, "name", None)
		if isinstance(name_attr, str) and name_attr.lower() in {"left", "button1", "primary"}:
			return True
		return False

	def _handle_scale_bar_mouse_event(self, event: Any) -> bool:
		if event is None:
			return False
		event_type = getattr(event, "type", None) or getattr(event, "name", None)
		pos = getattr(event, "pos", None)
		if pos is None:
			return False
		try:
			x, y = float(pos[0]), float(pos[1])
		except (TypeError, ValueError):
			return False
		if event_type == "mouse_press" and self._scale_bar_contains_point(x, y) and self._is_primary_mouse_button(event):
			self._scale_bar_click_armed = True
			if hasattr(event, "handled"):
				event.handled = True
			return True
		if event_type == "mouse_release":
			if self._scale_bar_click_armed and self._scale_bar_contains_point(x, y) and self._is_primary_mouse_button(event):
				self._cycle_scale_bar_mode()
				if hasattr(event, "handled"):
					event.handled = True
				self._scale_bar_click_armed = False
				return True
			self._scale_bar_click_armed = False
		elif event_type == "mouse_move" and not self._scale_bar_contains_point(x, y):
			self._scale_bar_click_armed = False
		return False

	def begin_frame(
		self,
		*,
		xlim: Tuple[float, float],
		ylim: Tuple[float, float],
		aspect_ratio: Optional[float] = None,
		domain_xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
		domain_ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
		preserve_view: bool = False,
	) -> None:
		self.set_limits(
			xlim,
			ylim,
			aspect_ratio=aspect_ratio,
			domain_xlim=domain_xlim,
			domain_ylim=domain_ylim,
			preserve_view=preserve_view,
		)
		self.reset_frame_state()
		self._clear_persistent_visuals()
		self._set_hover(None)

	def _clear_persistent_visuals(self) -> None:
		for polygon in self._hull_polygons:
			polygon.parent = None
		self._hull_polygons.clear()
		for text in self._label_texts:
			text.parent = None
		self._label_texts.clear()
		self._label_bounds.clear()
		self._label_line_segments.clear()
		self._label_line_colors.clear()

	def add_glow_markers(self, positions: np.ndarray, *, base_color: Sequence[float]) -> None:
		if positions.size == 0:
			return
		positions_scene = self._to_scene_units_array(positions).astype(np.float32, copy=False)
		count = positions_scene.shape[0]
		if count == 0:
			return

		base_rgb = np.clip(np.array(to_rgb(base_color), dtype=np.float32), 0.0, 1.0)
		view_rect = self._current_view_rect or self._base_view_rect or self._scene_rect
		if view_rect is not None:
			span = float(max(getattr(view_rect, "width", 0.0), getattr(view_rect, "height", 0.0), 1.0))
		else:
			span = float(
				max(
					float(np.ptp(positions_scene[:, 0]) if count > 1 else 1.0),
					float(np.ptp(positions_scene[:, 1]) if count > 1 else 1.0),
					1.0,
				)
			)
		scene_rect = self._scene_rect or view_rect
		baseline_span = float(max(getattr(scene_rect, "width", span), getattr(scene_rect, "height", span), span)) if scene_rect is not None else span
		zoom_level = float(np.clip(baseline_span / max(span, 1e-3), 0.2, 5.0))

		density = np.zeros((count,), dtype=np.float32)
		scale_reference = 1.0
		if count > 1:
			diff = positions_scene[:, None, :2] - positions_scene[None, :, :2]
			dist_sq = np.sum(diff * diff, axis=2).astype(np.float32, copy=False)
			mask = ~np.eye(count, dtype=bool)
			valid = dist_sq[mask]
			positive = valid[valid > 1e-6]
			if positive.size > 0:
				scale_reference = float(np.median(positive))
			elif valid.size > 0:
				scale_reference = float(np.mean(valid))
			if not math.isfinite(scale_reference) or scale_reference <= 1e-6:
				scale_reference = 1.0
			sigma = max(math.sqrt(scale_reference) * 0.7 + 14.0, 5.5)
			sigma_sq = max(sigma * sigma, 1.0)
			influence = np.exp(-dist_sq / sigma_sq).astype(np.float32, copy=False)
			np.fill_diagonal(influence, 0.0)
			density = influence.sum(axis=1).astype(np.float32, copy=False)

		if density.size > 0:
			max_d = float(np.max(density))
			min_d = float(np.min(density))
			if max_d - min_d > 1e-6:
				density = (density - min_d) / (max_d - min_d)
			else:
				density.fill(0.0)

		density_gamma = density.astype(np.float32, copy=False) ** 0.72
		global_density = float(np.mean(density_gamma)) if count else 0.0
		max_density = float(np.max(density_gamma)) if count else 0.0
		global_penalty = float(np.clip(1.0 / (1.0 + global_density * 1.8 + max_density * 1.25), 0.15, 0.8))
		camera_intensity = float(np.clip(0.26 + zoom_level ** 0.65 * 0.32, 0.22, 0.85))
		density_penalty = 1.0 / (1.0 + density_gamma * (0.82 + 0.28 * camera_intensity))
		density_penalty = density_penalty.astype(np.float32, copy=False)
		brightness_scale = camera_intensity * global_penalty * 0.46
		size_scale = float(np.clip(np.interp(zoom_level, [0.2, 1.0, 3.5, 5.0], [2.4, 1.0, 0.65, 0.5]), 0.45, 2.6))
		spacing_scale = float(np.clip(math.sqrt(scale_reference), 0.75, 5.0)) if scale_reference > 0.0 else 1.0
		base_size = np.clip(54.0 * size_scale * spacing_scale ** 0.28, 16.0, 200.0)
		base_rgb_broadcast = np.tile(base_rgb, (count, 1)).astype(np.float32, copy=False)
		base_mean = float(np.mean(base_rgb))
		color_offset = (base_rgb - base_mean).astype(np.float32, copy=False)
		offset = color_offset.reshape(1, 3)
		core_gain = (0.72 + density_gamma[:, None] * 0.05 + camera_intensity * 0.04).astype(np.float32, copy=False)
		core_rgb = np.clip(
			base_rgb_broadcast * core_gain
			+ offset * (0.32 + density_gamma[:, None] * 0.18),
			0.0,
			1.0,
		)
		highlight_bias = np.array([0.17, 0.15, 0.11], dtype=np.float32)
		highlight_rgb = np.clip(
			base_rgb_broadcast * (0.54 + camera_intensity * 0.05)
			+ offset * (0.55 + density_gamma[:, None] * 0.22)
			+ highlight_bias[None, :],
			0.0,
			1.0,
		)
		sat_factor = np.clip(1.08 + density_gamma[:, None] * 0.18 + camera_intensity * 0.06, 1.0, 1.34).astype(np.float32, copy=False)
		layer_specs = (
			(0.58, 0.22, 0.0),
			(1.12, 0.11, 0.25),
			(1.85, 0.055, 0.47),
			(2.65, 0.028, 0.74),
		)
		for radius_scale, alpha_base, mix_bias in layer_specs:
			mix = np.clip(mix_bias + density_gamma * 0.17, 0.0, 1.0)
			color_rgb = (1.0 - mix)[:, None] * core_rgb + mix[:, None] * highlight_rgb
			grey = np.mean(color_rgb, axis=1, keepdims=True).astype(np.float32, copy=False)
			color_rgb = grey + (color_rgb - grey) * sat_factor
			color_rgb = np.clip(color_rgb * 0.64 + base_rgb[None, :] * 0.36, 0.0, 1.0)
			layer_alpha = np.clip(alpha_base * brightness_scale * density_penalty, 0.003, 0.11)
			size_array = np.clip(base_size * radius_scale * (0.82 + density_gamma * 0.24), 8.0, 220.0).astype(np.float32, copy=False)
			color_array = np.empty((count, 4), dtype=np.float32)
			color_array[:, :3] = color_rgb.astype(np.float32, copy=False)
			color_array[:, 3] = layer_alpha.astype(np.float32, copy=False)
			self._glow_positions.append(positions_scene)
			self._glow_sizes.append(size_array)
			self._glow_colors.append(color_array)

	def add_body_markers(
		self,
		positions: np.ndarray,
		*,
		base_color: Sequence[float],
		labels: Sequence[str],
		mouse_id: str,
		edge_color: Sequence[float],
		edge_width: float,
		size: float = 7.0,
	) -> None:
		if positions.size == 0:
			return
		positions_scene = self._to_scene_units_array(positions).astype(np.float32, copy=False)
		count = positions_scene.shape[0]
		if count == 0:
			return
		size_array = np.full((count,), float(size), dtype=np.float32)
		face_rgba = np.tile(_rgba(base_color, 0.92), (count, 1))
		edge_rgba = np.tile(_rgba(edge_color, 0.85), (count, 1))
		edge_widths = np.full((count,), float(edge_width), dtype=np.float32)
		self._body_positions.append(positions_scene)
		self._body_sizes.append(size_array)
		self._body_face_colors.append(face_rgba)
		self._body_edge_colors.append(edge_rgba)
		self._body_edge_widths.append(edge_widths)
		label_tuple = tuple(labels)
		if len(label_tuple) < count:
			label_tuple = label_tuple + tuple(f"node-{idx}" for idx in range(len(label_tuple), count))
		self._hover_datasets.append(
			HoverDataset(
				positions=positions_scene[:, :2].astype(np.float32, copy=False),
				labels=label_tuple,
				mouse_id=mouse_id,
			)
		)

	def add_tail_markers(
		self,
		positions: np.ndarray,
		*,
		base_color: Sequence[float],
		labels: Sequence[str],
		mouse_id: str,
		edge_color: Sequence[float],
		edge_width: float,
		size: float = 5.0,
	) -> None:
		if positions.size == 0:
			return
		positions_scene = self._to_scene_units_array(positions).astype(np.float32, copy=False)
		count = positions_scene.shape[0]
		if count == 0:
			return
		size_array = np.full((count,), float(size), dtype=np.float32)
		face_rgba = np.tile(_rgba(base_color, 0.9), (count, 1))
		edge_rgba = np.tile(_rgba(edge_color, 0.8), (count, 1))
		edge_widths = np.full((count,), float(edge_width), dtype=np.float32)
		self._tail_positions.append(positions_scene)
		self._tail_sizes.append(size_array)
		self._tail_face_colors.append(face_rgba)
		self._tail_edge_colors.append(edge_rgba)
		self._tail_edge_widths.append(edge_widths)
		label_tuple = tuple(labels)
		if len(label_tuple) < count:
			label_tuple = label_tuple + tuple(f"tail-{idx}" for idx in range(len(label_tuple), count))
		self._hover_datasets.append(
			HoverDataset(
				positions=positions_scene[:, :2].astype(np.float32, copy=False),
				labels=label_tuple,
				mouse_id=mouse_id,
			)
		)

	def add_trail_segments(
		self,
		segments: Iterable[Tuple[np.ndarray, np.ndarray, float]],
		*,
		color: Sequence[float],
		width: float,
	) -> None:
		if not segments:
			return
		segment_positions: List[np.ndarray] = []
		segment_colors: List[np.ndarray] = []
		for entry in segments:
			if len(entry) == 3:
				start, end, alpha = entry
			elif len(entry) == 2:
				start, end = entry  # type: ignore[misc]
				alpha = 1.0
			else:
				continue
			stack = np.vstack((np.asarray(start, dtype=np.float32), np.asarray(end, dtype=np.float32)))
			stack_scene = self._to_scene_units_array(stack).astype(np.float32, copy=False)
			segment_positions.append(stack_scene)
			segment_color = np.tile(_rgba(color, float(alpha)), (stack_scene.shape[0], 1))
			segment_colors.append(segment_color)
		if not segment_positions:
			return
		self._trail_segments.extend(segment_positions)
		self._trail_colors.extend(segment_colors)
		self._trail_widths.append(float(width))

	def add_body_edges(
		self,
		segments: Iterable[Tuple[np.ndarray, np.ndarray]],
		*,
		color: Sequence[float],
		width: float,
	) -> None:
		if not segments:
			return
		for start, end in segments:
			stack = np.vstack((np.asarray(start, dtype=np.float32), np.asarray(end, dtype=np.float32)))
			stack_scene = self._to_scene_units_array(stack).astype(np.float32, copy=False)
			self._edge_segments.append(stack_scene)
			self._edge_colors.append(np.tile(_rgba(color, 0.9), (stack_scene.shape[0], 1)))

	def add_tail_polyline(
		self,
		polyline: np.ndarray,
		*,
		primary_color: Sequence[float],
		primary_width: float,
		secondary_color: Sequence[float],
		secondary_width: float,
	) -> None:
		if polyline.size == 0:
			return
		polyline_scene = self._to_scene_units_array(polyline).astype(np.float32, copy=False)
		if polyline_scene.shape[0] < 2:
			return
		self._tail_core_points.append(polyline_scene)
		self._tail_core_colors.append(np.tile(_rgba(primary_color, 0.6), (polyline_scene.shape[0], 1)))
		self._tail_core_widths.append(float(primary_width))
		self._tail_overlay_points.append(polyline_scene.copy())
		self._tail_overlay_colors.append(np.tile(_rgba(secondary_color, 0.45), (polyline_scene.shape[0], 1)))
		self._tail_overlay_widths.append(float(secondary_width))

	def add_label(
		self,
		text: str,
		position: np.ndarray,
		*,
		points: Optional[np.ndarray] = None,
		color: Sequence[float],
		border_color: Sequence[float],
	) -> None:
		pos_array = np.asarray(position, dtype=np.float32)
		pos_scene = self._to_scene_units_array(pos_array.reshape(1, -1))[0]
		anchor_xy = np.array([float(pos_scene[0]), float(pos_scene[1])], dtype=np.float32)
		if points is None:
			points_scene = np.zeros((0, 2), dtype=np.float32)
		else:
			points_array = np.asarray(points, dtype=np.float32)
			if points_array.ndim == 1:
				points_array = points_array.reshape(1, -1)
			points_scene_full = self._to_scene_units_array(points_array)
			points_scene = points_scene_full[:, :2].astype(np.float32, copy=False)
		color_values = np.asarray(color, dtype=np.float32).flatten()
		if color_values.size == 0:
			color_values = np.array([1.0, 1.0, 1.0], dtype=np.float32)
		elif color_values.size < 3:
			color_values = np.pad(color_values, (0, 3 - color_values.size), mode="edge")
		color_tuple = (float(color_values[0]), float(color_values[1]), float(color_values[2]))
		border_values = np.asarray(border_color, dtype=np.float32).flatten()
		if border_values.size == 0:
			border_values = np.array([1.0, 1.0, 1.0], dtype=np.float32)
		elif border_values.size < 3:
			border_values = np.pad(border_values, (0, 3 - border_values.size), mode="edge")
		border_tuple = (float(border_values[0]), float(border_values[1]), float(border_values[2]))
		definition = LabelDefinition(
			text=text,
			anchor=anchor_xy,
			points=points_scene,
			color=color_tuple,
			border_color=border_tuple,
		)
		self._label_definitions.append(definition)
		self._label_layout_dirty = True

	def add_whisker_segments(
		self,
		segments: Iterable[Tuple[np.ndarray, np.ndarray]],
		*,
		primary_color: Sequence[float],
		primary_width: float,
		secondary_color: Sequence[float],
		secondary_width: float,
	) -> None:
		primary_positions: List[np.ndarray] = []
		secondary_positions: List[np.ndarray] = []
		for start, end in segments:
			primary_positions.append(np.vstack((start, end)))
			secondary_positions.append(np.vstack((start, end * 0.66 + start * 0.34)))
		if primary_positions:
			primary_stack = self._to_scene_units_array(np.vstack(primary_positions))
			self._whisker_primary_segments.append(primary_stack)
			self._whisker_primary_colors.append(np.tile(_rgba(primary_color, 0.6), (primary_stack.shape[0], 1)))
			self._whisker_primary_widths.append(float(primary_width * 0.65))
		if secondary_positions:
			secondary_stack = self._to_scene_units_array(np.vstack(secondary_positions))
			self._whisker_secondary_segments.append(secondary_stack)
			self._whisker_secondary_colors.append(np.tile(_rgba(secondary_color, 0.4), (secondary_stack.shape[0], 1)))
			self._whisker_secondary_widths.append(float(secondary_width * 0.65))

	def add_hull(self, polygon: np.ndarray, *, color: Sequence[float]) -> None:
		if polygon.size == 0:
			return
		polygon_scene = self._to_scene_units_array(polygon)
		poly = visuals.Polygon(
			pos=polygon_scene,
			color=_rgba(color, 0.22),
			border_color=None,
			parent=self.view.scene,
		)
		poly.set_gl_state("translucent", depth_test=False)
		self._hull_polygons.append(poly)

	def _label_identifier_from_text(self, text: str) -> int:
		digits = "".join(ch for ch in text if ch.isdigit())
		if digits:
			try:
				return int(digits)
			except Exception:
				pass
		return abs(hash(text)) % 104729 or 1

	def _label_direction_vector(self, anchor_xy: np.ndarray, label_key: str) -> np.ndarray:
		anchor = np.asarray(anchor_xy[:2], dtype=np.float32)
		view_rect = self._current_view_rect or self._base_view_rect or self._scene_rect
		if view_rect is not None:
			center = np.array(view_rect.center, dtype=np.float32)
			vector = anchor - center
			norm = float(np.linalg.norm(vector))
			if norm > 1e-5:
				return vector / norm
		identifier = self._label_identifier_from_text(label_key)
		angle = (identifier * GOLDEN_ANGLE_RADIANS) % (2.0 * math.pi)
		return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

	def _label_offset_distance(self) -> float:
		view_rect = self._current_view_rect or self._base_view_rect or self._scene_rect
		diagonal = None
		if view_rect is not None:
			diagonal = math.hypot(float(view_rect.width), float(view_rect.height))
		if diagonal is None or not math.isfinite(diagonal):
			return 36.0
		return float(min(max(diagonal * 0.045, 28.0), 160.0))

	@staticmethod
	def _label_anchor_for_direction(direction: np.ndarray) -> Tuple[str, str]:
		dx = float(direction[0])
		dy = float(direction[1])
		if dx > 0.35:
			anchor_x = "left"
		elif dx < -0.35:
			anchor_x = "right"
		else:
			anchor_x = "center"
		if dy > 0.35:
			anchor_y = "bottom"
		elif dy < -0.35:
			anchor_y = "top"
		else:
			anchor_y = "center"
		return anchor_x, anchor_y

	def _estimate_text_extent(self, text: str, font_size: float) -> Tuple[float, float]:
		characters = max(len(text), 1)
		width = max(font_size * (LABEL_TEXT_WIDTH_FACTOR * characters + LABEL_TEXT_EXTRA_CHARS), font_size * 2.8)
		height = font_size * LABEL_TEXT_LINE_HEIGHT
		return float(width), float(height)

	@staticmethod
	def _bounds_from_anchor(
		position: Tuple[float, float],
		anchor: Tuple[str, str],
		extent: Tuple[float, float],
		padding: float = 0.0,
	) -> Tuple[float, float, float, float]:
		x, y = position
		width, height = extent
		anchor_x, anchor_y = anchor
		half_w = width * 0.5
		half_h = height * 0.5
		if anchor_x == "left":
			x0, x1 = x - padding * 0.5, x + width + padding * 0.5
		elif anchor_x == "right":
			x0, x1 = x - width - padding * 0.5, x + padding * 0.5
		else:  # center
			x0, x1 = x - half_w - padding * 0.5, x + half_w + padding * 0.5
		if anchor_y == "bottom":
			y0, y1 = y - padding * 0.5, y + height + padding * 0.5
		elif anchor_y == "top":
			y0, y1 = y - height - padding * 0.5, y + padding * 0.5
		else:
			y0, y1 = y - half_h - padding * 0.5, y + half_h + padding * 0.5
		if x0 > x1:
			x0, x1 = x1, x0
		if y0 > y1:
			y0, y1 = y1, y0
		return (float(x0), float(y0), float(x1), float(y1))

	def _scene_scale_at_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
		transform = self._scene_to_canvas_transform()
		if transform is None:
			return (1.0, 1.0)
		x, y = point
		origin = np.asarray(transform.map([float(x), float(y), 0.0])[:2], dtype=np.float64)
		x_unit = np.asarray(transform.map([float(x) + 1.0, float(y), 0.0])[:2], dtype=np.float64)
		y_unit = np.asarray(transform.map([float(x), float(y) + 1.0, 0.0])[:2], dtype=np.float64)
		scale_x = float(np.linalg.norm(x_unit - origin))
		scale_y = float(np.linalg.norm(y_unit - origin))
		if not math.isfinite(scale_x) or scale_x <= 1e-6:
			scale_x = scale_y if math.isfinite(scale_y) and scale_y > 1e-6 else 1.0
		if not math.isfinite(scale_y) or scale_y <= 1e-6:
			scale_y = scale_x if math.isfinite(scale_x) and scale_x > 1e-6 else 1.0
		return (max(scale_x, 1e-6), max(scale_y, 1e-6))

	def _bounds_in_view(self, bounds: Tuple[float, float, float, float]) -> bool:
		view_rect = self._current_view_rect or self._base_view_rect or self._scene_rect
		if view_rect is None:
			return True
		x0, y0, x1, y1 = bounds
		left = float(view_rect.x)
		right = float(view_rect.x + view_rect.width)
		bottom = float(view_rect.y)
		top = float(view_rect.y + view_rect.height)
		return x0 >= left and x1 <= right and y0 >= bottom and y1 <= top

	@staticmethod
	def _bounds_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
		a_x0, a_y0, a_x1, a_y1 = a
		b_x0, b_y0, b_x1, b_y1 = b
		return not (a_x1 <= b_x0 or a_x0 >= b_x1 or a_y1 <= b_y0 or a_y0 >= b_y1)

	def add_label(
		self,
		text: str,
		position: np.ndarray,
		*,
		points: Optional[np.ndarray] = None,
		color: Sequence[float],
		border_color: Sequence[float],
	) -> None:
		pos_array = np.asarray(position, dtype=np.float32)
		pos_scene = self._to_scene_units_array(pos_array.reshape(1, -1))[0]
		anchor_xy = np.array([float(pos_scene[0]), float(pos_scene[1])], dtype=np.float32)
		if points is None:
			points_scene = np.zeros((0, 2), dtype=np.float32)
		else:
			points_array = np.asarray(points, dtype=np.float32)
			if points_array.ndim == 1:
				points_array = points_array.reshape(1, -1)
			points_scene_full = self._to_scene_units_array(points_array)
			points_scene = points_scene_full[:, :2].astype(np.float32, copy=False)
		color_values = np.asarray(color, dtype=np.float32).flatten()
		if color_values.size == 0:
			color_values = np.array([1.0, 1.0, 1.0], dtype=np.float32)
		elif color_values.size < 3:
			color_values = np.pad(color_values, (0, 3 - color_values.size), mode="edge")
		color_tuple = (float(color_values[0]), float(color_values[1]), float(color_values[2]))
		border_values = np.asarray(border_color, dtype=np.float32).flatten()
		if border_values.size == 0:
			border_values = np.array([1.0, 1.0, 1.0], dtype=np.float32)
		elif border_values.size < 3:
			border_values = np.pad(border_values, (0, 3 - border_values.size), mode="edge")
		border_tuple = (float(border_values[0]), float(border_values[1]), float(border_values[2]))
		definition = LabelDefinition(
			text=text,
			anchor=anchor_xy,
			points=points_scene,
			color=color_tuple,
			border_color=border_tuple,
		)
		self._label_definitions.append(definition)
		self._label_layout_dirty = True

	def _point_in_view(self, point: Sequence[float]) -> bool:
		view_rect = self._current_view_rect or self._base_view_rect or self._scene_rect
		if view_rect is None:
			return True
		try:
			x_val = float(point[0])
			y_val = float(point[1])
		except Exception:
			return False
		left = float(view_rect.x)
		right = float(view_rect.x + view_rect.width)
		bottom = float(view_rect.y)
		top = float(view_rect.y + view_rect.height)
		return (left <= x_val <= right) and (bottom <= y_val <= top)

	def _points_visible(self, points: np.ndarray, anchor: np.ndarray) -> bool:
		if points.size > 0:
			for point in points:
				if self._point_in_view(point):
					return True
			return False
		return self._point_in_view(anchor)

	def _layout_single_label(self, definition: LabelDefinition) -> None:
		if not self._points_visible(definition.points, definition.anchor):
			return
		anchor_xy = definition.anchor.astype(np.float32, copy=False)
		base_direction = self._label_direction_vector(anchor_xy, definition.text)
		norm = float(np.linalg.norm(base_direction))
		if norm <= 1e-5:
			base_direction = np.array([0.0, 1.0], dtype=np.float32)
		else:
			base_direction = (base_direction / norm).astype(np.float32, copy=False)
		base_angle = math.atan2(float(base_direction[1]), float(base_direction[0]))
		offset = self._label_offset_distance()
		label = visuals.Text(
			definition.text,
			color=_rgba(definition.color, 0.95),
			font_size=11,
			parent=self.view.scene,
		)
		extent_px = self._estimate_text_extent(definition.text, float(label.font_size or 11.0))
		best_bounds: Optional[Tuple[float, float, float, float]] = None
		best_position: Optional[Tuple[float, float, float]] = None
		best_anchor: Tuple[str, str] = ("center", "center")
		candidate_found = False
		for attempt in range(LABEL_MAX_ATTEMPTS):
			angle = base_angle + attempt * GOLDEN_ANGLE_RADIANS
			direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
			direction_norm = float(np.linalg.norm(direction))
			if direction_norm <= 1e-6:
				continue
			direction = (direction / direction_norm).astype(np.float32, copy=False)
			for radius_scale in LABEL_RADIUS_SCALES:
				offset_value = float(offset * radius_scale)
				label_xy = anchor_xy + direction * offset_value
				anchor_x, anchor_y = self._label_anchor_for_direction(direction)
				label.anchors = (anchor_x, anchor_y)
				label.pos = (float(label_xy[0]), float(label_xy[1]), 0.0)
				scale_x, scale_y = self._scene_scale_at_point((float(label_xy[0]), float(label_xy[1])))
				padding_scene = LABEL_OVERLAP_PADDING / max(min(scale_x, scale_y), 1e-6)
				extent_scene = (extent_px[0] / scale_x, extent_px[1] / scale_y)
				bounds = self._bounds_from_anchor((float(label_xy[0]), float(label_xy[1])), (anchor_x, anchor_y), extent_scene, padding_scene)
				if not self._bounds_in_view(bounds):
					continue
				if any(self._bounds_overlap(bounds, existing) for existing in self._label_bounds):
					continue
				best_bounds = bounds
				best_position = (float(label_xy[0]), float(label_xy[1]), 0.0)
				best_anchor = (anchor_x, anchor_y)
				candidate_found = True
				break
			if candidate_found:
				break
		if not candidate_found:
			fallback_anchor = self._label_anchor_for_direction(base_direction)
			label.anchors = fallback_anchor
			label.pos = (float(anchor_xy[0]), float(anchor_xy[1]), 0.0)
			scale_x, scale_y = self._scene_scale_at_point((float(anchor_xy[0]), float(anchor_xy[1])))
			padding_scene = LABEL_OVERLAP_PADDING / max(min(scale_x, scale_y), 1e-6)
			extent_scene = (extent_px[0] / scale_x, extent_px[1] / scale_y)
			best_bounds = self._bounds_from_anchor((float(anchor_xy[0]), float(anchor_xy[1])), fallback_anchor, extent_scene, padding_scene)
			best_position = tuple(label.pos)
			best_anchor = fallback_anchor
		if best_position is None:
			best_position = (float(anchor_xy[0]), float(anchor_xy[1]), 0.0)
		if best_bounds is None:
			scale_x, scale_y = self._scene_scale_at_point((float(best_position[0]), float(best_position[1])))
			padding_scene = LABEL_OVERLAP_PADDING / max(min(scale_x, scale_y), 1e-6)
			extent_scene = (extent_px[0] / scale_x, extent_px[1] / scale_y)
			best_bounds = self._bounds_from_anchor((float(best_position[0]), float(best_position[1])), best_anchor, extent_scene, padding_scene)
		label.anchors = best_anchor
		label.pos = best_position
		self._label_texts.append(label)
		self._label_bounds.append(best_bounds)
		line_start = np.array(best_position, dtype=np.float32)
		line_end = np.array([float(anchor_xy[0]), float(anchor_xy[1]), 0.0], dtype=np.float32)
		if np.linalg.norm(line_start - line_end) <= 1e-4:
			return
		line_segment = np.vstack((line_start, line_end)).astype(np.float32, copy=False)
		line_color = np.tile(_rgba(definition.color, 0.55), (2, 1)).astype(np.float32, copy=False)
		self._label_line_segments.append(line_segment)
		self._label_line_colors.append(line_color)

	def _layout_labels(self) -> None:
		if not self._label_layout_dirty:
			return
		for text in self._label_texts:
			text.parent = None
		self._label_texts.clear()
		self._label_bounds.clear()
		self._label_line_segments.clear()
		self._label_line_colors.clear()
		if not self._label_definitions:
			self._label_lines.visible = False
			self._label_lines.set_data(pos=np.zeros((0, 3), dtype=np.float32), color=np.zeros((0, 4), dtype=np.float32), width=0.0)
			self._label_layout_dirty = False
			return
		for definition in self._label_definitions:
			self._layout_single_label(definition)
		if self._label_line_segments:
			segments = np.vstack(self._label_line_segments)
			colors = np.vstack(self._label_line_colors)
			self._label_lines.visible = True
			self._label_lines.set_data(pos=_ensure_3d(segments), color=colors, width=1.0)
		else:
			self._label_lines.visible = False
			self._label_lines.set_data(pos=np.zeros((0, 3), dtype=np.float32), color=np.zeros((0, 4), dtype=np.float32), width=0.0)
		self._label_layout_dirty = False

	def finalise_frame(self) -> None:
		def _update_line_visual(line: visuals.Line, pos: Optional[np.ndarray], color: Optional[np.ndarray], width: float) -> None:
			if (
				pos is not None
				and pos.size > 0
				and color is not None
				and color.size > 0
				and color.shape[0] == pos.shape[0]
			):
				line.visible = True
				line.set_data(pos=_ensure_3d(pos), color=color, width=width)
			else:
				line.visible = False
				dummy_pos = np.zeros((1, 3), dtype=np.float32)
				dummy_color = np.zeros((1, 4), dtype=np.float32)
				line.set_data(pos=dummy_pos, color=dummy_color, width=0.0)

		def _concat(arrays: List[np.ndarray]) -> Optional[np.ndarray]:
			if not arrays:
				return None
			if len(arrays) == 1:
				return arrays[0]
			return np.concatenate(arrays, axis=0)

		glow_pos = _concat(self._glow_positions)
		if glow_pos is not None:
			self._glow_markers.set_data(
				pos=_ensure_3d(glow_pos),
				size=_concat(self._glow_sizes),
				face_color=_concat(self._glow_colors),
				edge_width=0.0,
				symbol="disc",
			)
		else:
			self._glow_markers.set_data(pos=np.zeros((0, 3), dtype=np.float32))

		body_pos = _concat(self._body_positions)
		if body_pos is not None:
			self._body_markers.set_data(
				pos=_ensure_3d(body_pos),
				size=_concat(self._body_sizes),
				face_color=_concat(self._body_face_colors),
				edge_color=_concat(self._body_edge_colors),
				edge_width=_concat(self._body_edge_widths),
			)
		else:
			self._body_markers.set_data(pos=np.zeros((0, 3), dtype=np.float32))

		tail_pos = _concat(self._tail_positions)
		if tail_pos is not None:
			self._tail_markers.set_data(
				pos=_ensure_3d(tail_pos),
				size=_concat(self._tail_sizes),
				face_color=_concat(self._tail_face_colors),
				edge_color=_concat(self._tail_edge_colors),
				edge_width=_concat(self._tail_edge_widths),
			)
		else:
			self._tail_markers.set_data(pos=np.zeros((0, 3), dtype=np.float32))

		trail_pos = _concat(self._trail_segments)
		trail_colors = _concat(self._trail_colors)
		trail_width = float(np.mean(self._trail_widths or [2.0]))
		_update_line_visual(self._trail_lines, trail_pos, trail_colors, trail_width)

		edge_pos = _concat(self._edge_segments)
		edge_colors = _concat(self._edge_colors)
		_update_line_visual(self._edge_lines, edge_pos, edge_colors, 1.2)

		def _stack_polylines(points_list: List[np.ndarray], color_list: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
			if not points_list or not color_list:
				return None, None
			valid_pairs = [
				(_ensure_3d(pts), cols.astype(np.float32, copy=False))
				for pts, cols in zip(points_list, color_list)
				if pts.size > 0 and cols.size > 0
			]
			if not valid_pairs:
				return None, None
			pos_parts: List[np.ndarray] = []
			color_parts: List[np.ndarray] = []
			last_index = len(valid_pairs) - 1
			for idx, (pts3d, cols_arr) in enumerate(valid_pairs):
				pos_parts.append(pts3d)
				color_parts.append(cols_arr)
				if idx != last_index:
					pos_parts.append(np.full((1, 3), np.nan, dtype=np.float32))
					color_parts.append(np.zeros((1, cols_arr.shape[1]), dtype=np.float32))
			return np.vstack(pos_parts), np.vstack(color_parts)

		core_pos, core_colors = _stack_polylines(self._tail_core_points, self._tail_core_colors)
		core_width = float(np.mean(self._tail_core_widths or [2.4]))
		_update_line_visual(self._tail_core_lines, core_pos, core_colors, core_width)

		overlay_pos, overlay_colors = _stack_polylines(self._tail_overlay_points, self._tail_overlay_colors)
		overlay_width = float(np.mean(self._tail_overlay_widths or [0.9]))
		_update_line_visual(self._tail_overlay_lines, overlay_pos, overlay_colors, overlay_width)

		whisker_primary_pos = _concat(self._whisker_primary_segments)
		whisker_primary_colors = _concat(self._whisker_primary_colors)
		primary_width = float(np.mean(self._whisker_primary_widths or [1.4]))
		_update_line_visual(self._whisker_lines_primary, whisker_primary_pos, whisker_primary_colors, primary_width)

		whisker_secondary_pos = _concat(self._whisker_secondary_segments)
		whisker_secondary_colors = _concat(self._whisker_secondary_colors)
		secondary_width = float(np.mean(self._whisker_secondary_widths or [1.0]))
		_update_line_visual(self._whisker_lines_secondary, whisker_secondary_pos, whisker_secondary_colors, secondary_width)

		self._layout_labels()

		self._refresh_hover_cache()
		self.request_draw()

	def request_draw(self) -> None:
		self.canvas.update()

	def capture_frame(self) -> np.ndarray:
		return self.canvas.render(alpha=False)

	def get_camera_override_rect(self) -> Optional[SceneRect]:
		return self._override_view_rect

	def get_current_camera_rect(self) -> Optional[SceneRect]:
		return self._current_view_rect

	def _handle_native_resize(self, width: Union[int, float], height: Union[int, float], device_pixel_ratio: float) -> None:
		logical_width = max(float(width), 1.0)
		logical_height = max(float(height), 1.0)
		self._device_pixel_ratio = max(float(device_pixel_ratio), 1.0)
		self._viewport_size = (logical_width, logical_height)
		self._update_reset_button_geometry(logical_width, logical_height)
		self._update_square_layout()
		self._apply_geometry()
		self._refresh_hover_cache()

	def _refresh_hover_cache(self) -> None:
		if not self._hover_datasets:
			return
		transform = self._scene_to_canvas_transform()
		for dataset in self._hover_datasets:
			if transform is None:
				dataset.screen_positions = None
				continue
			coords = self._map_scene_to_canvas(dataset.positions, transform=transform, divide_by_dpr=False)
			dataset.screen_positions = coords.astype(np.float32, copy=False)

	def _on_resize(self, event: Any) -> None:
		size = getattr(event, "size", None)
		if size is not None:
			width, height = size
			self._handle_native_resize(width, height, self._device_pixel_ratio)
		self.request_draw()

	def _on_mouse_move(self, event: Any) -> None:
		if not self._hover_datasets:
			return
		if event.pos is None:
			self._set_hover(None)
			return
		transform = self._scene_to_canvas_transform()
		if transform is None:
			self._set_hover(None)
			return
		pos = np.array(event.pos[:2], dtype=np.float32)
		closest: Optional[Tuple[HoverDataset, int, float]] = None
		for dataset in self._hover_datasets:
			positions = dataset.positions
			if positions.size == 0:
				continue
			screen_coords = self._map_scene_to_canvas(positions, transform=transform, divide_by_dpr=False)
			dataset.screen_positions = screen_coords.astype(np.float32, copy=False)
			if screen_coords.size == 0:
				continue
			deltas = screen_coords - pos
			dists = np.linalg.norm(deltas, axis=1)
			idx = int(np.argmin(dists))
			dist = float(dists[idx])
			if dist <= self._hover_threshold_px and (closest is None or dist < closest[2]):
				closest = (dataset, idx, dist)
		if closest is None:
			self._set_hover(None)
			return
		dataset, idx, _ = closest
		data_pos = dataset.positions[idx]
		label = dataset.labels[idx] if idx < len(dataset.labels) else f"point {idx}"
		self._set_hover(
			{
				"x": float(data_pos[0]),
				"y": float(data_pos[1]),
				"text": f"{dataset.mouse_id} · {label}",
			}
		)

	def _set_hover(self, payload: Optional[Dict[str, Any]]) -> None:
		if payload is None:
			self._hover_text.visible = False
			if self._hover_callback:
				self._hover_callback(None)
			self.request_draw()
			return

		self._hover_text.text = payload["text"]
		self._hover_text.pos = (payload["x"], payload["y"], 0.0)
		self._hover_text.visible = True
		if self._hover_callback:
			self._hover_callback(payload)
		self.request_draw()

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

	def _describe_mouse_event(self, event: Any) -> str:
		if event is None:
			return "event=None"
		event_type = getattr(event, "type", None) or getattr(event, "name", None) or type(event).__name__
		pos = getattr(event, "pos", None)
		handled = getattr(event, "handled", None)
		return f"event={event_type} pos={pos} handled={handled}"

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
		self._label_layout_dirty = True
		self._layout_labels()
		self._refresh_hover_cache()
		self.request_draw()
		self._emit_camera_change(source="user")
		print(
			"[PoseScene] _on_camera_transform_change applied rect={self._current_view_rect} override={self._override_view_rect}"
		)

	def _on_view_resize(self, event: Any) -> None:
		if self._in_view_resize:
			return
		self._in_view_resize = True
		try:
			self._update_square_layout()
			self._apply_geometry()
		finally:
			self._in_view_resize = False


def create_scene_canvas(
	parent: Optional[Any] = None,
	*,
	size: Tuple[int, int] = (820, 820),
	dpi: int = 110,
	antialias: int = 4,
) -> PoseScene:
	"""Create a GPU-accelerated scene canvas based on Vispy."""

	scene_controller = PoseScene(size=size, dpi=dpi, antialias=antialias)
	native = scene_controller.native_widget()
	if parent is not None and hasattr(native, "setParent"):
		native.setParent(parent)
	try:
		from PyQt6 import QtCore, QtWidgets  # type: ignore import
		if isinstance(native, QtWidgets.QWidget):
			native.setSizePolicy(
				QtWidgets.QSizePolicy.Policy.Expanding,
				QtWidgets.QSizePolicy.Policy.Expanding,
			)
			native.setMinimumSize(1, 1)

			device_ratio = float(native.devicePixelRatioF()) if hasattr(native, "devicePixelRatioF") else 1.0
			initial_width = native.width() or size[0]
			initial_height = native.height() or size[1]
			scene_controller._handle_native_resize(initial_width, initial_height, device_ratio)

			class _ResizeWatcher(QtCore.QObject):
				def __init__(self, scene_obj: PoseScene) -> None:
					super().__init__()
					self._scene = scene_obj

				def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # type: ignore[override]
					if event.type() == QtCore.QEvent.Type.Resize:
						size = event.size()
						ratio = float(obj.devicePixelRatioF()) if hasattr(obj, "devicePixelRatioF") else 1.0
						self._scene._handle_native_resize(size.width(), size.height(), ratio)
					elif event.type() in {
						QtCore.QEvent.Type.Show,
						QtCore.QEvent.Type.PolishRequest,
						getattr(QtCore.QEvent.Type, "DpiChanged", QtCore.QEvent.Type.PolishRequest),
						getattr(QtCore.QEvent.Type, "DpiChange", QtCore.QEvent.Type.PolishRequest),
					}:
						ratio = float(obj.devicePixelRatioF()) if hasattr(obj, "devicePixelRatioF") else 1.0
						self._scene._handle_native_resize(obj.width(), obj.height(), ratio)
					return False

			watcher = _ResizeWatcher(scene_controller)
			native.installEventFilter(watcher)
			scene_controller._qt_resize_watcher = watcher
	except Exception:
		pass
	return scene_controller


def initialise_viewer_theme(overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
	"""Vispy-based theme initialiser (kept for backwards compatibility)."""

	params: Dict[str, Any] = {
		"background": UI_BACKGROUND,
		"grid": (0.105, 0.149, 0.247, 0.28),
	}
	if overrides:
		params.update(dict(overrides))
	return params


def get_palette_color(index: int, palette: str = "tab20") -> Tuple[float, float, float]:
	if palette.lower() != "tab20":
		raise ValueError(f"Unsupported palette '{palette}' for Vispy renderer")
	return TAB20_RGB[index % len(TAB20_RGB)]


class _NoOpTemporaryRC:
	def __enter__(self) -> None:
		return None

	def __exit__(self, exc_type, exc, tb) -> None:
		return False


def temporary_rc(overrides: Mapping[str, Any]) -> _NoOpTemporaryRC:
	"""Compatibility shim for legacy Matplotlib rc-context usage."""

	return _NoOpTemporaryRC()


def is_interactive_backend() -> bool:
	return True


__all__ = [
	"PoseScene",
	"create_scene_canvas",
	"initialise_viewer_theme",
	"get_palette_color",
	"temporary_rc",
	"is_interactive_backend",
	"to_rgb",
	"to_rgba",
]
