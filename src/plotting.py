"""Vispy-based plotting utilities for the pose viewer application."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from vispy import app, scene
from vispy.geometry import PolygonData
from vispy.scene import visuals
from vispy.visuals.axis import MaxNLocator
from vispy.visuals.transforms import STTransform

try:  # pragma: no cover - GUI unavailable
	from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - PyQt not installed
	QtCore = None
	QtGui = None
	QtWidgets = None

from .camera import PoseSceneCameraMixin
from .constants import UI_ACCENT, UI_BACKGROUND, UI_SURFACE, UI_TEXT_MUTED, UI_TEXT_PRIMARY
from .lighting import add_glow_markers as lighting_add_glow_markers, initialise_lighting_visuals
from .optional_dependencies import qtawesome
from .scene_types import HoverDataset, LabelDefinition, SceneRect
from .visual_utils import ColorInput, _ensure_3d, _rgba, to_rgb, to_rgba


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
LABEL_MOUSE_PADDING = 18.0
VIDEO_LABEL_OFFSET = 12.0
ARENA_LABEL_OFFSET = 12.0
class PoseScene(PoseSceneCameraMixin):
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


		self._initialise_camera(width=float(width), height=float(height))

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
		self._video_rect: Optional[SceneRect] = None
		self._scene_rects_dirty = False
		self._overlay_center: Optional[Tuple[float, float]] = None
		self._data_translation: Tuple[float, float] = (0.0, 0.0)
		self._arena_size_cm: Optional[Tuple[float, float]] = None
		self._hover_callback: Optional[Callable[[Optional[Dict[str, Any]]], None]] = None
		self._hover_datasets: List[HoverDataset] = []
		self._hover_threshold_px = 16.0
		self._in_layout_update = False
		self._unit_scale = 1.0
		self._unit_label = "pixels"
		self._cm_per_pixel_hint: Optional[float] = None
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

		self._video_border_color = np.asarray(_rgba(UI_TEXT_PRIMARY, 0.45), dtype=np.float32)
		self._arena_border_color = np.asarray(_rgba(UI_ACCENT, 0.62), dtype=np.float32)
		self._video_border = visuals.Line(parent=self.view.scene, connect="segments")
		self._video_border.set_gl_state("translucent", depth_test=False)
		self._video_border.visible = False

		self._arena_border = visuals.Line(parent=self.view.scene, connect="segments")
		self._arena_border.set_gl_state("translucent", depth_test=False)
		self._arena_border.visible = False

		self._video_label_base_color = np.asarray(_rgba(UI_TEXT_PRIMARY, 0.88), dtype=np.float32)
		self._arena_label_base_color = np.asarray(_rgba(UI_ACCENT, 0.92), dtype=np.float32)
		self._video_label = visuals.Text(
			"",
			color=tuple(self._video_label_base_color),
			font_size=11,
			parent=self.view.scene,
			anchor_x="center",
			anchor_y="bottom",
		)
		self._video_label.visible = False
		self._arena_label = visuals.Text(
			"",
			color=tuple(self._arena_label_base_color),
			font_size=11,
			parent=self.view.scene,
			anchor_x="center",
			anchor_y="top",
		)
		self._arena_label.visible = False
		self._label_fade_thresholds = (0.45, 0.9)

		initialise_lighting_visuals(self)

		self._body_markers = visuals.Markers(parent=self.view.scene)
		self._body_markers.set_gl_state("translucent", depth_test=False)

		self._tail_markers = visuals.Markers(parent=self.view.scene)
		self._tail_markers.set_gl_state("translucent", depth_test=False)

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
		self._overlay_hover_sources: Set[str] = set()
		self._overlay_hover_filters: List[Any] = []

		self._reset_button: Optional[Any] = None
		self._reset_button_effect: Optional[Any] = None
		self._reset_button_callback: Optional[Callable[[], None]] = None
		self._reset_button_margin = 14.0
		self._labels_button: Optional[Any] = None
		self._labels_button_effect: Optional[Any] = None
		self._labels_enabled = True
		self._labels_button_spacing = 8.0
		self._initialise_reset_button()
		self._initialise_labels_button()

		self._hull_polygons: List[visuals.Polygon] = []
		self._label_texts: List[visuals.Text] = []
		self._label_bounds: List[Tuple[float, float, float, float]] = []
		self._label_definitions: List[LabelDefinition] = []
		self._label_layout_dirty = False
		self._label_font_size = 11.0
		self._label_text_pool: List[visuals.Text] = []
		self._label_layout_timer = app.Timer(
			interval=0.0,
			connect=self._on_label_layout_timer,
			start=False,
		)

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
		self._release_active_labels()
		self._label_bounds = []
		self._label_definitions = []
		self._label_layout_dirty = False
		try:
			self._label_layout_timer.stop()
		except Exception:
			pass

		self._hover_datasets = []

	def native_widget(self) -> Any:
		return self.canvas.native

	@property
	def unit_label(self) -> str:
		return self._unit_label

	def set_unit_scale(self, *, cm_per_pixel: Optional[float]) -> None:
		print(f"[PoseScene] set_unit_scale requested cm_per_pixel={cm_per_pixel}")
		current_scale = float(self._unit_scale)
		current_label = self._unit_label
		previous_hint = getattr(self, "_cm_per_pixel_hint", None)
		parsed_hint = None
		if cm_per_pixel is not None:
			try:
				candidate = float(cm_per_pixel)
			except (TypeError, ValueError):
				candidate = None
			if candidate is not None and math.isfinite(candidate) and candidate > 0.0:
				parsed_hint = candidate
		scale_changed = not math.isclose(current_scale, 1.0, rel_tol=1e-9, abs_tol=1e-9) or current_label != "pixels"
		hint_changed = not (
			(previous_hint is None and parsed_hint is None)
			or (
				previous_hint is not None
				and parsed_hint is not None
				and math.isclose(previous_hint, parsed_hint, rel_tol=1e-6, abs_tol=1e-9)
			)
		)
		self._unit_scale = 1.0
		self._unit_label = "pixels"
		self._cm_per_pixel_hint = parsed_hint
		if not scale_changed and not hint_changed:
			print("[PoseScene] set_unit_scale unchanged; retaining existing geometry")
			self._update_scale_bar()
			return
		if scale_changed:
			self._base_view_rect = None
			self._current_view_rect = None
			self._scene_rect = None
			self._user_camera_override = False
			self._override_view_rect = None
			self._scene_rects_dirty = True
			print("[PoseScene] set_unit_scale reset geometry to pixel domain")
		else:
			print("[PoseScene] set_unit_scale updated display hint without altering geometry")
		self._update_scale_bar()

	def _to_scene_units_scalar(self, value: Optional[float]) -> Optional[float]:
		if value is None:
			return None
		return float(value) * self._unit_scale

	def _to_scene_units_array(self, array: np.ndarray) -> np.ndarray:
		if array.size == 0:
			return array.astype(np.float32, copy=False)
		scaled = array.astype(np.float32, copy=False)
		scale_needed = not math.isclose(self._unit_scale, 1.0, rel_tol=1e-9, abs_tol=1e-9)
		offset_x, offset_y = self._data_translation
		translate_needed = not (
			math.isclose(offset_x, 0.0, abs_tol=1e-9)
			and math.isclose(offset_y, 0.0, abs_tol=1e-9)
		)
		if scale_needed or (translate_needed and scaled.ndim >= 1 and scaled.shape[-1] >= 2):
			scaled = scaled.copy()
		if scale_needed:
			if scaled.ndim >= 1 and scaled.shape[-1] >= 1:
				scaled[..., 0] *= self._unit_scale
			if scaled.ndim >= 1 and scaled.shape[-1] >= 2:
				scaled[..., 1] *= self._unit_scale
		if translate_needed and scaled.ndim >= 1 and scaled.shape[-1] >= 2:
			scaled[..., 0] += float(offset_x)
			scaled[..., 1] += float(offset_y)
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
		self._scene_rects_dirty = True
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
		arena_size_cm: Optional[Tuple[float, float]] = None,
	) -> None:
		geometry_changed = False
		overlay_dirty = False

		def _normalise_scene_pair(raw: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
			if raw is None or len(raw) < 2:
				return None
			try:
				tuple_val = (float(raw[0]), float(raw[1]))
			except (TypeError, ValueError):
				return None
			if not (
				math.isfinite(tuple_val[0])
				and math.isfinite(tuple_val[1])
				and tuple_val[0] > 0.0
				and tuple_val[1] > 0.0
			):
				return None
			converted_w = self._to_scene_units_scalar(tuple_val[0])
			converted_h = self._to_scene_units_scalar(tuple_val[1])
			if converted_w is None or converted_h is None:
				return (tuple_val[0], tuple_val[1])
			return (float(converted_w), float(converted_h))

		video_pair = _normalise_scene_pair(video_size)
		if video_pair is not None:
			if self._video_size != video_pair:
				self._video_size = video_pair
				geometry_changed = True
				overlay_dirty = True
				self._scene_rects_dirty = True
			elif self._video_rect is None:
				self._scene_rects_dirty = True
		elif video_size is None and (self._video_size is not None or self._video_rect is not None):
			self._video_size = None
			self._video_rect = None
			geometry_changed = True
			overlay_dirty = True
			self._scene_rects_dirty = True

		arena_pair = _normalise_scene_pair(arena_size)
		if arena_pair is not None:
			if self._arena_size != arena_pair:
				self._arena_size = arena_pair
				geometry_changed = True
				overlay_dirty = True
				self._scene_rects_dirty = True
			elif self._arena_rect is None:
				self._scene_rects_dirty = True
		elif arena_size is None and (self._arena_size is not None or self._arena_rect is not None):
			self._arena_size = None
			self._arena_rect = None
			geometry_changed = True
			overlay_dirty = True
			self._scene_rects_dirty = True

		arena_cm_pair = self._size_to_pair(arena_size_cm)
		if arena_cm_pair is not None:
			new_cm = (float(arena_cm_pair[0]), float(arena_cm_pair[1]))
			if self._arena_size_cm != new_cm:
				self._arena_size_cm = new_cm
				overlay_dirty = True
		elif arena_size_cm is None and self._arena_size_cm is not None:
			self._arena_size_cm = None
			overlay_dirty = True

		rects_changed = self._ensure_scene_rects()

		if geometry_changed or rects_changed:
			print(
				f"[PoseScene] set_scene_dimensions video={self._video_size} arena={self._arena_size} arena_cm={self._arena_size_cm} center={self._overlay_center}"
			)
			self._user_camera_override = False
			self._override_view_rect = None
			self._apply_geometry()
			self._update_layout_for_dimensions()
		if geometry_changed or overlay_dirty or rects_changed:
			self._update_area_overlays()

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

	@staticmethod
	def _rect_equal(lhs: Optional[SceneRect], rhs: Optional[SceneRect], *, tol: float = 1e-6) -> bool:
		if lhs is None or rhs is None:
			return lhs is None and rhs is None
		return (
			abs(lhs.x - rhs.x) <= tol
			and abs(lhs.y - rhs.y) <= tol
			and abs(lhs.width - rhs.width) <= tol
			and abs(lhs.height - rhs.height) <= tol
		)

	def _ensure_scene_rects(self) -> bool:
		if not getattr(self, "_scene_rects_dirty", False):
			return False
		prev_video = self._video_rect
		prev_arena = self._arena_rect
		prev_translation = getattr(self, "_data_translation", (0.0, 0.0))
		base_x_min = 0.0
		base_y_min = 0.0
		if self._domain_xlim is not None and self._domain_xlim[0] is not None:
			base_x_min = float(self._domain_xlim[0])
		elif self._data_rect is not None:
			base_x_min = float(self._data_rect.x)
		if self._domain_ylim is not None and self._domain_ylim[0] is not None:
			base_y_min = float(self._domain_ylim[0])
		elif self._data_rect is not None:
			base_y_min = float(self._data_rect.y)

		center_x: Optional[float] = None
		center_y: Optional[float] = None
		if self._video_size is not None:
			center_x = base_x_min + float(self._video_size[0]) * 0.5
			center_y = base_y_min + float(self._video_size[1]) * 0.5
		elif self._arena_size is not None:
			center_x = base_x_min + float(self._arena_size[0]) * 0.5
			center_y = base_y_min + float(self._arena_size[1]) * 0.5

		if center_x is None:
			if self._data_rect is not None:
				center_x = float(self._data_rect.center[0])
			elif self._domain_xlim is not None and self._domain_xlim[0] is not None and self._domain_xlim[1] is not None:
				center_x = (float(self._domain_xlim[0]) + float(self._domain_xlim[1])) * 0.5
			else:
				center_x = base_x_min
		if center_y is None:
			if self._data_rect is not None:
				center_y = float(self._data_rect.center[1])
			elif self._domain_ylim is not None and self._domain_ylim[0] is not None and self._domain_ylim[1] is not None:
				center_y = (float(self._domain_ylim[0]) + float(self._domain_ylim[1])) * 0.5
			else:
				center_y = base_y_min

		self._overlay_center = (center_x, center_y)

		if self._video_size is not None:
			video_w, video_h = self._video_size
			self._video_rect = SceneRect(center_x - video_w * 0.5, center_y - video_h * 0.5, video_w, video_h)
		else:
			self._video_rect = None
		if self._arena_size is not None:
			arena_w, arena_h = self._arena_size
			self._arena_rect = SceneRect(center_x - arena_w * 0.5, center_y - arena_h * 0.5, arena_w, arena_h)
		else:
			self._arena_rect = None

		self._scene_rects_dirty = False

		if self._video_rect is not None:
			offset_x = float(self._video_rect.x)
			offset_y = float(self._video_rect.y)
		elif self._arena_rect is not None:
			offset_x = float(self._arena_rect.x)
			offset_y = float(self._arena_rect.y)
		else:
			offset_x = float(base_x_min)
			offset_y = float(base_y_min)
		new_translation = (offset_x, offset_y)
		translation_changed = not (
			math.isclose(prev_translation[0], new_translation[0], rel_tol=1e-6, abs_tol=1e-9)
			and math.isclose(prev_translation[1], new_translation[1], rel_tol=1e-6, abs_tol=1e-9)
		)
		self._data_translation = new_translation

		changed = (
			not self._rect_equal(prev_video, self._video_rect)
			or not self._rect_equal(prev_arena, self._arena_rect)
			or translation_changed
		)
		return changed

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
		rects_changed = self._ensure_scene_rects()
		arena_rect = self._arena_rect
		video_rect = self._video_rect

		dom_x_min = self._domain_xlim[0] if self._domain_xlim else None
		dom_x_max = self._domain_xlim[1] if self._domain_xlim else None
		dom_y_min = self._domain_ylim[0] if self._domain_ylim else None
		dom_y_max = self._domain_ylim[1] if self._domain_ylim else None
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

		if data_rect is not None:
			min_x = float(data_rect.x)
			max_x = float(data_rect.x + data_rect.width)
			min_y = float(data_rect.y)
			max_y = float(data_rect.y + data_rect.height)
		elif video_rect is not None:
			min_x = float(video_rect.x)
			max_x = float(video_rect.x + video_rect.width)
			min_y = float(video_rect.y)
			max_y = float(video_rect.y + video_rect.height)
		elif arena_rect is not None:
			min_x = float(arena_rect.x)
			max_x = float(arena_rect.x + arena_rect.width)
			min_y = float(arena_rect.y)
			max_y = float(arena_rect.y + arena_rect.height)
		elif domain_rect is not None:
			min_x = float(domain_rect.x)
			max_x = float(domain_rect.x + domain_rect.width)
			min_y = float(domain_rect.y)
			max_y = float(domain_rect.y + domain_rect.height)
		else:
			min_x = 0.0
			max_x = 1.0
			min_y = 0.0
			max_y = 1.0

		def _expand_with_rect(rect: Optional[SceneRect]) -> None:
			if rect is None or rect.width <= 0.0 or rect.height <= 0.0:
				return
			nonlocal min_x, min_y, max_x, max_y
			min_x = min(min_x, float(rect.x))
			min_y = min(min_y, float(rect.y))
			max_x = max(max_x, float(rect.x + rect.width))
			max_y = max(max_y, float(rect.y + rect.height))

		_expand_with_rect(domain_rect)
		_expand_with_rect(video_rect)
		_expand_with_rect(arena_rect)
		_expand_with_rect(data_rect)

		if dom_x_min is not None:
			min_x = min(min_x, float(dom_x_min))
		if dom_x_max is not None:
			max_x = max(max_x, float(dom_x_max))
		elif video_rect is not None:
			max_x = max(max_x, float(video_rect.x + video_rect.width))
		if dom_y_min is not None:
			min_y = min(min_y, float(dom_y_min))
		if dom_y_max is not None:
			max_y = max(max_y, float(dom_y_max))
		elif video_rect is not None:
			max_y = max(max_y, float(video_rect.y + video_rect.height))

		min_x = float(min_x)
		max_x = float(max_x)
		min_y = float(min_y)
		max_y = float(max_y)

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
			arena_rect=arena_rect,
		)
		self._target_aspect_ratio = float(max(target_ratio, 1e-6))
		if self._target_aspect_ratio <= 0.0 or not math.isfinite(self._target_aspect_ratio):
			self._target_aspect_ratio = 1.0

		width = max(float(max_x - min_x), 1e-6)
		height = max(float(max_y - min_y), 1e-6)
		center_x = (min_x + max_x) * 0.5
		center_y = (min_y + max_y) * 0.5

		ratio = width / max(height, 1e-9)
		target_ratio = self._target_aspect_ratio
		if ratio < target_ratio:
			width = target_ratio * height
		elif ratio > target_ratio:
			height = width / target_ratio

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

		flip_x = self._x_axis_flipped
		flip_y = self._y_axis_flipped

		self._set_camera_rect(active_rect, flip_x=flip_x, flip_y=flip_y)
		if rects_changed:
			self._update_area_overlays()
		self._update_overlay_label_alpha()
		print(
			f"[PoseScene] _apply_geometry completed; current_rect={self._current_view_rect} override={self._override_view_rect}"
		)

	def _update_frame_border(self) -> None:
		target_rect = self._arena_rect or self._scene_rect or self._current_view_rect
		if target_rect is None or target_rect.width <= 0.0 or target_rect.height <= 0.0:
			self._frame_border.visible = False
			self._frame_border.set_data(pos=np.zeros((0, 3), dtype=np.float32))
			self._update_overlay_label_alpha()
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
		self._update_overlay_label_alpha()

	def _update_area_overlays(self) -> None:
		self._ensure_scene_rects()
		self._update_overlay_border(self._video_border, self._video_rect, self._video_border_color, width=1.3)
		self._update_overlay_border(self._arena_border, self._arena_rect, self._arena_border_color, width=1.6)
		self._update_overlay_labels()
		self._update_overlay_label_alpha()

	def _update_overlay_border(
		self,
		visual: visuals.Line,
		rect: Optional[SceneRect],
		color: np.ndarray,
		*,
		width: float,
	) -> None:
		if rect is None or rect.width <= 0.0 or rect.height <= 0.0:
			visual.visible = False
			visual.set_data(pos=np.zeros((0, 3), dtype=np.float32))
			return

		x_min = float(rect.x)
		x_max = float(rect.x + rect.width)
		y_min = float(rect.y)
		y_max = float(rect.y + rect.height)

		positions = np.array(
			[
				[x_min, y_max, 0.0],
				[x_max, y_max, 0.0],
				[x_max, y_min, 0.0],
				[x_min, y_min, 0.0],
			],
			dtype=np.float32,
		)
		indices = np.array(
			[
				[0, 1],
				[1, 2],
				[2, 3],
				[3, 0],
			],
			dtype=np.uint32,
		)
		color_rgba = np.tile(np.asarray(color, dtype=np.float32), (positions.shape[0], 1))
		visual.visible = True
		visual.set_data(pos=_ensure_3d(positions), color=color_rgba, connect=indices, width=float(width))

	def _update_overlay_labels(self) -> None:
		if self._video_rect is None or self._video_rect.width <= 0.0 or self._video_rect.height <= 0.0:
			self._video_label.text = ""
			self._video_label.visible = False
		else:
			rect = self._video_rect
			width_px = float(rect.width)
			height_px = float(rect.height)
			text = f"Video {width_px:.0f}px × {height_px:.0f}px"
			if self._video_label.text != text:
				self._video_label.text = text
			center_x = float(rect.x + rect.width * 0.5)
			top_y = float(rect.y + rect.height)
			self._video_label.pos = (
				center_x,
				top_y + VIDEO_LABEL_OFFSET,
				0.0,
			)
			self._video_label.visible = True
		if self._arena_rect is None or self._arena_rect.width <= 0.0 or self._arena_rect.height <= 0.0:
			self._arena_label.text = ""
			self._arena_label.visible = False
		else:
			rect = self._arena_rect
			width_px = float(rect.width)
			height_px = float(rect.height)
			parts = [f"Arena {width_px:.0f}px × {height_px:.0f}px"]
			if self._arena_size_cm is not None:
				cm_w, cm_h = self._arena_size_cm
				if math.isfinite(cm_w) and math.isfinite(cm_h) and cm_w > 0.0 and cm_h > 0.0:
					parts.append(f"({cm_w:.1f} cm × {cm_h:.1f} cm)")
			text = " ".join(parts)
			if self._arena_label.text != text:
				self._arena_label.text = text
			center_x = float(rect.x + rect.width * 0.5)
			bottom_y = float(rect.y)
			self._arena_label.pos = (
				center_x,
				bottom_y - ARENA_LABEL_OFFSET,
				0.0,
			)
			self._arena_label.visible = True

	def _apply_label_alpha(self, label: visuals.Text, base_color: np.ndarray, alpha: float) -> None:
		alpha_clamped = float(max(0.0, min(1.0, alpha)))
		if alpha_clamped <= 0.0 or not label.text:
			label.visible = False
			return
		color_rgba = np.asarray(base_color, dtype=np.float32).copy()
		color_rgba[3] = float(color_rgba[3] * alpha_clamped)
		label.color = tuple(color_rgba.tolist())
		label.visible = True

	def _update_overlay_label_alpha(self) -> None:
		view_rect = self._current_view_rect
		if view_rect is None or view_rect.width <= 0.0 or view_rect.height <= 0.0:
			self._apply_label_alpha(self._video_label, self._video_label_base_color, 0.0)
			self._apply_label_alpha(self._arena_label, self._arena_label_base_color, 0.0)
			return

		fade_out, fade_in = self._label_fade_thresholds
		fade_out = float(max(fade_out, 1e-6))
		fade_in = float(max(fade_in, fade_out + 1e-6))

		def _ratio_for(rect: Optional[SceneRect]) -> float:
			if rect is None or rect.width <= 0.0 or rect.height <= 0.0:
				return 0.0
			width_ratio = view_rect.width / max(rect.width, 1e-9)
			height_ratio = view_rect.height / max(rect.height, 1e-9)
			return float(min(width_ratio, height_ratio))

		def _alpha_from_ratio(ratio: float) -> float:
			if ratio <= fade_out:
				return 0.0
			if ratio >= fade_in:
				return 1.0
			return (ratio - fade_out) / (fade_in - fade_out)

		video_alpha = _alpha_from_ratio(_ratio_for(self._video_rect))
		arena_alpha = _alpha_from_ratio(_ratio_for(self._arena_rect))
		self._apply_label_alpha(self._video_label, self._video_label_base_color, video_alpha)
		self._apply_label_alpha(self._arena_label, self._arena_label_base_color, arena_alpha)

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
		self._update_overlay_label_alpha()

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

	def _canvas_point_to_scene(
		self,
		point: Sequence[float],
		*,
		transform: Optional[Any] = None,
	) -> Optional[Tuple[float, float]]:
		if transform is None:
			transform = self._scene_to_canvas_transform()
		if transform is None:
			return None
		try:
			mapped = transform.imap([float(point[0]), float(point[1]), 0.0])
		except Exception:
			return None
		result = np.asarray(mapped[:2], dtype=np.float64)
		if result.size < 2 or not np.all(np.isfinite(result[:2])):
			return None
		return (float(result[0]), float(result[1]))

	def _format_coordinate_text(self, x: float, y: float) -> str:
		unit = self._unit_label or "units"
		return f"x: {x:.2f} {unit}, y: {y:.2f} {unit}"

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
		def _valid_rect(candidate: Optional[SceneRect]) -> bool:
			return candidate is not None and candidate.width > 0.0 and candidate.height > 0.0

		video_rect = self._video_rect if _valid_rect(self._video_rect) else None
		if video_rect is not None:
			target_rect = video_rect
			grid_bounds = video_rect
		else:
			target_rect = rect if _valid_rect(rect) else None
			if target_rect is None:
				for fallback_rect in (self._arena_rect, self._scene_rect, rect):
					if _valid_rect(fallback_rect):
						target_rect = fallback_rect
						break
			if target_rect is None:
				self._grid_lines.visible = False
				self._grid_lines.set_data(pos=np.zeros((0, 3), dtype=np.float32))
				return
			if _valid_rect(self._arena_rect):
				grid_bounds = self._arena_rect  # fall back to legacy alignment when video data is unavailable
			elif _valid_rect(self._scene_rect):
				grid_bounds = self._scene_rect
			else:
				grid_bounds = target_rect
		try:
			scene_transform = self.view.scene.transform
			origin = scene_transform.map([target_rect.x, target_rect.y, 0.0])
			x_edge = scene_transform.map([target_rect.x + target_rect.width, target_rect.y, 0.0])
			y_edge = scene_transform.map([target_rect.x, target_rect.y + target_rect.height, 0.0])
			x_span_px = float(np.linalg.norm(x_edge[:2] - origin[:2]))
			y_span_px = float(np.linalg.norm(y_edge[:2] - origin[:2]))
		except Exception:
			x_span_px = float(self._viewport_size[0])
			y_span_px = float(self._viewport_size[1])
		x_pixels_per_unit = x_span_px / max(target_rect.width, 1e-9)
		y_pixels_per_unit = y_span_px / max(target_rect.height, 1e-9)

		def _dynamic_bins(pixels_per_unit: float, span_px: float) -> int:
			if not math.isfinite(pixels_per_unit) or pixels_per_unit <= 0.0 or not math.isfinite(span_px) or span_px <= 0.0:
				return 7
			base = max(span_px / 140.0, 5.0)
			zoom_bonus = max(0.0, math.log10(max(pixels_per_unit, 1.0))) * 5.0
			count = int(round(base + zoom_bonus))
			return int(max(5, min(24, count)))

		x_bins = _dynamic_bins(x_pixels_per_unit, x_span_px)
		y_bins = _dynamic_bins(y_pixels_per_unit, y_span_px)
		x_ticks = self._compute_tick_positions(target_rect.x, target_rect.x + target_rect.width, target_tick_count=x_bins)
		y_ticks = self._compute_tick_positions(target_rect.y, target_rect.y + target_rect.height, target_tick_count=y_bins)

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

		cm_hint = getattr(self, "_cm_per_pixel_hint", None)
		if cm_hint is not None and math.isfinite(cm_hint) and cm_hint > 0.0:
			centimetres = scale_units * cm_hint
			label = f"{_format_units(scale_units)} px  (≈ {centimetres:.2f} cm)"
		else:
			label = f"{_format_units(scale_units)} px"
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
			self._apply_labels_button_alpha(0.0)
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
		self._apply_labels_button_alpha(alpha)
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
		self._attach_overlay_hover_filter(button, "reset_button")
		self._update_reset_button_geometry(*self._viewport_size)

		if self._labels_button is not None:
			self._update_labels_button_icon()
			self._apply_labels_button_alpha(self._scale_bar_alpha)

	def _initialise_labels_button(self) -> None:
		if self._labels_button is not None:
			return
		if QtWidgets is None:
			return
		native = getattr(self.canvas, "native", None)
		if not isinstance(native, QtWidgets.QWidget):
			return
		button = QtWidgets.QToolButton(native)
		button.setObjectName("PoseSceneLabelsButton")
		button.setAutoRaise(True)
		if QtCore is not None:
			button.setIconSize(QtCore.QSize(22, 22))
			button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
			button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
		button.setCheckable(True)
		button.setChecked(self._labels_enabled)
		button.setToolTip("Toggle labels (Q)")
		button.setStyleSheet(
			"QToolButton#PoseSceneLabelsButton {"
			" background-color: rgba(17, 26, 48, 170);"
			" border: 1px solid rgba(90, 114, 168, 160);"
			" border-radius: 16px;"
			" padding: 6px;"
			" }"
			" QToolButton#PoseSceneLabelsButton:hover {"
			" background-color: rgba(28, 40, 68, 210);"
			" }"
			" QToolButton#PoseSceneLabelsButton:pressed {"
			" background-color: rgba(12, 20, 38, 230);"
			" }"
		)
		button.setFixedSize(34, 34)
		self._labels_button = button
		self._update_labels_button_icon()
		effect = QtWidgets.QGraphicsOpacityEffect(button)
		effect.setOpacity(0.0)
		button.setGraphicsEffect(effect)
		button.hide()
		button.clicked.connect(self._on_labels_button_clicked)
		button.raise_()
		self._labels_button_effect = effect
		self._attach_overlay_hover_filter(button, "labels_button")
		self._update_reset_button_geometry(*self._viewport_size)
		self._apply_labels_button_alpha(self._scale_bar_alpha)

	def _update_labels_button_icon(self) -> None:
		button = self._labels_button
		if button is None:
			return
		keys_on = ("fa5s.tags", "fa.tags", "mdi.tag-multiple")
		keys_off = ("fa5s.tag", "fa.tag", "mdi.tag")
		icon: Optional[Any] = None
		if qtawesome is not None:
			candidates = keys_on if self._labels_enabled else keys_off
			for key in candidates:
				try:
					icon = qtawesome.icon(key, color="#f4f7ff")
					break
				except Exception:
					continue
		if icon is not None:
			button.setIcon(icon)
			button.setText("")
		else:
			if QtGui is not None:
				button.setIcon(QtGui.QIcon())
			button.setText("LBL" if self._labels_enabled else "OFF")
		if QtCore is not None:
			button.setChecked(self._labels_enabled)

	def _update_reset_button_geometry(self, canvas_w: float, canvas_h: float) -> None:
		button = self._reset_button
		label_button = self._labels_button
		if button is None and label_button is None:
			return
		if canvas_w <= 0.0 or canvas_h <= 0.0:
			return
		margin = float(self._reset_button_margin)
		spacing = float(self._labels_button_spacing)
		base_y = int(max(margin, margin))
		reset_width = button.width() if button is not None else 0
		reset_height = button.height() if button is not None else 0
		if button is not None:
			if reset_width <= 0:
				reset_width = button.sizeHint().width()
			if reset_height <= 0:
				reset_height = button.sizeHint().height()
		reset_x = int(max(margin, canvas_w - margin - reset_width)) if button is not None else 0
		if button is not None:
			button.move(reset_x, base_y)
		label_width = 0
		label_height = 0
		if label_button is not None:
			label_width = label_button.width()
			label_height = label_button.height()
			if label_width <= 0:
				label_width = label_button.sizeHint().width()
			if label_height <= 0:
				label_height = label_button.sizeHint().height()
			label_x = canvas_w - margin - label_width if button is None else reset_x - spacing - label_width
			label_x = int(max(margin, label_x))
			label_y = base_y if button is None else base_y + max(0, (reset_height - label_height) // 2)
			label_button.move(label_x, label_y)
			label_button.raise_()
		if button is not None:
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

	def _apply_labels_button_alpha(self, alpha: float) -> None:
		button = self._labels_button
		effect = self._labels_button_effect
		if button is None or effect is None:
			return
		clamped = float(max(0.0, min(1.0, alpha)))
		effect.setOpacity(clamped)
		is_visible = clamped > 0.02
		button.setVisible(is_visible)
		button.setEnabled(clamped >= 0.35)
		if QtCore is not None:
			button.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, clamped <= 0.02)
		button.setChecked(self._labels_enabled)

	def _on_reset_button_clicked(self) -> None:
		button = self._reset_button
		effect = self._reset_button_effect
		if button is None or effect is None:
			return
		if effect.opacity() <= 0.02 or not button.isVisible():
			return
		self._invoke_reset_callback()

	def _on_labels_button_clicked(self) -> None:
		button = self._labels_button
		effect = self._labels_button_effect
		if button is None or effect is None:
			return
		if effect.opacity() <= 0.02 or not button.isVisible():
			return
		self._toggle_labels_enabled()

	def _invoke_reset_callback(self) -> None:
		callback = self._reset_button_callback
		if callable(callback):
			callback()
		else:
			self.reset_camera_view()

	def _toggle_labels_enabled(self) -> None:
		self._set_labels_enabled(not self._labels_enabled)

	def _set_labels_enabled(self, enabled: bool) -> None:
		enabled_flag = bool(enabled)
		if self._labels_enabled == enabled_flag:
			return
		self._labels_enabled = enabled_flag
		self._update_labels_button_icon()
		if self._labels_button is not None:
			self._labels_button.setChecked(enabled_flag)
		self._request_label_layout(delay=0.0)
		self.request_draw()

	def set_labels_visible(self, enabled: bool) -> None:
		self._set_labels_enabled(enabled)

	def labels_visible(self) -> bool:
		return self._labels_enabled

	def toggle_labels_visible(self) -> None:
		self._toggle_labels_enabled()

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
			if self._overlay_hover_sources:
				self._cancel_scale_bar_fade()
				self._set_scale_bar_alpha(1.0)
		self._scale_bar_click_armed = False

	def _scale_bar_contains_point(self, x: float, y: float) -> bool:
		rect = self._scale_bar_hit_rect
		if rect is None:
			return False
		left, bottom, right, top = rect
		return left <= x <= right and bottom <= y <= top

	def _set_overlay_hover_state(self, source: str, active: bool) -> None:
		source_key = str(source)
		if active:
			if source_key in self._overlay_hover_sources:
				return
			self._overlay_hover_sources.add(source_key)
		else:
			if source_key not in self._overlay_hover_sources:
				return
			self._overlay_hover_sources.remove(source_key)
		mode = getattr(self, "_scale_bar_mode", "auto")
		if mode != "auto":
			return
		if self._overlay_hover_sources:
			if not self._scale_bar_geometry_ready:
				self._update_scale_bar()
			self._cancel_scale_bar_fade()
			self._set_scale_bar_alpha(1.0)
		else:
			self._schedule_scale_bar_fade(self._scale_bar_idle_delay)

	def _update_scale_bar_hover_from_pos(self, pos: Optional[Sequence[float]]) -> None:
		if pos is None:
			self._set_overlay_hover_state("scale_bar", False)
			return
		try:
			x_val = float(pos[0])
			y_val = float(pos[1])
		except (TypeError, ValueError, IndexError):
			self._set_overlay_hover_state("scale_bar", False)
			return
		inside = self._scale_bar_contains_point(x_val, y_val)
		self._set_overlay_hover_state("scale_bar", inside)

	def _attach_overlay_hover_filter(self, widget: Any, source: str) -> None:
		if QtCore is None or widget is None:
			return
		try:
			widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
		except Exception:
			pass
		source_key = str(source)
		owner = self

		class _OverlayHoverFilter(QtCore.QObject):
			def eventFilter(self, obj: Optional[QtCore.QObject], event: Optional[QtCore.QEvent]) -> bool:  # type: ignore[override]
				if event is None:
					return False
				etype = event.type()
				if etype in (
					QtCore.QEvent.Type.Enter,
					QtCore.QEvent.Type.HoverEnter,
				):
					owner._set_overlay_hover_state(source_key, True)
				elif etype in (
					QtCore.QEvent.Type.Leave,
					QtCore.QEvent.Type.HoverLeave,
				):
					owner._set_overlay_hover_state(source_key, False)
				elif etype in (
					QtCore.QEvent.Type.Hide,
					QtCore.QEvent.Type.Close,
				):
					owner._set_overlay_hover_state(source_key, False)
				return False

		filter_obj = _OverlayHoverFilter(widget)
		widget.installEventFilter(filter_obj)
		self._overlay_hover_filters.append(filter_obj)

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
		if event_type in {"mouse_move", "mouse_press", "mouse_release"}:
			self._update_scale_bar_hover_from_pos(pos)
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

	add_glow_markers = lighting_add_glow_markers

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
		base_rgba = _rgba(base_color, 1.0)
		self._hover_datasets.append(
			HoverDataset(
				positions=positions_scene[:, :2].astype(np.float32, copy=False),
				labels=label_tuple,
				mouse_id=mouse_id,
				color=(float(base_rgba[0]), float(base_rgba[1]), float(base_rgba[2])),
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
		base_rgba = _rgba(base_color, 1.0)
		self._hover_datasets.append(
			HoverDataset(
				positions=positions_scene[:, :2].astype(np.float32, copy=False),
				labels=label_tuple,
				mouse_id=mouse_id,
				color=(float(base_rgba[0]), float(base_rgba[1]), float(base_rgba[2])),
			)
		)

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
		self._request_label_layout()

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

	def _scene_scale_at_point(
		self,
		point: Tuple[float, float],
		*,
		transform: Optional[Any] = None,
	) -> Tuple[float, float]:
		if transform is None:
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

	def _acquire_label_visual(self, definition: LabelDefinition) -> visuals.Text:
		if self._label_text_pool:
			label = self._label_text_pool.pop()
			label.text = definition.text
			label.color = _rgba(definition.color, 0.95)
			label.font_size = float(self._label_font_size)
			label.parent = self.view.scene
			label.visible = True
			return label
		return visuals.Text(
			definition.text,
			color=_rgba(definition.color, 0.95),
			font_size=float(self._label_font_size),
			parent=self.view.scene,
		)

	def _release_active_labels(self) -> None:
		if not self._label_texts:
			return
		for text in self._label_texts:
			try:
				text.visible = False
				text.parent = None
			except Exception:
				pass
			self._label_text_pool.append(text)
		self._label_texts.clear()

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
		self._request_label_layout()

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

	def _layout_single_label(
		self,
		definition: LabelDefinition,
		*,
		transform: Optional[Any] = None,
		protected_regions: Optional[List[Tuple[float, float, float, float]]] = None,
		current_index: int = -1,
	) -> None:
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
		label = self._acquire_label_visual(definition)
		font_size = float(getattr(label, "font_size", self._label_font_size) or self._label_font_size)
		extent_px = self._estimate_text_extent(definition.text, font_size)
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
				scale_x, scale_y = self._scene_scale_at_point((float(label_xy[0]), float(label_xy[1])), transform=transform)
				padding_scene = LABEL_OVERLAP_PADDING / max(min(scale_x, scale_y), 1e-6)
				extent_scene = (extent_px[0] / scale_x, extent_px[1] / scale_y)
				bounds = self._bounds_from_anchor((float(label_xy[0]), float(label_xy[1])), (anchor_x, anchor_y), extent_scene, padding_scene)
				if not self._bounds_in_view(bounds):
					continue
				if any(self._bounds_overlap(bounds, existing) for existing in self._label_bounds):
					continue
				if protected_regions is not None and 0 <= current_index < len(protected_regions):
					blocked = False
					for region_index, region in enumerate(protected_regions):
						if region_index == current_index:
							continue
						if self._bounds_overlap(bounds, region):
							blocked = True
							break
					if blocked:
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
			fallback_pos = (float(anchor_xy[0]), float(anchor_xy[1]), 0.0)
			label.pos = fallback_pos
			scale_x, scale_y = self._scene_scale_at_point((float(anchor_xy[0]), float(anchor_xy[1])), transform=transform)
			padding_scene = LABEL_OVERLAP_PADDING / max(min(scale_x, scale_y), 1e-6)
			extent_scene = (extent_px[0] / scale_x, extent_px[1] / scale_y)
			best_bounds = self._bounds_from_anchor(fallback_pos[:2], fallback_anchor, extent_scene, padding_scene)
			if protected_regions is not None and 0 <= current_index < len(protected_regions):
				for region_index, region in enumerate(protected_regions):
					if region_index == current_index:
						continue
					if self._bounds_overlap(best_bounds, region):
						expand = LABEL_MOUSE_PADDING / max(min(scale_x, scale_y), 1e-6)
						direction = (base_direction / max(float(np.linalg.norm(base_direction)), 1e-6)).astype(np.float32, copy=False)
						label_offset = anchor_xy + direction * (offset + expand)
						adjusted_pos = (float(label_offset[0]), float(label_offset[1]), 0.0)
						label.pos = adjusted_pos
						best_position = adjusted_pos
						best_bounds = self._bounds_from_anchor(adjusted_pos[:2], fallback_anchor, extent_scene, padding_scene)
						break
			best_position = fallback_pos if best_position is None else best_position
			best_anchor = fallback_anchor
		if best_position is None:
			best_position = (float(anchor_xy[0]), float(anchor_xy[1]), 0.0)
		if best_bounds is None:
			scale_x, scale_y = self._scene_scale_at_point((float(best_position[0]), float(best_position[1])), transform=transform)
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
		line_color = np.tile(_rgba(definition.color, 0.32), (2, 1)).astype(np.float32, copy=False)
		self._label_line_segments.append(line_segment)
		self._label_line_colors.append(line_color)

	def _on_label_layout_timer(self, event: Any) -> None:
		try:
			self._label_layout_timer.stop()
		except Exception:
			pass
		if self._label_layout_dirty:
			self._layout_labels()

	def _request_label_layout(self, *, delay: float = 0.0) -> None:
		self._label_layout_dirty = True
		timer = getattr(self, "_label_layout_timer", None)
		if timer is None:
			self._layout_labels()
			return
		try:
			timer.stop()
		except Exception:
			pass
		interval = float(delay)
		if interval <= 0.0:
			interval = 1.0 / 120.0
		timer.start(interval)

	def _layout_labels(self) -> None:
		timer = getattr(self, "_label_layout_timer", None)
		if timer is not None:
			try:
				timer.stop()
			except Exception:
				pass
		if not self._label_layout_dirty:
			return
		self._release_active_labels()
		self._label_bounds.clear()
		self._label_line_segments.clear()
		self._label_line_colors.clear()
		if not self._labels_enabled:
			self._label_lines.visible = False
			self._label_lines.set_data(
				pos=np.zeros((0, 3), dtype=np.float32),
				color=np.zeros((0, 4), dtype=np.float32),
				width=0.0,
			)
			self._label_layout_dirty = False
			return
		if not self._label_definitions:
			self._label_lines.visible = False
			self._label_lines.set_data(pos=np.zeros((0, 3), dtype=np.float32), color=np.zeros((0, 4), dtype=np.float32), width=0.0)
			self._label_layout_dirty = False
			return
		transform = self._scene_to_canvas_transform()
		protected_regions: List[Tuple[float, float, float, float]] = []
		for definition in self._label_definitions:
			points = definition.points
			if points.size == 0:
				points = definition.anchor.reshape(1, 2)
			x_values = points[:, 0].astype(np.float64, copy=False)
			y_values = points[:, 1].astype(np.float64, copy=False)
			anchor_point = (float(definition.anchor[0]), float(definition.anchor[1]))
			scale_x, scale_y = self._scene_scale_at_point(anchor_point, transform=transform)
			padding_scene = LABEL_MOUSE_PADDING / max(min(scale_x, scale_y), 1e-6)
			x_min = float(np.min(x_values)) - padding_scene
			y_min = float(np.min(y_values)) - padding_scene
			x_max = float(np.max(x_values)) + padding_scene
			y_max = float(np.max(y_values)) + padding_scene
			protected_regions.append((x_min, y_min, x_max, y_max))
		for index, definition in enumerate(self._label_definitions):
			self._layout_single_label(
				definition,
				transform=transform,
				protected_regions=protected_regions,
				current_index=index,
			)
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
		pos_value = getattr(event, "pos", None)
		self._update_scale_bar_hover_from_pos(pos_value)
		if not self._hover_datasets:
			return
		if pos_value is None:
			self._set_hover(None)
			return
		transform = self._scene_to_canvas_transform()
		if transform is None:
			self._set_hover(None)
			return
		pos = np.array(pos_value[:2], dtype=np.float32)
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
			scene_point = self._canvas_point_to_scene(pos, transform=transform)
			if scene_point is None:
				self._set_hover(None)
				return
			x_val, y_val = scene_point
			bounds = self._video_rect
			if bounds is None or bounds.width <= 0.0 or bounds.height <= 0.0:
				arena = self._arena_rect
				if arena is not None and arena.width > 0.0 and arena.height > 0.0:
					bounds = arena
			if bounds is not None:
				if not (
					bounds.x - 1e-6 <= x_val <= bounds.x + bounds.width + 1e-6
					and bounds.y - 1e-6 <= y_val <= bounds.y + bounds.height + 1e-6
				):
					self._set_hover(None)
					return
			elif not self._point_in_view((x_val, y_val)):
				self._set_hover(None)
				return
			coords_text = self._format_coordinate_text(x_val, y_val)
			self._set_hover(
				{
					"x": float(x_val),
					"y": float(y_val),
					"status": f"Cursor — {coords_text}",
					"show_text": False,
				}
			)
			return
		dataset, idx, _ = closest
		data_pos = dataset.positions[idx]
		label = dataset.labels[idx] if idx < len(dataset.labels) else f"point {idx}"
		coords_text = self._format_coordinate_text(float(data_pos[0]), float(data_pos[1]))
		self._set_hover(
			{
				"x": float(data_pos[0]),
				"y": float(data_pos[1]),
				"text": f"{dataset.mouse_id} · {label}",
				"color": getattr(dataset, "color", (1.0, 1.0, 1.0)),
				"status": f"{dataset.mouse_id} · {label} — {coords_text}",
			}
		)

	def _set_hover(self, payload: Optional[Dict[str, Any]]) -> None:
		if payload is None:
			self._hover_text.visible = False
			self._hover_text.color = _rgba(UI_ACCENT, 0.94)
			if self._hover_callback:
				self._hover_callback(None)
			self.request_draw()
			return

		show_text = bool(payload.get("show_text", True))
		if show_text:
			color_value = payload.get("color")
			if color_value is not None:
				self._hover_text.color = _rgba(color_value, 0.94)
			else:
				self._hover_text.color = _rgba(UI_ACCENT, 0.94)
			text_value = str(payload.get("text", ""))
			self._hover_text.text = text_value
			self._hover_text.pos = (payload["x"], payload["y"], 0.0)
			self._hover_text.visible = bool(text_value)
		else:
			self._hover_text.visible = False
			self._hover_text.text = ""
			self._hover_text.color = _rgba(UI_ACCENT, 0.94)
		if self._hover_callback:
			self._hover_callback(payload)
		self.request_draw()

	def _describe_mouse_event(self, event: Any) -> str:
		if event is None:
			return "event=None"
		event_type = getattr(event, "type", None) or getattr(event, "name", None) or type(event).__name__
		pos = getattr(event, "pos", None)
		handled = getattr(event, "handled", None)
		return f"event={event_type} pos={pos} handled={handled}"



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


def get_palette_color(index: int, palette: str = "tab20") -> Tuple[float, float, float]:
	if palette.lower() != "tab20":
		raise ValueError(f"Unsupported palette '{palette}' for Vispy renderer")
	return TAB20_RGB[index % len(TAB20_RGB)]

__all__ = [
	"PoseScene",
	"PoseSceneCameraMixin",
	"create_scene_canvas",
	"get_palette_color",
	"to_rgb",
	"to_rgba",
]
