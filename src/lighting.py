"""Lighting helpers for Vispy-based pose rendering."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, TYPE_CHECKING

import numpy as np
from vispy.scene import visuals

from .visual_utils import _ensure_3d, _rgba, to_rgb

if TYPE_CHECKING:
    from .plotting import PoseScene


def initialise_lighting_visuals(scene: "PoseScene") -> None:
    """Initialise lighting-related visuals on the provided scene controller."""
    scene._glow_markers = visuals.Markers(parent=scene.view.scene)
    scene._glow_markers.set_gl_state(
        blend=True,
        depth_test=False,
        blend_func=("src_alpha", "one"),
    )
    scene._glow_markers.antialias = 4


def add_glow_markers(scene: "PoseScene", positions: np.ndarray, *, base_color: Sequence[float]) -> None:
    if positions.size == 0:
        return
    positions_scene = scene._to_scene_units_array(positions).astype(np.float32, copy=False)
    count = positions_scene.shape[0]
    if count == 0:
        return

    base_rgb = np.clip(np.array(to_rgb(base_color), dtype=np.float32), 0.0, 1.0)
    view_rect = scene._current_view_rect or scene._base_view_rect or scene._scene_rect
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
    scene_rect = scene._scene_rect or view_rect
    baseline_span = float(
        max(getattr(scene_rect, "width", span), getattr(scene_rect, "height", span), span)
    ) if scene_rect is not None else span
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
    layer_specs: Iterable[tuple[float, float, float]] = (
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
        scene._glow_positions.append(positions_scene)
        scene._glow_sizes.append(size_array)
        scene._glow_colors.append(color_array)


__all__ = [
    "initialise_lighting_visuals",
    "add_glow_markers",
]
