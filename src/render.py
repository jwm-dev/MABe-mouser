"""Rendering helpers shared between scene and playback logic."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .models import MouseGroup
from .scene_types import HoverDataset
from .visual_utils import _ensure_3d, _rgba


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
    base_rgb = _rgba(base_color, 1.0)[:3]
    hover_color = (
        float(base_rgb[0]),
        float(base_rgb[1]),
        float(base_rgb[2]),
    )
    self._hover_datasets.append(
        HoverDataset(
            positions=positions_scene[:, :2].astype(np.float32, copy=False),
            labels=label_tuple,
            mouse_id=mouse_id,
            color=hover_color,
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
    base_rgb = _rgba(base_color, 1.0)[:3]
    hover_color = (
        float(base_rgb[0]),
        float(base_rgb[1]),
        float(base_rgb[2]),
    )
    self._hover_datasets.append(
        HoverDataset(
            positions=positions_scene[:, :2].astype(np.float32, copy=False),
            labels=label_tuple,
            mouse_id=mouse_id,
            color=hover_color,
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


def split_tail_points(
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


def order_tail_sequence(
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


def build_tail_polyline(
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


def draw_whiskers(
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
    whiskers_per_side = 3
    side_angles = np.linspace(0.25, 0.68, whiskers_per_side, dtype=np.float32)
    length_factors = np.linspace(0.92, 1.08, whiskers_per_side, dtype=np.float32)
    accent_color = self._lighten_color(base_color, 0.55)
    underline_color = self._lighten_color(base_color, 0.2)
    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for angle, factor in zip(side_angles[::-1], length_factors[::-1]):
        oriented = self._rotate_vector(direction_unit, -float(angle))
        tip = nose_point + oriented * (whisker_length * float(factor))
        segments.append((nose_point.astype(np.float32), tip.astype(np.float32)))
    for angle, factor in zip(side_angles, length_factors):
        oriented = self._rotate_vector(direction_unit, float(angle))
        tip = nose_point + oriented * (whisker_length * float(factor))
        segments.append((nose_point.astype(np.float32), tip.astype(np.float32)))

    if segments:
        self._scene_add_whiskers(
            segments,
            primary_color=accent_color,
            secondary_color=underline_color,
        )


def draw_mouse_group(
    self,
    mouse_id: str,
    group: MouseGroup,
    base_color: Tuple[float, float, float],
) -> None:
    points = group.points
    if len(points) == 0:
        return

    core_points, core_labels, tail_points, tail_labels = split_tail_points(self, group)
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
        sorted_tail_points, sorted_tail_labels = order_tail_sequence(
            self,
            anchor_point,
            tail_points,
            tail_labels,
        )
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
        draw_whiskers(self, anchor_point, nose_point, base_color)

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
        polyline = build_tail_polyline(self, mouse_id, connection_point, sorted_tail_points)
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


__all__ = [
    "add_body_markers",
    "add_tail_markers",
    "add_body_edges",
    "add_tail_polyline",
    "add_whisker_segments",
    "draw_whiskers",
    "split_tail_points",
    "order_tail_sequence",
    "build_tail_polyline",
    "draw_mouse_group",
]
