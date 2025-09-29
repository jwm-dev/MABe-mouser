"""Geometry helpers mixin for the pose viewer application."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .visual_utils import to_rgb, to_rgba


class PoseViewerGeometryMixin:
    @staticmethod
    def _smooth_polyline(points: np.ndarray, iterations: int = 2) -> np.ndarray:
        if len(points) < 3 or iterations <= 0:
            return points
        result = points.astype(np.float32, copy=True)
        for _ in range(iterations):
            new_points: List[np.ndarray] = [result[0]]
            for i in range(len(result) - 1):
                p0 = result[i]
                p1 = result[i + 1]
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                new_points.extend([q, r])
            new_points.append(result[-1])
            result = np.asarray(new_points, dtype=np.float32)
        return result

    @staticmethod
    def _perp_unit(vector: np.ndarray) -> np.ndarray:
        perp = np.array([-vector[1], vector[0]], dtype=np.float32)
        norm = float(np.linalg.norm(perp))
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return perp / norm

    @staticmethod
    def _rotate_vector(vector: np.ndarray, angle_rad: float) -> np.ndarray:
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        x, y = float(vector[0]), float(vector[1])
        return np.array([cos_a * x - sin_a * y, sin_a * x + cos_a * y], dtype=np.float32)

    @staticmethod
    def _ray_hull_intersection(polygon: np.ndarray, origin: np.ndarray, target: np.ndarray) -> np.ndarray:
        if polygon is None or len(polygon) < 3:
            return origin
        origin = origin.astype(np.float32)
        target = target.astype(np.float32)
        best_point = origin
        best_t = 0.0
        for idx in range(len(polygon)):
            p1 = polygon[idx]
            p2 = polygon[(idx + 1) % len(polygon)]
            intersection = PoseViewerGeometryMixin._segment_intersection(origin, target, p1, p2)
            if intersection is None:
                continue
            point, t = intersection
            if t > best_t:
                best_t = t
                best_point = point
        return best_point

    @staticmethod
    def _project_point_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        start_f = start.astype(np.float32, copy=False)
        end_f = end.astype(np.float32, copy=False)
        point_f = point.astype(np.float32, copy=False)
        segment = end_f - start_f
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-9:
            return start_f.copy()
        t = float(np.dot(point_f - start_f, segment) / length_sq)
        t = float(np.clip(t, 0.0, 1.0))
        return (start_f + segment * t).astype(np.float32, copy=False)

    @staticmethod
    def _nearest_point_on_polygon(point: np.ndarray, polygon: Optional[np.ndarray]) -> Tuple[np.ndarray, int, float]:
        if polygon is None or len(polygon) == 0:
            return point.astype(np.float32, copy=False), 0, 0.0
        best_point = polygon[0].astype(np.float32, copy=True)
        best_dist = math.inf
        best_index = 0
        for idx in range(len(polygon)):
            p1 = polygon[idx]
            p2 = polygon[(idx + 1) % len(polygon)]
            candidate = PoseViewerGeometryMixin._project_point_to_segment(point, p1, p2)
            dist = float(np.linalg.norm(candidate - point))
            if dist < best_dist:
                best_dist = dist
                best_point = candidate.astype(np.float32, copy=True)
                best_index = idx
        return best_point, best_index, best_dist

    @staticmethod
    def _tail_base_connection(tail_points: np.ndarray, polygon: Optional[np.ndarray]) -> Tuple[np.ndarray, int]:
        if tail_points.size == 0:
            return np.zeros(2, dtype=np.float32), 0
        if polygon is None or len(polygon) < 2:
            return tail_points[0].astype(np.float32, copy=False), 0
        best_point = tail_points[0].astype(np.float32, copy=True)
        best_index = 0
        best_dist = math.inf
        for idx, tail_point in enumerate(tail_points):
            candidate, _, dist = PoseViewerGeometryMixin._nearest_point_on_polygon(tail_point, polygon)
            if dist < best_dist:
                best_dist = dist
                best_point = candidate.astype(np.float32, copy=True)
                best_index = idx
        return best_point, best_index

    @staticmethod
    def _nearest_tail_index_to_body(tail_points: np.ndarray, body_points: np.ndarray) -> int:
        if tail_points.size == 0:
            return 0
        if body_points.size == 0:
            return 0
        distances = np.linalg.norm(body_points[:, None, :] - tail_points[None, :, :], axis=2)
        if distances.size == 0:
            return 0
        nearest = np.min(distances, axis=0)
        return int(np.argmin(nearest))

    @staticmethod
    def _segment_intersection(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, float]]:
        r = b - a
        s = d - c
        denom = PoseViewerGeometryMixin._cross_2d(r, s)
        if abs(denom) < 1e-8:
            return None
        numerator_t = PoseViewerGeometryMixin._cross_2d(c - a, s)
        numerator_u = PoseViewerGeometryMixin._cross_2d(c - a, r)
        t = numerator_t / denom
        u = numerator_u / denom
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            point = a + t * r
            return point.astype(np.float32), float(t)
        return None

    @staticmethod
    def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    @staticmethod
    def _minimum_spanning_edges(points: np.ndarray) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        n = len(points)
        if n <= 1:
            return []
        indices = list(range(n))
        visited = {indices[0]}
        edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        while len(visited) < n:
            best_edge = None
            best_dist = math.inf
            for i in visited:
                for j in range(n):
                    if j in visited:
                        continue
                    d = float(distances[i, j])
                    if d < best_dist:
                        best_dist = d
                        best_edge = (i, j)
            if best_edge is None:
                break
            i, j = best_edge
            visited.add(j)
            edges.append(((float(points[i, 0]), float(points[i, 1])), (float(points[j, 0]), float(points[j, 1]))))
        return edges

    @staticmethod
    def _convex_hull(points: np.ndarray) -> Optional[np.ndarray]:
        unique_points = np.unique(points, axis=0)
        if len(unique_points) < 3:
            return None

        points_sorted = unique_points[np.lexsort((unique_points[:, 1], unique_points[:, 0]))]

        def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: List[np.ndarray] = []
        for p in points_sorted:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper: List[np.ndarray] = []
        for p in reversed(points_sorted):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = np.vstack((lower[:-1], upper[:-1]))
        return hull

    @staticmethod
    def _lighten_color(color: Tuple[float, float, float], amount: float) -> Tuple[float, float, float]:
        r, g, b = to_rgb(color)
        amount = np.clip(amount, 0.0, 1.0)
        r = r + (1.0 - r) * amount
        g = g + (1.0 - g) * amount
        b = b + (1.0 - b) * amount
        return (r, g, b)

    @staticmethod
    def _normalize_mouse_identifier(value: object) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            val = float(value)
            if math.isnan(val):
                return None
            return str(int(val))
        if pd.isna(value):  # type: ignore[arg-type]
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _rgba(color: Tuple[float, float, float], alpha: float) -> Tuple[float, float, float, float]:
        return to_rgba(color, alpha=np.clip(alpha, 0.0, 1.0))

    @staticmethod
    def _cubic_bezier_chain(points: np.ndarray, samples_per_segment: int = 18) -> np.ndarray:
        if len(points) <= 1:
            return points.astype(np.float32, copy=False)

        tangents = np.zeros_like(points, dtype=np.float32)
        for idx in range(len(points)):
            if idx == 0:
                tangents[idx] = points[1] - points[0]
            elif idx == len(points) - 1:
                tangents[idx] = points[-1] - points[-2]
            else:
                tangents[idx] = 0.5 * (points[idx + 1] - points[idx - 1])

        curve: List[np.ndarray] = [points[0].astype(np.float32, copy=True)]
        steps = max(4, samples_per_segment)
        for idx in range(len(points) - 1):
            p0 = points[idx]
            p3 = points[idx + 1]
            seg_len = float(np.linalg.norm(p3 - p0))
            if seg_len < 1e-6:
                continue
            t0 = PoseViewerGeometryMixin._limit_tangent(tangents[idx], seg_len)
            t1 = PoseViewerGeometryMixin._limit_tangent(tangents[idx + 1], seg_len)
            p1 = p0 + t0 / 3.0
            p2 = p3 - t1 / 3.0

            for t in np.linspace(0.0, 1.0, steps, endpoint=True)[1:]:
                point = (
                    (1.0 - t) ** 3 * p0
                    + 3.0 * (1.0 - t) ** 2 * t * p1
                    + 3.0 * (1.0 - t) * t ** 2 * p2
                    + t ** 3 * p3
                )
                curve.append(point.astype(np.float32, copy=False))

        return np.vstack(curve)

    @staticmethod
    def _limit_tangent(tangent: np.ndarray, segment_length: float) -> np.ndarray:
        tangent = tangent.astype(np.float32, copy=False)
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            return np.zeros_like(tangent, dtype=np.float32)
        max_length = segment_length * 0.45
        scale = min(1.0, max_length / norm)
        return tangent * scale


__all__ = ["PoseViewerGeometryMixin"]
