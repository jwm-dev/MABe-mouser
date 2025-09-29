"""Shared dataclasses for scene and playback coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


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


__all__ = [
    "HoverDataset",
    "LabelDefinition",
    "SceneRect",
]
