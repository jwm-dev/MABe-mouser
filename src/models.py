"""Datamodels used by the pose viewer application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class MouseGroup:
    """Pose points and labels for a single mouse within one frame."""

    points: np.ndarray
    labels: Tuple[str, ...]


@dataclass(frozen=True)
class FramePayload:
    """Container for per-frame pose data."""

    frame_number: int
    mouse_groups: Dict[str, MouseGroup]
    behaviors: Dict[str, str] = field(default_factory=dict)
