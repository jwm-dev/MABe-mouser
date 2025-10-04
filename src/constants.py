"""UI and cache constants for the pose viewer application."""

from __future__ import annotations

CACHE_VERSION = "pose_viewer_v4"

UI_BACKGROUND = "#05060b"
UI_SURFACE = "#0b1020"
UI_ACCENT = "#66d9ff"
UI_TEXT_PRIMARY = "#dce9ff"
UI_TEXT_MUTED = "#7ca2d8"

# Behavior color palette for visualizations
BEHAVIOR_COLORS = {
    "attack": (1.0, 0.2, 0.2),      # Red
    "investigation": (0.2, 1.0, 0.2),  # Green
    "mount": (0.2, 0.5, 1.0),       # Blue
    "other": (1.0, 1.0, 0.2),       # Yellow
    "groom": (1.0, 0.2, 1.0),       # Magenta
    "sniff": (0.2, 1.0, 1.0),       # Cyan
    "walk": (1.0, 0.6, 0.2),        # Orange
    "run": (0.6, 0.2, 1.0),         # Purple
    "freeze": (0.2, 0.8, 0.4),      # Lime
    "rear": (0.8, 0.4, 0.2),        # Brown
}
