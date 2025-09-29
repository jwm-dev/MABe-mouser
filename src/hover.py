"""Hover interaction mixin for the pose viewer application."""

from __future__ import annotations

from typing import Any, Dict, Optional


class PoseViewerHoverMixin:
    _hover_prev_status: Optional[str] = None

    def _handle_hover_payload(self, payload: Optional[Dict[str, Any]]) -> None:
        if payload:
            if self._hover_prev_status is None:
                try:
                    self._hover_prev_status = str(self.status_var.get())
                except Exception:
                    self._hover_prev_status = None
            display = payload.get("status")
            if display is None:
                x_val = float(payload.get("x", 0.0))
                y_val = float(payload.get("y", 0.0))
                coords = f"x: {x_val:.2f}, y: {y_val:.2f}"
                label = payload.get("text")
                if label:
                    display = f"{label} — {coords}"
                else:
                    display = coords
            self._set_status(display)
        else:
            if self._hover_prev_status is not None:
                self._set_status(self._hover_prev_status)
            self._hover_prev_status = None


__all__ = ["PoseViewerHoverMixin"]
