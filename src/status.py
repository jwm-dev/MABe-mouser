"""Status helpers and lifecycle mixin for the pose viewer application."""

from __future__ import annotations

import time
from typing import Callable

from PyQt6 import QtCore, QtWidgets


class PoseViewerStatusMixin:
    class _StatusDispatcher(QtCore.QObject):
        progress = QtCore.pyqtSignal(float, str)
        invoke = QtCore.pyqtSignal(object)

    def _initialise_status_dispatcher(self) -> None:
        dispatcher = getattr(self, "_status_dispatcher", None)
        if dispatcher is not None:
            return
        dispatcher = PoseViewerStatusMixin._StatusDispatcher(parent=QtWidgets.QApplication.instance())
        dispatcher.progress.connect(self._apply_progress_update)
        dispatcher.invoke.connect(self._execute_on_main_thread)
        self._status_dispatcher = dispatcher
        self._last_progress_time = 0.0
        self._progress_throttle_interval = 0.1  # 10 updates per second max

    def _queue_progress_update(self, fraction: float, message: str) -> None:
        self._initialise_status_dispatcher()
        dispatcher = getattr(self, "_status_dispatcher", None)
        if dispatcher is None:
            return
        
        # Throttle progress updates to reduce UI thread congestion
        current_time = time.perf_counter()
        last_time = getattr(self, "_last_progress_time", 0.0)
        throttle = getattr(self, "_progress_throttle_interval", 0.1)
        
        # Always send 0.0 (start) and 1.0 (complete) updates
        is_significant = fraction <= 0.0 or fraction >= 1.0
        time_elapsed = current_time - last_time >= throttle
        
        if is_significant or time_elapsed:
            fraction = float(max(0.0, min(1.0, fraction)))
            dispatcher.progress.emit(fraction, str(message))
            self._last_progress_time = current_time

    def _invoke_on_main_thread(self, func: Callable[[], None]) -> None:
        if not callable(func):
            return
        self._initialise_status_dispatcher()
        dispatcher = getattr(self, "_status_dispatcher", None)
        if dispatcher is None:
            return
        dispatcher.invoke.emit(func)

    def _execute_on_main_thread(self, func: object) -> None:
        if callable(func):
            func()

    def _apply_progress_update(self, fraction: float, message: str) -> None:
        self._set_status(message)
        # Update loading overlay based on progress
        tab_loader = getattr(self, "_set_tab_loading", None)
        if callable(tab_loader):
            if fraction >= 1.0:
                tab_loader("viewer", False, None)
            else:
                # Show overlay for any loading state (including 0.0 = starting)
                tab_loader("viewer", True, message)

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _on_close(self) -> None:
        self._force_pause_playback()
        if self.loading_thread and self.loading_thread.is_alive():
            self.loading_thread.join(timeout=0.2)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.quit()


__all__ = ["PoseViewerStatusMixin"]
