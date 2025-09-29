"""Status helpers and lifecycle mixin for the pose viewer application."""

from __future__ import annotations

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

    def _queue_progress_update(self, fraction: float, message: str) -> None:
        self._initialise_status_dispatcher()
        dispatcher = getattr(self, "_status_dispatcher", None)
        if dispatcher is None:
            return
        fraction = float(max(0.0, min(1.0, fraction)))
        dispatcher.progress.emit(fraction, str(message))

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
        self._set_progress(fraction * 100.0)
        self._set_status(message)

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
