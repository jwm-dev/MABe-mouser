"""Qt + Vispy UI construction mixin for the pose viewer application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

import numpy as np

from .analysis import AnalysisPane
from .bar import CustomTitleBar
from .constants import UI_ACCENT, UI_BACKGROUND, UI_SURFACE, UI_TEXT_MUTED, UI_TEXT_PRIMARY
from .plotting import PoseScene, create_scene_canvas


class BindingVar:
    """A minimal observable value helper mirroring Tk's Variable API."""

    def __init__(self, initial: Any, setter: Callable[[Any], None]) -> None:
        self._value = initial
        self._setter = setter
        self._setter(initial)

    def set(self, value: Any) -> None:
        self._value = value
        self._setter(value)

    def get(self) -> Any:
        return self._value


class BusySpinner(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, diameter: int = 44, line_width: float = 3.0, interval_ms: int = 80) -> None:
        super().__init__(parent)
        self._diameter = diameter
        self._line_width = line_width
        self._interval = max(16, int(interval_ms))
        self._angle = 0
        self._line_count = 12
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._advance)
        self.setMinimumSize(diameter, diameter)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start(self._interval)

    def stop(self) -> None:
        self._timer.stop()

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        self.start()
        super().showEvent(event)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:  # noqa: N802
        self.stop()
        super().hideEvent(event)

    def _advance(self) -> None:
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        center = self.rect().center()
        radius = min(self.width(), self.height()) / 2.0 - self._line_width
        if radius <= 0:
            return
        for i in range(self._line_count):
            angle = (360 / self._line_count) * i + self._angle
            opacity = 0.2 + (0.8 * i / self._line_count)
            color = QtGui.QColor(UI_ACCENT)
            color.setAlphaF(max(0.0, min(1.0, opacity)))
            painter.save()
            painter.translate(center)
            painter.rotate(angle)
            pen = QtGui.QPen(color)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            pen.setWidthF(self._line_width)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(0, -radius), QtCore.QPointF(0, -radius / 2.2))
            painter.restore()


class LoadingOverlay(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, *, message: str = "Loading…") -> None:
        super().__init__(parent)
        self._message = message
        self._spinner = BusySpinner(self, diameter=48)
        self._label = QtWidgets.QLabel(message, self)
        self._label.setStyleSheet("color: #f1f5ff; font-weight: 600; font-size: 14px;")
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)
        layout.addWidget(self._spinner, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(10)
        layout.addWidget(self._label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(2)
        self.setVisible(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAutoFillBackground(False)
        parent.installEventFilter(self)
        self._sync_geometry()

    def set_loading(self, active: bool, message: Optional[str] = None) -> None:
        if message:
            self._message = message
            self._label.setText(message)
        if active:
            self._spinner.start()
            self._sync_geometry()
            self.show()
            self.raise_()
        else:
            self._spinner.stop()
            self.hide()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if watched is self.parent() and event.type() in {
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.Show,
        }:
            self._sync_geometry()
        return super().eventFilter(watched, event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        gradient = QtGui.QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QtGui.QColor(13, 18, 32, 140))
        gradient.setColorAt(1.0, QtGui.QColor(13, 18, 32, 220))
        painter.fillRect(self.rect(), gradient)

    def _sync_geometry(self) -> None:
        parent = self.parent()
        if isinstance(parent, QtWidgets.QWidget):
            self.setGeometry(parent.rect())


class ClickableSlider(QtWidgets.QSlider):
    valueChangedInt = QtCore.pyqtSignal(int)

    def __init__(self, orientation: QtCore.Qt.Orientation, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(orientation, parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setTracking(True)
        self.valueChanged.connect(self.valueChangedInt.emit)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            self._jump_to(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._jump_to(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def _jump_to(self, position: QtCore.QPointF) -> None:
        if self.orientation() == QtCore.Qt.Orientation.Horizontal:
            span = max(self.width() - 1, 1)
            fraction = _clamp(position.x() / span, 0.0, 1.0)
        else:
            span = max(self.height() - 1, 1)
            fraction = 1.0 - _clamp(position.y() / span, 0.0, 1.0)
        value = self.minimum() + (self.maximum() - self.minimum()) * fraction
        self.setValue(int(round(value)))


class FloatSlider(ClickableSlider):
    valueChangedFloat = QtCore.pyqtSignal(float)

    def __init__(
        self,
        *,
        minimum: float,
        maximum: float,
        step: float = 0.1,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self._factor = max(1, int(round(1.0 / float(step))))
        self.setRange(int(round(minimum * self._factor)), int(round(maximum * self._factor)))
        self.valueChangedInt.connect(self._emit_float)
        self.setSingleStep(1)
        self.setPageStep(max(1, self._factor))
        self.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.setTickInterval(1)

    def _emit_float(self, value: int) -> None:
        self.valueChangedFloat.emit(value / self._factor)

    def setFloatValue(self, value: float) -> None:
        self.blockSignals(True)
        self.setValue(int(round(value * self._factor)))
        self.blockSignals(False)

    def floatValue(self) -> float:
        return self.value() / self._factor


class PoseViewerWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._close_callback: Optional[Callable[[], None]] = None
        self.setWindowTitle("MABe Pose Carousel Viewer")
        self.setMinimumSize(1040, 860)

    def register_close_callback(self, callback: Callable[[], None]) -> None:
        self._close_callback = callback

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self._close_callback:
            self._close_callback()
        event.accept()


_QT_APP: Optional[QtWidgets.QApplication] = None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _ensure_application() -> QtWidgets.QApplication:
    global _QT_APP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv or ["pose_viewer"])
        app.setStyle("Fusion")
        palette = app.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(UI_BACKGROUND))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(UI_SURFACE))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(UI_SURFACE))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(UI_TEXT_PRIMARY))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(UI_SURFACE))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(UI_TEXT_PRIMARY))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(UI_ACCENT))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(UI_BACKGROUND))
        app.setPalette(palette)
        _QT_APP = app
    return app


def create_tk_root() -> PoseViewerWindow:
    _ensure_application()
    return PoseViewerWindow()


def run_mainloop(root: PoseViewerWindow) -> None:
    app = _ensure_application()
    root.show()
    app.exec()


def ask_directory(**kwargs: Any) -> str:
    app = _ensure_application()
    parent = kwargs.get("parent")
    caption = kwargs.get("title", "Select Directory")
    initial_dir = kwargs.get("initialdir", "")
    path = QtWidgets.QFileDialog.getExistingDirectory(parent, caption, initial_dir)
    return str(path)


def ask_save_filename(**kwargs: Any) -> str:
    app = _ensure_application()
    parent = kwargs.get("parent")
    caption = kwargs.get("title", "Save As")
    initial_dir = kwargs.get("initialdir", "")
    initial_file = kwargs.get("initialfile", "")
    filter_entries = kwargs.get("filetypes", [])
    filters: List[str] = []
    for label, pattern in filter_entries:
        filters.append(f"{label} ({pattern})")
    selected_filter = ";;".join(filters) if filters else "All Files (*.*)"
    filename, _ = QtWidgets.QFileDialog.getSaveFileName(parent, caption, str(QtCore.QDir(initial_dir).filePath(initial_file)), selected_filter)
    return str(filename)


def show_info(title: str, message: str) -> None:
    _ensure_application()
    QtWidgets.QMessageBox.information(None, title, message)


def show_warning(title: str, message: str) -> None:
    _ensure_application()
    QtWidgets.QMessageBox.warning(None, title, message)


def show_error(title: str, message: str) -> None:
    _ensure_application()
    QtWidgets.QMessageBox.critical(None, title, message)


class PoseViewerUIMixin:
    def _initialise_ui_variables(self) -> None:
        self._status_value = "Ready"
        self._behavior_value = ""
        self._frame_value = 0
        self._speed_value = float(self.playback_speed_multiplier)
        self._progress_value = 0.0

    def _ask_directory(self, **kwargs: Any) -> str:
        return ask_directory(**kwargs)

    def _ask_save_filename(self, **kwargs: Any) -> str:
        return ask_save_filename(**kwargs)

    def _show_info(self, title: str, message: str) -> None:
        show_info(title, message)

    def _show_warning(self, title: str, message: str) -> None:
        show_warning(title, message)

    def _show_error(self, title: str, message: str) -> None:
        show_error(title, message)

    def _build_ui(self) -> None:
        def _style_flat_button(button: QtWidgets.QAbstractButton, *, padding: str = "6px 12px") -> None:
            button.setFlat(True)
            button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {UI_BACKGROUND};
                    color: {UI_TEXT_PRIMARY};
                    border: none;
                    padding: {padding};
                    border-radius: 6px;
                }}
                QPushButton:hover {{
                    background-color: rgba(102, 217, 255, 0.12);
                }}
                QPushButton:pressed,
                QPushButton:checked {{
                    background-color: rgba(102, 217, 255, 0.22);
                }}
                QPushButton:disabled {{
                    color: {UI_TEXT_MUTED};
                }}
                """
            )

        root: PoseViewerWindow = self.root
        central = QtWidgets.QWidget(root)
        root.setCentralWidget(central)
        root.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint, True)
        root.setWindowFlag(QtCore.Qt.WindowType.WindowSystemMenuHint, True)
        root.setStyleSheet(
            """
            QWidget { background: #0d1220; color: #f1f5ff; }
            QPushButton {
                background-color: #151d33;
                color: #e8ecfa;
                padding: 6px 12px;
                border: 1px solid #1f2a44;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #1c2844; }
            QPushButton:pressed { background-color: #111a32; }
            QPushButton:disabled { color: #4e5a74; border-color: #1a2336; }
            QLabel#TitleLabel, QToolButton#TitleLabel {
                font-weight: 600;
                font-size: 14px;
            }
            QToolButton#TitleLabel {
                color: #f1f5ff;
                background: transparent;
                border: none;
                padding: 2px 10px;
            }
            QToolButton#TitleLabel::menu-indicator { image: none; }
            QToolButton#TitleLabel:hover { color: #ffffff; }
            QLabel#BehaviorLabel { color: #95a3c4; font-size: 12px; }
            """
        )

        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.title_bar = CustomTitleBar(root)
        self.title_bar.open_action.triggered.connect(self._prompt_for_folder)
        self.title_bar.export_action.triggered.connect(self._export_current_video)
        self.export_action = self.title_bar.export_action
        self.export_button = self.title_bar.export_button
        self.title_bar.viewChanged.connect(self._handle_view_mode_change)
        self.title_bar.set_active_view("viewer")
        root.windowTitleChanged.connect(self.title_bar.set_title)
        main_layout.addWidget(self.title_bar)

        top_bar = QtWidgets.QFrame(central)
        top_layout = QtWidgets.QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 8, 12, 8)
        top_layout.setSpacing(10)

        self.prev_button = QtWidgets.QPushButton("◀", top_bar)
        self.prev_button.setFixedWidth(44)
        self.prev_button.clicked.connect(self._go_to_previous_file)
        _style_flat_button(self.prev_button, padding="6px 10px")
        top_layout.addWidget(self.prev_button)

        self.file_menu = QtWidgets.QMenu(top_bar)
        self.file_menu.setStyleSheet("QMenu { padding: 0; }")
        self.file_menu.setMinimumWidth(260)

        self.file_list_widget = QtWidgets.QListWidget(self.file_menu)
        self.file_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.file_list_widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.file_list_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.file_list_widget.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.file_list_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.file_list_widget.setUniformItemSizes(True)
        self.file_list_widget.setMaximumHeight(320)
        self.file_list_widget.itemClicked.connect(self._handle_file_menu_item_clicked)
        self.file_list_widget.itemActivated.connect(self._handle_file_menu_item_clicked)

        self.file_menu_action = QtWidgets.QWidgetAction(self.file_menu)
        self.file_menu_action.setDefaultWidget(self.file_list_widget)
        self.file_menu.addAction(self.file_menu_action)

        self.file_menu_button = QtWidgets.QToolButton(top_bar)
        self.file_menu_button.setObjectName("TitleLabel")
        self.file_menu_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.file_menu_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.file_menu_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.file_menu_button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.file_menu_button.setMenu(self.file_menu)
        self.file_menu_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._set_file_header_text(None)
        top_layout.addWidget(
            self.file_menu_button,
            stretch=1,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )

        self.next_button = QtWidgets.QPushButton("▶", top_bar)
        self.next_button.setFixedWidth(44)
        self.next_button.clicked.connect(self._go_to_next_file)
        _style_flat_button(self.next_button, padding="6px 10px")
        top_layout.addWidget(self.next_button)

        main_layout.addWidget(top_bar)

        self.content_stack = QtWidgets.QStackedWidget(central)
        self.content_stack.setObjectName("MainContentStack")
        main_layout.addWidget(self.content_stack, 1)

        viewer_container = QtWidgets.QWidget(self.content_stack)
        viewer_layout = QtWidgets.QVBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(0, 4, 0, 0)
        viewer_layout.setSpacing(8)

        figure_frame = QtWidgets.QFrame(viewer_container)
        figure_layout = QtWidgets.QVBoxLayout(figure_frame)
        figure_layout.setContentsMargins(0, 0, 0, 0)
        figure_layout.setSpacing(0)
        self.scene: PoseScene = create_scene_canvas(parent=figure_frame)
        canvas_widget = self.scene.native_widget()
        figure_layout.addWidget(canvas_widget, 1)
        viewer_layout.addWidget(figure_frame, 1)

        self.behavior_label = QtWidgets.QLabel("", viewer_container)
        self.behavior_label.setObjectName("BehaviorLabel")
        self.behavior_label.setContentsMargins(14, 6, 14, 6)
        viewer_layout.addWidget(self.behavior_label)

        controls_frame = QtWidgets.QFrame(viewer_container)
        controls_layout = QtWidgets.QGridLayout(controls_frame)
        controls_layout.setContentsMargins(12, 10, 12, 12)
        controls_layout.setHorizontalSpacing(10)
        controls_layout.setVerticalSpacing(10)

        self.play_button = QtWidgets.QPushButton("▶", controls_frame)
        self.play_button.setCheckable(True)
        self.play_button.setChecked(False)
        self.play_button.setFixedSize(40, 36)
        self.play_button.clicked.connect(self._toggle_play)
        _style_flat_button(self.play_button, padding="6px 12px")
        controls_layout.addWidget(self.play_button, 0, 0)

        self.prev_frame_button = QtWidgets.QPushButton("⏮", controls_frame)
        self.prev_frame_button.setFixedSize(36, 36)
        self.prev_frame_button.clicked.connect(self._step_backward)
        _style_flat_button(self.prev_frame_button, padding="6px 10px")
        controls_layout.addWidget(self.prev_frame_button, 0, 1)

        self.next_frame_button = QtWidgets.QPushButton("⏭", controls_frame)
        self.next_frame_button.setFixedSize(36, 36)
        self.next_frame_button.clicked.connect(self._step_forward)
        _style_flat_button(self.next_frame_button, padding="6px 10px")
        controls_layout.addWidget(self.next_frame_button, 0, 2)

        self.frame_slider = ClickableSlider(QtCore.Qt.Orientation.Horizontal, controls_frame)
        self.frame_slider.setRange(0, 1)
        self.frame_slider.valueChangedInt.connect(self._on_slider_change_from_qt)
        self.frame_slider.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        controls_layout.addWidget(self.frame_slider, 0, 3)

        self.time_label = QtWidgets.QLabel("00:00 / 00:00", controls_frame)
        self.time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        controls_layout.addWidget(self.time_label, 0, 4)

        speed_label = QtWidgets.QLabel("Speed:", controls_frame)
        controls_layout.addWidget(speed_label, 1, 0)

        self.speed_slider = FloatSlider(minimum=0.25, maximum=4.0, step=0.25, parent=controls_frame)
        self.speed_slider.valueChangedFloat.connect(self._on_speed_change_from_qt)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.setStyleSheet(
            f"""
            QSlider::groove:horizontal {{
                background: #121d32;
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: rgba(102, 217, 255, 0.35);
                border-radius: 3px;
            }}
            QSlider::add-page:horizontal {{
                background: #090f1c;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {UI_ACCENT};
                width: 18px;
                border-radius: 9px;
                margin: -6px 0;
            }}
            QSlider::handle:horizontal:hover {{
                background: #7fe3ff;
            }}
            QSlider::tick:horizontal {{
                background: transparent;
                border-left: 2px solid rgba(102, 217, 255, 0.42);
                height: 10px;
                margin: 2px 0;
            }}
            """
        )
        controls_layout.addWidget(self.speed_slider, 1, 1, 1, 3)

        self.speed_display = QtWidgets.QLabel("1.00×", controls_frame)
        self.speed_display.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        controls_layout.addWidget(self.speed_display, 1, 4)

        controls_layout.setColumnStretch(3, 1)
        controls_layout.setColumnStretch(4, 0)

        viewer_layout.addWidget(controls_frame)

        self.content_stack.addWidget(viewer_container)
        self.viewer_container = viewer_container
        self.viewer_overlay = LoadingOverlay(viewer_container, message="Loading viewer…")

        # Create analysis pane with executor and dataset stats
        executor = getattr(self, "_io_executor", None)
        dataset_stats = getattr(self, "_dataset_stats", None)
        on_main_thread = getattr(self, "_invoke_on_main_thread", None)
        
        # Create graphs pane for visual analysis
        from .graphs import GraphsPane
        self.graphs_pane = GraphsPane(
            parent=self.content_stack,
            executor=executor,
            dataset_stats=dataset_stats,
            on_main_thread=on_main_thread
        )
        
        # Connect signal to hide loading overlay
        self.graphs_pane.graphs_complete.connect(
            lambda: self._on_graphs_complete()
        )
        
        self.content_stack.addWidget(self.graphs_pane)
        
        # Create analysis pane for statistical analysis
        self.analysis_pane = AnalysisPane(
            parent=self.content_stack,
            executor=executor,
            dataset_stats=dataset_stats,
            on_main_thread=on_main_thread
        )
        
        # Connect signal to hide loading overlay
        self.analysis_pane.analysis_complete.connect(
            lambda: self._on_analysis_complete()
        )
        
        self.content_stack.addWidget(self.analysis_pane)
        self.content_stack.setCurrentWidget(viewer_container)
        self._active_view = "viewer"
        
        # Analysis tab state
        self._tables_visible = False
        self._tables_current_path: Optional[Path] = None
        self._tables_current_payload: Optional[Dict[str, object]] = None
        self._tables_dirty = True
        self._tables_prepared_payload: Optional[Dict[str, object]] = None
        self._analysis_loaded_path: Optional[Path] = None  # Track which file analysis has processed
        self._analysis_is_loading = False  # Track if analysis is currently loading
        self.tables_overlay = LoadingOverlay(self.analysis_pane, message="Preparing analysis…")
        
        # Graphs tab state  
        self._graphs_visible = False
        self._graphs_current_path: Optional[Path] = None
        self._graphs_current_payload: Optional[Dict[str, object]] = None
        self._graphs_dirty = True
        self._graphs_loaded_path: Optional[Path] = None
        self._graphs_is_loading = False
        self.graphs_overlay = LoadingOverlay(self.graphs_pane, message="Rendering graphs…")
        
        # Track preloading state
        self._preload_in_progress = False

        status_frame = QtWidgets.QFrame(central)
        status_layout = QtWidgets.QHBoxLayout(status_frame)
        status_layout.setContentsMargins(12, 8, 12, 10)
        status_layout.setSpacing(6)

        self.status_label = QtWidgets.QLabel(self._status_value, status_frame)
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        status_layout.addWidget(self.status_label, 1)

        main_layout.addWidget(status_frame)

        grip_container = QtWidgets.QWidget(central)
        grip_layout = QtWidgets.QHBoxLayout(grip_container)
        grip_layout.setContentsMargins(0, 0, 0, 0)
        grip_layout.addStretch(1)
        self.size_grip = QtWidgets.QSizeGrip(grip_container)
        self.size_grip.setFixedSize(18, 18)
        grip_layout.addWidget(self.size_grip)
        main_layout.addWidget(grip_container, 0)

        self.status_var = BindingVar(self._status_value, self.status_label.setText)
        self.behavior_var = BindingVar(self._behavior_value, self.behavior_label.setText)
        self.frame_var = BindingVar(self._frame_value, self._update_time_label)
        self.speed_var = BindingVar(self._speed_value, self._update_speed_display)

        self._update_file_menu()
        self._update_play_button()
        self._update_time_label(0)
        self.speed_slider.setFloatValue(self._speed_value)

    def _register_bindings(self) -> None:
        self._shortcuts: List[QtGui.QShortcut] = []
        shortcut_space = QtGui.QShortcut(QtGui.QKeySequence("Space"), self.root)
        shortcut_space.activated.connect(self._toggle_play)
        self._shortcuts.append(shortcut_space)

        shortcut_k = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_K), self.root)
        shortcut_k.activated.connect(self._toggle_play)
        self._shortcuts.append(shortcut_k)

        shortcut_left = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self.root)
        shortcut_left.activated.connect(self._step_backward)
        self._shortcuts.append(shortcut_left)

        shortcut_j = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_J), self.root)
        shortcut_j.activated.connect(self._step_backward)
        self._shortcuts.append(shortcut_j)

        shortcut_right = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self.root)
        shortcut_right.activated.connect(self._step_forward)
        self._shortcuts.append(shortcut_right)

        shortcut_l = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_L), self.root)
        shortcut_l.activated.connect(self._step_forward)
        self._shortcuts.append(shortcut_l)

        shortcut_reset = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_R), self.root)
        shortcut_reset.activated.connect(self._handle_reset_view_shortcut)
        self._shortcuts.append(shortcut_reset)

        shortcut_labels = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Q), self.root)
        shortcut_labels.activated.connect(self._toggle_labels)
        self._shortcuts.append(shortcut_labels)

        self.root.register_close_callback(self._on_close)

    def _on_slider_change_from_qt(self, value: int) -> None:
        if self.slider_active:
            return
        self._on_slider_change(str(float(value)))

    def _on_speed_change_from_qt(self, value: float) -> None:
        self._on_speed_change(str(value))

    def _update_speed_display(self, value: float) -> None:
        self.speed_display.setText(f"{value:.2f}×")
        if abs(self.speed_slider.floatValue() - value) > 1e-3:
            self.speed_slider.setFloatValue(value)

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _set_behavior(self, message: str) -> None:
        self.behavior_var.set(message)

    def _set_file_header_text(self, path: Path | None) -> None:
        button = getattr(self, "file_menu_button", None)
        if button is None:
            return
        if path is None:
            label = "Select parquet"
            tooltip = "Choose a parquet file from the current folder"
        else:
            label = path.name if isinstance(path, Path) else str(path)
            tooltip = str(path)
        button.setText(f"{label} ▾")
        button.setToolTip(tooltip)

    def _update_file_menu(self) -> None:
        button = getattr(self, "file_menu_button", None)
        list_widget = getattr(self, "file_list_widget", None)
        if button is None or list_widget is None:
            return

        files: List[Path] = list(getattr(self, "parquet_files", []))
        list_widget.blockSignals(True)
        list_widget.clear()

        if not files:
            button.setEnabled(False)
            self._set_file_header_text(None)
            placeholder = QtWidgets.QListWidgetItem("No parquet files loaded")
            placeholder.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            list_widget.addItem(placeholder)
            list_widget.blockSignals(False)
            return

        button.setEnabled(True)
        current_index = int(getattr(self, "current_file_index", 0))
        if current_index < 0 or current_index >= len(files):
            current_index = 0
            self.current_file_index = current_index
        self._set_file_header_text(files[current_index])

        for idx, path in enumerate(files):
            item = QtWidgets.QListWidgetItem(path.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, idx)
            item.setToolTip(str(path))
            list_widget.addItem(item)

        list_widget.setCurrentRow(current_index)
        current_item = list_widget.currentItem()
        if current_item is not None:
            list_widget.scrollToItem(current_item, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)

        list_widget.blockSignals(False)

    def _handle_file_menu_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        if item is None:
            return
        index = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(index, int):
            return
        menu = getattr(self, "file_menu", None)
        if isinstance(menu, QtWidgets.QMenu):
            menu.close()
        self._jump_to_file_index(int(index))

    def _jump_to_file_index(self, index: int) -> None:
        files: List[Path] = list(getattr(self, "parquet_files", []))
        if not files:
            return
        if index < 0 or index >= len(files):
            return
        if index == int(getattr(self, "current_file_index", 0)):
            return
        self.current_file_index = index
        self._load_current_file()

    def _set_frame_value(self, value: int) -> None:
        self.frame_var.set(int(value))
        if self.frame_slider.maximum() < value:
            self.frame_slider.setMaximum(value)
        if self.frame_slider.value() != value:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(value)
            self.frame_slider.blockSignals(False)

    def _set_speed_value(self, value: float) -> None:
        self.speed_var.set(value)

    def _update_play_button(self) -> None:
        if not hasattr(self, "play_button"):
            return
        self.play_button.blockSignals(True)
        self.play_button.setChecked(self.playing)
        self.play_button.setText("⏸" if self.playing else "▶")
        self.play_button.setToolTip("Pause" if self.playing else "Play")
        self.play_button.blockSignals(False)

    def _update_time_label(self, frame_value: int) -> None:  # noqa: ARG002 - value unused but retained for binding signature
        frame_index = max(0, int(self.frame_slider.value()))
        if self.playing:
            current_seconds = getattr(self, "_playback_time", self._frame_time_for_index(frame_index))
        else:
            current_seconds = self._frame_time_for_index(frame_index)
        total_seconds = float(getattr(self, "_frame_total_duration", 0.0) or 0.0)
        current_text = self._format_timestamp(current_seconds)
        total_text = self._format_timestamp(total_seconds)
        self.time_label.setText(f"{current_text} / {total_text}")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _connect_scene_hover(self, callback: Callable[[Optional[dict]], None]) -> None:
        self.scene.on_hover(callback)

    def _capture_canvas_rgb(self) -> Any:
        return self.scene.capture_frame()

    def _redraw_scene(self) -> None:
        self.scene.request_draw()

    def _update_canvas_limits(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
        self.scene.set_limits(xlim, ylim)

    def _update_canvas_labels(self, *, xlabel: str, ylabel: str, title: str) -> None:
        self.scene.set_axis_labels(xlabel=xlabel, ylabel=ylabel)
        self.scene.set_title(title)

    def _update_tables_tab(self, *, current_path: Path, data: Dict[str, object]) -> None:
        """Mark analysis and graphs tabs as needing refresh with new data."""
        # If analysis is currently loading and we're switching files, cancel it
        if self._analysis_is_loading and self._analysis_loaded_path != current_path:
            print(f"[UI] File changed during analysis, marking dirty and cancelling")
            self._analysis_is_loading = False
            self._set_tab_loading("tables", False, None)
            self._set_ui_enabled(True)
            
            # Cancel any pending analysis work
            pane = getattr(self, "analysis_pane", None)
            if pane is not None and hasattr(pane, "_update_future"):
                future = pane._update_future
                if future is not None:
                    try:
                        future.cancel()
                    except Exception:
                        pass
        
        # If graphs are currently loading and we're switching files, cancel
        if self._graphs_is_loading and self._graphs_loaded_path != current_path:
            print(f"[UI] File changed during graphs rendering, marking dirty and cancelling")
            self._graphs_is_loading = False
            self._set_tab_loading("graphs", False, None)
            self._set_ui_enabled(True)
        
        # Update analysis tab data
        self._tables_current_path = current_path
        self._tables_current_payload = data
        
        # Only mark dirty if this is a different file than what's currently loaded
        if self._analysis_loaded_path != current_path:
            self._tables_dirty = True
        
        # Update graphs tab data
        self._graphs_current_path = current_path
        self._graphs_current_payload = data
        
        # Only mark dirty if this is a different file than what's currently loaded
        if self._graphs_loaded_path != current_path:
            self._graphs_dirty = True
        
        # If analysis tab is visible and data changed, immediately update
        if getattr(self, "_tables_visible", False) and self._tables_dirty:
            self._refresh_tables_if_needed()
        
        # If graphs tab is visible and data changed, immediately update
        if getattr(self, "_graphs_visible", False) and self._graphs_dirty:
            self._refresh_graphs_if_needed()

    def _refresh_tables_if_needed(self) -> None:
        pane = getattr(self, "analysis_pane", None)
        if pane is None:
            return
        
        print(f"[UI] _refresh_tables_if_needed called, dirty={self._tables_dirty}, loading={self._analysis_is_loading}")
        
        # Don't start new load if already loading
        if self._analysis_is_loading:
            print(f"[UI] Already loading, skipping refresh")
            return
        
        # If not dirty, nothing to do
        if not self._tables_dirty:
            print(f"[UI] Not dirty, skipping refresh")
            return
        
        # Get stored data from _update_tables_tab
        current_path = getattr(self, "_tables_current_path", None)
        current_payload = getattr(self, "_tables_current_payload", None)
        
        if not current_payload or not isinstance(current_payload, dict):
            print(f"[UI] No valid payload, clearing analysis")
            pane.update_data(None, None)
            self._tables_dirty = False
            self._analysis_loaded_path = None
            return
        
        # Check if this data is already in smart cache - if so, no loading overlay needed
        from .smart_cache import get_cache_manager
        cache_mgr = get_cache_manager()
        payloads = current_payload.get("payloads", [])
        cache_key = cache_mgr.get_cache_key(path=current_path, params={'n_frames': len(payloads)})
        
        is_cached = cache_mgr.analysis_cache.get(cache_key) is not None
        print(f"[UI] Analysis cache check: {'HIT' if is_cached else 'MISS'} for {current_path}")
        
        # Only show loading if not cached
        if not is_cached:
            print(f"[UI] Showing loading overlay for uncached analysis")
            self._set_tab_loading("tables", True, "Analyzing data...")
            self._analysis_is_loading = True
            self._set_ui_enabled(False)
        
        print(f"[UI] Calling pane.update_data with path={current_path}")
        
        # Pass the data directly - it already has frames and metadata
        pane.update_data(current_path, current_payload)
        
        # Mark as loaded and clean
        self._tables_dirty = False
        self._analysis_loaded_path = current_path
    
    def _refresh_graphs_if_needed(self) -> None:
        pane = getattr(self, "graphs_pane", None)
        if pane is None:
            return
        
        print(f"[UI] _refresh_graphs_if_needed called, dirty={self._graphs_dirty}, loading={self._graphs_is_loading}")
        
        # Don't start new load if already loading
        if self._graphs_is_loading:
            print(f"[UI] Already loading graphs, skipping refresh")
            return
        
        # If not dirty, nothing to do
        if not self._graphs_dirty:
            print(f"[UI] Graphs not dirty, skipping refresh")
            return
        
        # Get stored data
        current_path = getattr(self, "_graphs_current_path", None)
        current_payload = getattr(self, "_graphs_current_payload", None)
        
        if not current_payload or not isinstance(current_payload, dict):
            print(f"[UI] No valid payload for graphs, clearing")
            pane.update_data(None, None)
            self._graphs_dirty = False
            self._graphs_loaded_path = None
            return
        
        # Graphs always need to render (visual output), so always show loading
        print(f"[UI] Showing loading overlay for graphs rendering")
        self._set_tab_loading("graphs", True, "Rendering graphs...")
        self._graphs_is_loading = True
        self._set_ui_enabled(False)
        
        print(f"[UI] Calling graphs_pane.update_data with path={current_path}")
        
        # Pass the data directly
        pane.update_data(current_path, current_payload)
        
        # Mark as loaded and clean
        self._graphs_dirty = False
        self._graphs_loaded_path = current_path

    def _handle_view_mode_change(self, key: str) -> None:
        if key == "graphs":
            self._show_graphs_panel()
            return
        if key == "tables":
            self._show_tables_panel()
            return
        self._show_viewer_panel()

    def _show_viewer_panel(self) -> None:
        # Pause playback when switching to viewer panel
        self._force_pause_playback()
        
        stack = getattr(self, "content_stack", None)
        viewer = getattr(self, "viewer_container", None)
        if stack is None or viewer is None:
            return
        stack.setCurrentWidget(viewer)
        self._active_view = "viewer"
        self._tables_visible = False
        self._graphs_visible = False
        title_bar = getattr(self, "title_bar", None)
        if title_bar is not None:
            title_bar.set_active_view("viewer")
    
    def _show_graphs_panel(self) -> None:
        # Pause playback when switching to graphs panel
        self._force_pause_playback()
        
        print(f"[UI] _show_graphs_panel called, dirty={getattr(self, '_graphs_dirty', False)}, loaded_path={getattr(self, '_graphs_loaded_path', None)}, current_path={getattr(self, '_graphs_current_path', None)}")
        
        stack = getattr(self, "content_stack", None)
        pane = getattr(self, "graphs_pane", None)
        if stack is None or pane is None:
            return
        
        # Switch to graphs panel immediately (no loading overlay on viewer)
        print(f"[UI] Switching to graphs panel")
        stack.setCurrentWidget(pane)
        self._active_view = "graphs"
        self._graphs_visible = True
        self._tables_visible = False
        
        # Update title bar
        title_bar = getattr(self, "title_bar", None)
        if title_bar is not None:
            title_bar.set_active_view("graphs")
        
        # Start the refresh if needed (this will handle loading overlay internally)
        self._refresh_graphs_if_needed()

    def _show_tables_panel(self) -> None:
        # Pause playback when switching to analysis panel
        self._force_pause_playback()
        
        print(f"[UI] _show_tables_panel called, dirty={getattr(self, '_tables_dirty', False)}, loaded_path={getattr(self, '_analysis_loaded_path', None)}, current_path={getattr(self, '_tables_current_path', None)}")
        
        stack = getattr(self, "content_stack", None)
        pane = getattr(self, "analysis_pane", None)
        if stack is None or pane is None:
            return
        
        # Switch to tables panel immediately (no loading overlay on viewer)
        print(f"[UI] Switching to tables panel")
        stack.setCurrentWidget(pane)
        self._active_view = "tables"
        self._tables_visible = True
        self._graphs_visible = False
        
        # Update title bar
        title_bar = getattr(self, "title_bar", None)
        if title_bar is not None:
            title_bar.set_active_view("tables")
        
        # Start the refresh if needed (this will handle loading overlay internally)
        self._refresh_tables_if_needed()
    
    def _get_current_tab_name(self) -> str:
        """Get the name of the currently active tab."""
        active_view = getattr(self, "_active_view", "viewer")
        return active_view if active_view in ("viewer", "graphs", "tables") else "viewer"

    def _set_tab_loading(self, tab: str, active: bool, message: Optional[str]) -> None:
        print(f"[UI] _set_tab_loading tab={tab}, active={active}, message={message}")
        overlay: Optional[LoadingOverlay] = None
        if tab == "viewer":
            overlay = getattr(self, "viewer_overlay", None)
        elif tab == "graphs":
            overlay = getattr(self, "graphs_overlay", None)
        elif tab == "tables":
            overlay = getattr(self, "tables_overlay", None)
        if overlay is not None:
            print(f"[UI] Setting overlay loading state: {active}")
            overlay.set_loading(active, message)
        else:
            print(f"[UI] Warning: No overlay found for tab {tab}")
    
    def _on_graphs_complete(self) -> None:
        """Handle graphs completion signal."""
        print(f"[UI] Graphs complete signal received, hiding loading overlay")
        self._graphs_is_loading = False
        self._set_tab_loading("graphs", False, None)
        self._set_ui_enabled(True)
    
    def _on_analysis_complete(self) -> None:
        """Handle analysis completion signal."""
        print(f"[UI] Analysis complete signal received, hiding loading overlay")
        self._analysis_is_loading = False
        self._set_tab_loading("tables", False, None)
        self._set_ui_enabled(True)
    
    def _refresh_graphs_if_needed(self) -> None:
        pane = getattr(self, "graphs_pane", None)
        if pane is None:
            return
        
        print(f"[UI] _refresh_graphs_if_needed called, dirty={self._graphs_dirty}, loading={self._graphs_is_loading}")
        
        # Don't start new load if already loading
        if self._graphs_is_loading:
            print(f"[UI] Already loading, skipping refresh")
            return
        
        # If not dirty, nothing to do
        if not self._graphs_dirty:
            print(f"[UI] Not dirty, skipping refresh")
            return
        
        # Get stored data
        current_path = getattr(self, "_graphs_current_path", None)
        current_payload = getattr(self, "_graphs_current_payload", None)
        
        if not current_payload or not isinstance(current_payload, dict):
            print(f"[UI] No valid payload, clearing graphs")
            pane.update_data(None, None)
            self._graphs_dirty = False
            self._graphs_loaded_path = None
            return
        
        # Graphs are visual - always show loading (no cache check needed)
        print(f"[UI] Showing loading overlay for graphs")
        self._set_tab_loading("graphs", True, "Rendering graphs...")
        self._graphs_is_loading = True
        self._set_ui_enabled(False)
        
        print(f"[UI] Calling graphs_pane.update_data with path={current_path}")
        
        # Pass the data directly
        pane.update_data(current_path, current_payload)
        
        # Mark as loaded and clean
        self._graphs_dirty = False
        self._graphs_loaded_path = current_path

    def _cancel_tables_future(self) -> None:
        tables_future = getattr(self, "_tables_future", None)
        if tables_future is not None:
            try:
                tables_future.cancel()
            except Exception:
                pass
    
    def _set_ui_enabled(self, enabled: bool) -> None:
        """Enable or disable UI interactions during loading states."""
        # Disable/enable navigation controls
        if hasattr(self, "prev_button"):
            self.prev_button.setEnabled(enabled)
        if hasattr(self, "next_button"):
            self.next_button.setEnabled(enabled)
        if hasattr(self, "file_menu_button"):
            self.file_menu_button.setEnabled(enabled)
        
        # Disable/enable playback controls
        if hasattr(self, "play_button"):
            self.play_button.setEnabled(enabled)
        if hasattr(self, "frame_slider"):
            self.frame_slider.setEnabled(enabled)
        if hasattr(self, "speed_slider"):
            self.speed_slider.setEnabled(enabled)
        
        # Keep view toggle always enabled so users can switch away from loading view
        # But update cursor to show busy state
        if enabled:
            self.root.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        else:
            self.root.setCursor(QtCore.Qt.CursorShape.WaitCursor)
        self._tables_future = None

    def _schedule_tables_payload_compile(self) -> None:
        """Deprecated - now handled directly by AnalysisPane."""
        pass

    def _apply_tables_payload(self, generation: int, payload: Dict[str, object]) -> None:
        """Deprecated - now handled directly by AnalysisPane."""
        pass

    def _handle_tables_error(self, exc: Exception) -> None:
        """Deprecated - now handled directly by AnalysisPane."""
        pass


    def _scene_begin_frame(
        self,
        *,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        aspect: float | None = None,
        domain_xlim: tuple[float | None, float | None] | None = None,
        domain_ylim: tuple[float | None, float | None] | None = None,
        preserve_view: bool = False,
    ) -> None:
        self.scene.begin_frame(
            xlim=xlim,
            ylim=ylim,
            aspect_ratio=aspect,
            domain_xlim=domain_xlim,
            domain_ylim=domain_ylim,
            preserve_view=preserve_view,
        )

    def _scene_finalize_frame(self) -> None:
        self.scene.finalise_frame()

    def _scene_add_glow(self, points: np.ndarray, color: tuple[float, float, float]) -> None:
        self.scene.add_glow_markers(points, base_color=color)

    def _scene_add_body(
        self,
        points: np.ndarray,
        *,
        base_color: tuple[float, float, float],
        labels: tuple[str, ...],
        mouse_id: str,
        edge_color: tuple[float, float, float],
    ) -> None:
        self.scene.add_body_markers(points, base_color=base_color, labels=labels, mouse_id=mouse_id, edge_color=edge_color, edge_width=1.4)

    def _scene_add_tail(
        self,
        points: np.ndarray,
        *,
        base_color: tuple[float, float, float],
        labels: tuple[str, ...],
        mouse_id: str,
        edge_color: tuple[float, float, float],
    ) -> None:
        self.scene.add_tail_markers(points, base_color=base_color, labels=labels, mouse_id=mouse_id, edge_color=edge_color, edge_width=0.8)

    def _scene_add_edges(self, segments: Any, *, color: tuple[float, float, float]) -> None:
        self.scene.add_body_edges(segments, color=color, width=2.0)

    def _scene_add_tail_polyline(
        self,
        polyline: np.ndarray,
        *,
        primary_color: tuple[float, float, float],
        secondary_color: tuple[float, float, float],
    ) -> None:
        self.scene.add_tail_polyline(polyline, primary_color=primary_color, primary_width=3.2, secondary_color=secondary_color, secondary_width=1.2)

    def _scene_add_whiskers(
        self,
        segments: Any,
        *,
        primary_color: tuple[float, float, float],
        secondary_color: tuple[float, float, float],
    ) -> None:
        self.scene.add_whisker_segments(segments, primary_color=primary_color, primary_width=1.4, secondary_color=secondary_color, secondary_width=1.0)

    def _scene_add_hull(self, polygon: np.ndarray, *, color: tuple[float, float, float]) -> None:
        self.scene.add_hull(polygon, color=color)

    def _scene_add_label(
        self,
        text: str,
        position: np.ndarray,
        points: np.ndarray,
        *,
        color: tuple[float, float, float],
        border_color: tuple[float, float, float],
    ) -> None:
        self.scene.add_label(text, position, points=points, color=color, border_color=border_color)


__all__ = [
    "PoseViewerUIMixin",
    "create_tk_root",
    "run_mainloop",
    "ask_directory",
    "ask_save_filename",
    "show_info",
    "show_warning",
    "show_error",
]
