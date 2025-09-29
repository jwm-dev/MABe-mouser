"""Qt + Vispy UI construction mixin for the pose viewer application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

import numpy as np

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
            QProgressBar {
                background: #111a30;
                border: 1px solid #1c2740;
                border-radius: 4px;
                padding: 1px;
            }
            QProgressBar::chunk { background: #2f86ff; border-radius: 4px; }
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

        figure_frame = QtWidgets.QFrame(central)
        figure_layout = QtWidgets.QVBoxLayout(figure_frame)
        figure_layout.setContentsMargins(0, 0, 0, 0)
        figure_layout.setSpacing(0)
        self.scene: PoseScene = create_scene_canvas(parent=figure_frame)
        canvas_widget = self.scene.native_widget()
        figure_layout.addWidget(canvas_widget, 1)
        main_layout.addWidget(figure_frame, 1)

        self.behavior_label = QtWidgets.QLabel("", central)
        self.behavior_label.setObjectName("BehaviorLabel")
        self.behavior_label.setContentsMargins(14, 6, 14, 6)
        main_layout.addWidget(self.behavior_label)

        controls_frame = QtWidgets.QFrame(central)
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

        main_layout.addWidget(controls_frame)

        status_frame = QtWidgets.QFrame(central)
        status_layout = QtWidgets.QVBoxLayout(status_frame)
        status_layout.setContentsMargins(12, 6, 12, 10)
        status_layout.setSpacing(6)

        self.status_label = QtWidgets.QLabel(self._status_value, status_frame)
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        status_layout.addWidget(self.status_label)

        self.progressbar = QtWidgets.QProgressBar(status_frame)
        self.progressbar.setRange(0, 100)
        self.progressbar.setValue(int(self._progress_value))
        self.progressbar.setTextVisible(False)
        status_layout.addWidget(self.progressbar)

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
        self.progress_var = BindingVar(self._progress_value, lambda value: self.progressbar.setValue(int(round(value))))

        self._update_file_menu()
        self._update_play_button()
        self._update_time_label(0)
        self.speed_slider.setFloatValue(self._speed_value)

    def _register_bindings(self) -> None:
        self._shortcuts: List[QtGui.QShortcut] = []
        shortcut_space = QtGui.QShortcut(QtGui.QKeySequence("Space"), self.root)
        shortcut_space.activated.connect(self._toggle_play)
        self._shortcuts.append(shortcut_space)

        shortcut_left = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self.root)
        shortcut_left.activated.connect(self._step_backward)
        self._shortcuts.append(shortcut_left)

        shortcut_right = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self.root)
        shortcut_right.activated.connect(self._step_forward)
        self._shortcuts.append(shortcut_right)

        shortcut_reset = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_R), self.root)
        shortcut_reset.activated.connect(self._handle_reset_view_shortcut)
        self._shortcuts.append(shortcut_reset)

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

    def _set_progress(self, value: float) -> None:
        self.progress_var.set(value)

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
        minutes, remainder = divmod(total_seconds, 60)
        return f"{minutes:02d}:{remainder:02d}"

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

    def _scene_add_trail(self, segments: Any, *, color: tuple[float, float, float]) -> None:
        self.scene.add_trail_segments(segments, color=color, width=2.2)

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
