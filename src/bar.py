"""Custom in-app title bar used by the pose viewer UI."""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

import qtawesome as qta

from .constants import UI_SURFACE, UI_TEXT_MUTED, UI_TEXT_PRIMARY

def _cat_icon_pixmap(size: int = 28) -> Optional[QtGui.QPixmap]:
	"""Return a Qt pixmap for the Font Awesome cat icon using qtawesome."""

	icon_variants = [
		"fa6s.cat",
		"fa5s.cat",
		"fa.cat",
	]
	for key in icon_variants:
		try:
			icon = qta.icon(key)
		except Exception:
			continue
		if icon is not None and not icon.isNull():
			pixmap = icon.pixmap(size, size, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
			if not pixmap.isNull():
				return pixmap
	return None


class TitleActionButton(QtWidgets.QToolButton):
	def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
		super().__init__(parent)
		self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
		self.setAutoRaise(True)
		self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
		self.setFixedHeight(32)
		self.setIconSize(QtCore.QSize(16, 16))


class WindowControlButton(QtWidgets.QToolButton):
	def __init__(self, symbol: str, tooltip: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
		super().__init__(parent)
		self.setAutoRaise(True)
		self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
		self.setFixedSize(34, 28)
		self.setText(symbol)
		self.setToolTip(tooltip)
		font = QtGui.QFont(self.font())
		font.setPointSize(12)
		self.setFont(font)


class CustomTitleBar(QtWidgets.QFrame):
	"""A frameless window title bar with themed styling and actions."""

	def __init__(self, window: QtWidgets.QWidget) -> None:
		super().__init__(window)
		self._window = window
		self._drag_offset = QtCore.QPoint()

		self.setObjectName("CustomTitleBar")
		self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
		self.setFixedHeight(48)
		self.setStyleSheet(
			f"""
			QFrame#CustomTitleBar {{
				background: {UI_SURFACE};
				color: {UI_TEXT_PRIMARY};
				border-bottom: 1px solid rgba(47, 134, 255, 0.1);
			}}
			QToolButton {{
				background: transparent;
				color: {UI_TEXT_PRIMARY};
				padding: 4px 10px;
				border-radius: 6px;
			}}
			QToolButton#WindowMinButton,
			QToolButton#WindowMaxButton,
			QToolButton#WindowCloseButton {{
				background: {UI_SURFACE};
				border: none;
				padding: 4px 6px;
				border-radius: 6px;
			}}
			QToolButton:hover {{
				background: rgba(47, 134, 255, 0.16);
				color: {UI_TEXT_PRIMARY};
			}}
			QToolButton:pressed {{
				background: rgba(47, 134, 255, 0.24);
			}}
			QToolButton:disabled {{
				color: {UI_TEXT_MUTED};
			}}
			QToolButton#WindowMinButton:hover,
			QToolButton#WindowMaxButton:hover {{
				background: rgba(102, 217, 255, 0.18);
			}}
			QToolButton#WindowCloseButton:hover {{
				background: rgba(255, 88, 88, 0.25);
			}}
			QToolButton#WindowCloseButton:pressed {{
				background: rgba(255, 88, 88, 0.4);
			}}
			QLabel#TitleBarTitle {{
				font-size: 13px;
				font-weight: 500;
				color: {UI_TEXT_PRIMARY};
			}}
			QLabel#TitleBarSubtitle {{
				font-size: 11px;
				color: {UI_TEXT_MUTED};
			}}
			"""
		)

		layout = QtWidgets.QHBoxLayout(self)
		layout.setContentsMargins(16, 8, 16, 8)
		layout.setSpacing(12)

		self.icon_label = QtWidgets.QLabel(self)
		self.icon_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		self.icon_label.setFixedSize(32, 32)
		self._apply_cat_icon()
		layout.addWidget(self.icon_label)

		title_container = QtWidgets.QVBoxLayout()
		title_container.setContentsMargins(0, 0, 0, 0)
		title_container.setSpacing(2)

		self.title_label = QtWidgets.QLabel(window.windowTitle(), self)
		self.title_label.setObjectName("TitleBarTitle")
		self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
		title_container.addWidget(self.title_label)

		self.subtitle_label = QtWidgets.QLabel("", self)
		self.subtitle_label.setObjectName("TitleBarSubtitle")
		self.subtitle_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
		self.subtitle_label.setVisible(False)
		title_container.addWidget(self.subtitle_label)

		text_container = QtWidgets.QWidget(self)
		text_container.setLayout(title_container)
		layout.addWidget(text_container)

		layout.addStretch(1)

		self.open_action = QtGui.QAction("Open Folder…", self)
		self.export_action = QtGui.QAction("Export…", self)
		self.export_action.setEnabled(True)

		self.open_button = TitleActionButton(self)
		self.open_button.setDefaultAction(self.open_action)
		layout.addWidget(self.open_button)

		self.export_button = TitleActionButton(self)
		self.export_button.setDefaultAction(self.export_action)
		layout.addWidget(self.export_button)

		layout.addSpacing(8)

		controls_container = QtWidgets.QHBoxLayout()
		controls_container.setContentsMargins(0, 0, 0, 0)
		controls_container.setSpacing(2)

		self.min_button = WindowControlButton("–", "Minimize", self)
		self.min_button.setObjectName("WindowMinButton")
		self.min_button.clicked.connect(self._window.showMinimized)
		controls_container.addWidget(self.min_button)

		self.max_button = WindowControlButton("⬜", "Maximize", self)
		self.max_button.setObjectName("WindowMaxButton")
		self.max_button.clicked.connect(self._toggle_max_restore)
		controls_container.addWidget(self.max_button)

		self.close_button = WindowControlButton("✕", "Close", self)
		self.close_button.setObjectName("WindowCloseButton")
		self.close_button.clicked.connect(self._window.close)
		controls_container.addWidget(self.close_button)

		controls_widget = QtWidgets.QWidget(self)
		controls_widget.setLayout(controls_container)
		layout.addWidget(controls_widget)

		window.installEventFilter(self)

	def _apply_cat_icon(self) -> None:
		pixmap = _cat_icon_pixmap(28)
		if pixmap is not None and not pixmap.isNull():
			self.icon_label.setPixmap(pixmap)
			self.icon_label.setText("")
			return

		self.icon_label.clear()
		self.icon_label.setText("🐈")
		font = QtGui.QFont(self.icon_label.font())
		font.setPointSize(20)
		self.icon_label.setFont(font)

	def set_title(self, title: str) -> None:
		self.title_label.setText(title)

	def set_subtitle(self, subtitle: str) -> None:
		self.subtitle_label.setText(subtitle)
		self.subtitle_label.setVisible(bool(subtitle))

	def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
		if watched is self._window and event.type() == QtCore.QEvent.Type.WindowStateChange:
			self._update_max_button()
		return super().eventFilter(watched, event)

	def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
		if event.button() == QtCore.Qt.MouseButton.LeftButton:
			self._drag_offset = event.globalPosition().toPoint() - self._window.frameGeometry().topLeft()
			event.accept()
		super().mousePressEvent(event)

	def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
		if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
			new_pos = event.globalPosition().toPoint() - self._drag_offset
			self._window.move(new_pos)
			event.accept()
			return
		super().mouseMoveEvent(event)

	def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
		if event.button() == QtCore.Qt.MouseButton.LeftButton:
			self._toggle_max_restore()
			event.accept()
			return
		super().mouseDoubleClickEvent(event)

	def _toggle_max_restore(self) -> None:
		if self._window.isMaximized():
			self._window.showNormal()
		else:
			self._window.showMaximized()
		self._update_max_button()

	def _update_max_button(self) -> None:
		if self._window.isMaximized():
			self.max_button.setText("❐")
			self.max_button.setToolTip("Restore")
		else:
			self.max_button.setText("⬜")
			self.max_button.setToolTip("Maximize")


__all__ = ["CustomTitleBar"]
