"""Lazy widget creation for deferred UI initialization."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from PyQt6 import QtWidgets


class LazyWidget(QtWidgets.QWidget):
    """
    Widget that creates its content only when first shown.
    
    Significantly improves startup time by deferring creation
    of complex widgets until they're actually needed.
    """
    
    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        creator: Optional[Callable[[], QtWidgets.QWidget]] = None
    ):
        """
        Initialize lazy widget.
        
        Args:
            parent: Parent widget
            creator: Function that creates the actual widget content
        """
        super().__init__(parent)
        
        self._creator = creator
        self._content_widget: Optional[QtWidgets.QWidget] = None
        self._initialized = False
        
        # Create placeholder layout
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        # Add loading indicator
        self._loading_label = QtWidgets.QLabel("Loading...", self)
        self._loading_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._loading_label)
    
    def set_creator(self, creator: Callable[[], QtWidgets.QWidget]) -> None:
        """Set or update the widget creator function."""
        self._creator = creator
        if self.isVisible() and not self._initialized:
            self._initialize_content()
    
    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Initialize content when widget is first shown."""
        super().showEvent(event)
        if not self._initialized and self._creator:
            self._initialize_content()
    
    def _initialize_content(self) -> None:
        """Create and display the actual widget content."""
        if self._initialized or not self._creator:
            return
        
        try:
            # Create the content widget
            self._content_widget = self._creator()
            
            # Remove loading indicator
            self._loading_label.setParent(None)
            self._loading_label.deleteLater()
            
            # Add content to layout
            self._layout.addWidget(self._content_widget)
            
            self._initialized = True
        except Exception as e:
            print(f"[LazyWidget] Failed to initialize content: {e}")
            import traceback
            traceback.print_exc()
    
    def get_content(self) -> Optional[QtWidgets.QWidget]:
        """Get the content widget (may trigger initialization)."""
        if not self._initialized and self._creator:
            self._initialize_content()
        return self._content_widget


class LazyTabWidget(QtWidgets.QTabWidget):
    """
    Tab widget that creates tab content lazily.
    
    Only creates tab widgets when the user first switches to them.
    """
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        
        self._tab_creators: Dict[int, Callable[[], QtWidgets.QWidget]] = {}
        self._tab_widgets: Dict[int, QtWidgets.QWidget] = {}
        self._initialized_tabs: set[int] = set()
        
        # Connect to tab change signal
        self.currentChanged.connect(self._on_tab_changed)
    
    def add_lazy_tab(
        self,
        creator: Callable[[], QtWidgets.QWidget],
        label: str,
        icon: Optional[QtGui.QIcon] = None
    ) -> int:
        """
        Add a tab with lazy content creation.
        
        Args:
            creator: Function that creates the tab widget
            label: Tab label
            icon: Optional tab icon
        
        Returns:
            Tab index
        """
        # Create placeholder widget
        placeholder = QtWidgets.QWidget(self)
        placeholder_layout = QtWidgets.QVBoxLayout(placeholder)
        loading_label = QtWidgets.QLabel("Loading tab content...", placeholder)
        loading_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        placeholder_layout.addWidget(loading_label)
        
        # Add tab
        if icon:
            index = self.addTab(placeholder, icon, label)
        else:
            index = self.addTab(placeholder, label)
        
        # Store creator
        self._tab_creators[index] = creator
        
        # Initialize first tab immediately
        if index == 0:
            self._initialize_tab(index)
        
        return index
    
    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change - initialize content if needed."""
        if index >= 0 and index not in self._initialized_tabs:
            self._initialize_tab(index)
    
    def _initialize_tab(self, index: int) -> None:
        """Initialize a tab's content."""
        if index in self._initialized_tabs:
            return
        
        creator = self._tab_creators.get(index)
        if not creator:
            return
        
        try:
            # Create the actual widget
            widget = creator()
            
            # Replace placeholder
            old_widget = self.widget(index)
            self.removeTab(index)
            
            # Re-insert at same position
            tab_text = old_widget.windowTitle() if old_widget else "Tab"
            self.insertTab(index, widget, tab_text)
            
            # Clean up old widget
            if old_widget:
                old_widget.deleteLater()
            
            # Mark as initialized
            self._initialized_tabs.add(index)
            self._tab_widgets[index] = widget
            
            print(f"[LazyTabWidget] Initialized tab {index}: {tab_text}")
        except Exception as e:
            print(f"[LazyTabWidget] Failed to initialize tab {index}: {e}")
            import traceback
            traceback.print_exc()
    
    def get_tab_widget(self, index: int) -> Optional[QtWidgets.QWidget]:
        """Get tab widget (may trigger initialization)."""
        if index not in self._initialized_tabs:
            self._initialize_tab(index)
        return self._tab_widgets.get(index)


# Need to import QtCore and QtGui
from PyQt6 import QtCore, QtGui


__all__ = ["LazyWidget", "LazyTabWidget"]
