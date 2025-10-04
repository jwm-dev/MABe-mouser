"""Advanced analysis tab for MABe mouse behavior detection dataset.

This module provides powerful analysis tools for ML engineers working on the
Kaggle MABe competition, including:
- Dataset statistics and distributions
- Behavior annotation analysis
- Trajectory visualization with Vispy
- Feature extraction for model training
- Interactive plots and exports
- Comparative analysis across dataset
"""

from __future__ import annotations

import csv
import math
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets
from vispy import scene
from vispy.scene import visuals

from .constants import UI_ACCENT, UI_BACKGROUND, UI_SURFACE, UI_TEXT_MUTED, UI_TEXT_PRIMARY
from .dataset_stats import DatasetStatsCache
from .models import FramePayload


def _compute_centroid(points: np.ndarray) -> Optional[np.ndarray]:
    """Compute centroid of valid points."""
    valid_points = [p for p in points if not np.isnan(p).any()]
    if not valid_points:
        return None
    return np.mean(valid_points, axis=0)


def _compute_velocity(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Compute Euclidean velocity between two positions."""
    return float(np.linalg.norm(pos2 - pos1))


def _compute_acceleration(vel1: float, vel2: float) -> float:
    """Compute acceleration as change in velocity."""
    return abs(vel2 - vel1)


def _compute_angular_velocity(pos1: np.ndarray, pos2: np.ndarray, pos3: np.ndarray) -> float:
    """Compute angular velocity from three consecutive positions."""
    v1 = pos2 - pos1
    v2 = pos3 - pos2
    
    # Avoid division by zero
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    # Compute angle between vectors
    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return float(angle)


def _compute_distance_matrix(positions: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
    """Compute pairwise distances between all mice."""
    distances = {}
    mouse_ids = sorted(positions.keys())
    
    for i, mouse1 in enumerate(mouse_ids):
        for mouse2 in mouse_ids[i + 1:]:
            dist = np.linalg.norm(positions[mouse1] - positions[mouse2])
            distances[(mouse1, mouse2)] = float(dist)
    
    return distances


class AnalysisPane(QtWidgets.QWidget):
    """Main analysis widget with multiple analysis tools."""
    
    # Signal emitted when analysis is complete
    analysis_complete = QtCore.pyqtSignal()
    
    def __init__(
        self, 
        parent: Optional[QtWidgets.QWidget] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        dataset_stats: Optional[DatasetStatsCache] = None,
        on_main_thread: Optional[Callable] = None
    ) -> None:
        super().__init__(parent)
        self.current_data: Optional[Dict[str, Any]] = None
        self.current_path: Optional[Path] = None
        self._executor = executor
        self._dataset_stats = dataset_stats
        self._on_main_thread = on_main_thread or (lambda fn: fn())
        self._update_future: Optional[Future] = None
        
        # Analysis results cache - keyed by file path and data signature
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
        self._max_cache_entries = 10  # Cache last 10 analyzed files
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Initialize the UI components."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create tab widget for different analysis views
        self.analysis_tabs = QtWidgets.QTabWidget(self)
        self.analysis_tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background-color: {UI_BACKGROUND};
            }}
            QTabBar::tab {{
                background-color: {UI_BACKGROUND};
                color: {UI_TEXT_MUTED};
                padding: 8px 16px;
                border: none;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {UI_TEXT_PRIMARY};
                border-bottom: 2px solid {UI_ACCENT};
            }}
            QTabBar::tab:hover {{
                color: {UI_TEXT_PRIMARY};
            }}
        """)
        
        # Add analysis tabs
        self._create_overview_tab()
        self._create_statistics_tab()
        # Trajectories moved to Graphs tab
        self._create_behaviors_tab()
        self._create_features_tab()
        self._create_feature_discovery_tab()
        
        layout.addWidget(self.analysis_tabs)
    
    def _create_overview_tab(self) -> None:
        """Create dataset overview tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QtWidgets.QLabel("Dataset Overview")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {UI_TEXT_PRIMARY};")
        layout.addWidget(title)
        
        # Scrollable area for stats
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        
        self.overview_content = QtWidgets.QWidget()
        self.overview_layout = QtWidgets.QVBoxLayout(self.overview_content)
        self.overview_layout.setSpacing(10)
        
        scroll.setWidget(self.overview_content)
        layout.addWidget(scroll, 1)
        
        self.analysis_tabs.addTab(widget, "Overview")
    
    def _create_statistics_tab(self) -> None:
        """Create statistical analysis tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Stats table with virtual scrolling
        from .virtual_table import VirtualScrollTable
        self.stats_table = VirtualScrollTable()
        self.stats_table.setStyleSheet(f"""
            QTableView {{
                background-color: {UI_BACKGROUND};
                color: {UI_TEXT_PRIMARY};
                gridline-color: {UI_TEXT_MUTED};
                border: none;
            }}
            QHeaderView::section {{
                background-color: {UI_BACKGROUND};
                color: {UI_TEXT_PRIMARY};
                padding: 8px;
                border: none;
                border-bottom: 1px solid {UI_TEXT_MUTED};
            }}
        """)
        self.stats_table.setAlternatingRowColors(True)
        layout.addWidget(self.stats_table)
        
        self.analysis_tabs.addTab(widget, "Statistics")
    
    # REMOVED: Trajectories tab - moved to graphs.py
    # def _create_trajectories_tab(self) -> None:
    #     """Create trajectory visualization tab using Vispy."""
    #     # This functionality is now in the Graphs tab
    #     pass
    
    def _create_behaviors_tab(self) -> None:
        """Create behavior analysis tab for annotated data."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Scope selector
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Show behaviors from:"))
        
        self.behavior_scope = QtWidgets.QComboBox()
        self.behavior_scope.addItem("Current File Only")
        self.behavior_scope.addItem("All Files in Lab")
        self.behavior_scope.addItem("Entire Dataset")
        self.behavior_scope.currentTextChanged.connect(self._update_behaviors)
        self.behavior_scope.setStyleSheet(f"""
            QComboBox {{
                background-color: {UI_BACKGROUND};
                color: {UI_TEXT_PRIMARY};
                padding: 5px;
                border: 1px solid {UI_TEXT_MUTED};
                border-radius: 4px;
            }}
        """)
        controls.addWidget(self.behavior_scope)
        controls.addStretch()
        
        layout.addLayout(controls)
        
        # Behavior content
        self.behavior_content = QtWidgets.QTextEdit()
        self.behavior_content.setReadOnly(True)
        self.behavior_content.setStyleSheet(f"""
            QTextEdit {{
                background-color: {UI_BACKGROUND};
                color: {UI_TEXT_PRIMARY};
                border: none;
                font-family: 'Courier New', monospace;
            }}
        """)
        layout.addWidget(self.behavior_content, 1)
        
        self.analysis_tabs.addTab(widget, "Behaviors")
    
    def _create_features_tab(self) -> None:
        """Create feature extraction and distribution analysis tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title and description
        title = QtWidgets.QLabel("Feature Engineering for ML Models")
        title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {UI_TEXT_PRIMARY};")
        layout.addWidget(title)
        
        info = QtWidgets.QLabel(
            "Computed features for each mouse:\n"
            "• Velocity (frame-to-frame movement)\n"
            "• Acceleration (velocity changes)\n"
            "• Angular velocity (turning rate)\n"
            "• Inter-mouse distances (social proximity)\n"
            "• Trajectory statistics"
        )
        info.setStyleSheet(f"color: {UI_TEXT_MUTED}; padding: 5px;")
        layout.addWidget(info)
        
        # Features table with virtual scrolling
        from .virtual_table import VirtualScrollTable
        self.features_table = VirtualScrollTable()
        self.features_table.setStyleSheet(f"""
            QTableView {{
                background-color: {UI_BACKGROUND};
                color: {UI_TEXT_PRIMARY};
                gridline-color: {UI_TEXT_MUTED};
                border: none;
            }}
            QHeaderView::section {{
                background-color: {UI_BACKGROUND};
                color: {UI_TEXT_PRIMARY};
                padding: 8px;
                border: none;
                border-bottom: 1px solid {UI_TEXT_MUTED};
            }}
        """)
        self.features_table.setAlternatingRowColors(True)
        layout.addWidget(self.features_table, 1)
        
        # Export button
        export_layout = QtWidgets.QHBoxLayout()
        export_layout.addStretch()
        
        self.export_features_btn = QtWidgets.QPushButton("Export Features to CSV")
        self.export_features_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {UI_ACCENT};
                color: #0d1220;
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #7dd3ff;
            }}
            QPushButton:disabled {{
                background-color: {UI_TEXT_MUTED};
                color: #555;
            }}
        """)
        self.export_features_btn.clicked.connect(self._export_features)
        self.export_features_btn.setEnabled(False)
        export_layout.addWidget(self.export_features_btn)
        
        layout.addLayout(export_layout)
        
        self.analysis_tabs.addTab(widget, "Features")
    
    def _create_feature_discovery_tab(self) -> None:
        """Create advanced feature discovery tab for ML model improvement."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title and description
        title = QtWidgets.QLabel("Feature Discovery for SVM/Classification")
        title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {UI_TEXT_PRIMARY};")
        layout.addWidget(title)
        
        info = QtWidgets.QLabel(
            "Advanced feature engineering to find decision boundaries:\n"
            "• Body orientation and relative angles\n"
            "• Convex hull properties (area, perimeter)\n"
            "• Inter-mouse spatial relationships\n"
            "• Pose geometry (triangle areas, distances)\n"
            "• Feature separability analysis"
        )
        info.setStyleSheet(f"color: {UI_TEXT_MUTED}; padding: 5px;")
        layout.addWidget(info)
        
        # Create tab widget for different discovery tools
        self.discovery_tabs = QtWidgets.QTabWidget()
        self.discovery_tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {UI_TEXT_MUTED};
                background-color: {UI_BACKGROUND};
            }}
            QTabBar::tab {{
                background-color: {UI_BACKGROUND};
                color: {UI_TEXT_MUTED};
                padding: 6px 12px;
                border: 1px solid {UI_TEXT_MUTED};
                border-bottom: none;
            }}
            QTabBar::tab:selected {{
                color: {UI_TEXT_PRIMARY};
                background-color: {UI_SURFACE};
            }}
        """)
        
        # Sub-tab 1: Geometric Features
        geom_widget = QtWidgets.QWidget()
        geom_layout = QtWidgets.QVBoxLayout(geom_widget)
        from .virtual_table import VirtualScrollTable
        self.geom_features_table = VirtualScrollTable()
        self.geom_features_table.setStyleSheet(f"""
            QTableView {{
                background-color: {UI_BACKGROUND};
                color: {UI_TEXT_PRIMARY};
                gridline-color: {UI_TEXT_MUTED};
                border: none;
            }}
            QHeaderView::section {{
                background-color: {UI_BACKGROUND};
                color: {UI_TEXT_PRIMARY};
                padding: 8px;
                border: none;
                border-bottom: 1px solid {UI_TEXT_MUTED};
            }}
        """)
        self.geom_features_table.setAlternatingRowColors(True)
        geom_layout.addWidget(self.geom_features_table)
        self.discovery_tabs.addTab(geom_widget, "Geometric Features")
        
        # Sub-tab 2: Separability Analysis
        sep_widget = QtWidgets.QWidget()
        sep_layout = QtWidgets.QVBoxLayout(sep_widget)
        self.separability_content = QtWidgets.QTextEdit()
        self.separability_content.setReadOnly(True)
        self.separability_content.setStyleSheet(f"""
            QTextEdit {{
                background-color: {UI_BACKGROUND};
                color: {UI_TEXT_PRIMARY};
                border: none;
                font-family: 'Courier New', monospace;
            }}
        """)
        sep_layout.addWidget(self.separability_content)
        self.discovery_tabs.addTab(sep_widget, "Feature Separability")
        
        # 3D Point Cloud moved to Graphs tab
        # # Sub-tab 3: 3D Point Cloud
        # cloud_widget = QtWidgets.QWidget()
        # cloud_layout = QtWidgets.QVBoxLayout(cloud_widget)
        # 
        # cloud_info = QtWidgets.QLabel(
        #     "3D Point Cloud Visualization\n\n"
        #     "View pose keypoints in 3D space (X, Y, Time) to identify spatial-temporal patterns."
        # )
        # cloud_info.setStyleSheet(f"color: {UI_TEXT_MUTED}; padding: 10px;")
        # cloud_layout.addWidget(cloud_info)
        # 
        # # Vispy 3D canvas - use simpler approach with arcball camera
        # self.cloud_canvas = scene.SceneCanvas(
        #     keys='interactive',
        #     bgcolor=UI_BACKGROUND,
        #     parent=cloud_widget
        # )
        # 
        # # Create a view - camera will be set when data is loaded
        # self.cloud_view = self.cloud_canvas.central_widget.add_view()
        # # Use ArcballCamera instead - it's more stable for 3D visualization
        # self.cloud_view.camera = scene.ArcballCamera(fov=60, distance=1000)
        # 
        # cloud_layout.addWidget(self.cloud_canvas.native)
        # 
        # cloud_hint = QtWidgets.QLabel(
        #     "Use mouse to rotate (drag), zoom (scroll), and pan (right-drag)."
        # )
        # cloud_hint.setStyleSheet(f"color: {UI_TEXT_MUTED}; font-size: 10px; padding: 5px;")
        # cloud_layout.addWidget(cloud_hint)
        # 
        # self.discovery_tabs.addTab(cloud_widget, "3D Point Cloud")
        
        layout.addWidget(self.discovery_tabs, 1)
        
        # Export button for discovered features
        export_layout = QtWidgets.QHBoxLayout()
        export_layout.addStretch()
        
        self.export_discovery_btn = QtWidgets.QPushButton("Export Discovered Features")
        self.export_discovery_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {UI_ACCENT};
                color: #0d1220;
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #7dd3ff;
            }}
            QPushButton:disabled {{
                background-color: {UI_TEXT_MUTED};
                color: #555;
            }}
        """)
        self.export_discovery_btn.clicked.connect(self._export_discovered_features)
        self.export_discovery_btn.setEnabled(False)
        export_layout.addWidget(self.export_discovery_btn)
        
        layout.addLayout(export_layout)
        
        self.analysis_tabs.addTab(widget, "Feature Discovery")
    
    def update_data(self, path: Optional[Path], data: Optional[Dict[str, Any]]) -> None:
        """Update analysis with new data."""
        print(f"[AnalysisPane] update_data called with path={path}, data keys={data.keys() if data else None}")
        
        self.current_path = path
        self.current_data = data
        
        if data is None or not data:
            self._clear_all()
            self.analysis_complete.emit()
            return
        
        # Extract payloads - the actual FramePayload objects
        payloads = data.get("payloads", [])
        print(f"[AnalysisPane] Got {len(payloads)} payloads")
        if payloads and len(payloads) > 0:
            print(f"[AnalysisPane] First payload type: {type(payloads[0])}")
        
        # NEW: Use smart cache for analysis results
        from .smart_cache import get_cache_manager
        cache_mgr = get_cache_manager()
        cache_key = cache_mgr.get_cache_key(path=path, params={'n_frames': len(payloads)})
        
        # Check analysis cache
        cached_analysis = cache_mgr.analysis_cache.get(cache_key)
        if cached_analysis is not None:
            print(f"[AnalysisPane] Using cached analysis results for {path}")
            self._cached_results = cached_analysis
            self._update_all_tabs_from_cache()
            self.analysis_complete.emit()
            return
        
        # Cancel any pending update
        if self._update_future is not None:
            self._update_future.cancel()
            self._update_future = None
        
        print(f"[AnalysisPane] Starting update (executor={'present' if self._executor else 'None'})")
        
        # If we have an executor, do heavy computations asynchronously
        if self._executor is not None:
            self._update_async()
        else:
            self._update_all_tabs()
    
    def _prepare_data_for_cache(self, path: Path, data: Dict[str, Any]) -> None:
        """Prepare analysis data in background for instant display later."""
        print(f"[AnalysisPane] Preparing cache data for {path.name}")
        
        # Extract payloads
        payloads = data.get("payloads", [])
        if not payloads:
            print(f"[AnalysisPane] No payloads to cache")
            return
        
        # Run the full analysis computation in background
        # This will cache all results for instant display later
        from .smart_cache import get_cache_manager
        cache_mgr = get_cache_manager()
        cache_key = cache_mgr.get_cache_key(path=path, params={'n_frames': len(payloads)})
        
        # Check if already cached
        if cache_mgr.analysis_cache.get(cache_key) is not None:
            print(f"[AnalysisPane] Already cached for {path.name}")
            return
        
        # Compute analysis results
        results = {
            'computed': True,
            'path': path,
            'n_frames': len(payloads)
        }
        
        # Pre-compute expensive features
        if payloads:
            try:
                # Compute geometric features (expensive)
                results['geometric_features'] = self._precompute_geometric_features(payloads)
                print(f"[AnalysisPane] Computed geometric features")
            except Exception as e:
                print(f"[AnalysisPane] Failed to precompute geometric features: {e}")
                import traceback
                traceback.print_exc()
        
        # Store in cache
        cache_mgr.analysis_cache.put(cache_key, results)
        print(f"[AnalysisPane] Cache prepared for {path.name}")
    
    def _get_cache_key(self, path: Optional[Path], data: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for analysis results."""
        if path is None:
            return "none"
        # Use path + number of frames as simple signature
        payloads = data.get("payloads", []) if data else []
        return f"{path}::{len(payloads)}"
    
    def _cache_results(self, cache_key: str, results: Dict[str, Any]) -> None:
        """Cache analysis results with LRU eviction."""
        self._analysis_cache[cache_key] = results
        
        # Prune cache if too large
        if len(self._analysis_cache) > self._max_cache_entries:
            # Simple FIFO eviction (first added gets removed)
            oldest_key = next(iter(self._analysis_cache))
            del self._analysis_cache[oldest_key]
    
    def _update_async(self) -> None:
        """Update all tabs asynchronously in background thread."""
        # Capture data for background thread
        current_data = self.current_data
        current_path = self.current_path
        cache_key = self._get_cache_key(current_path, current_data)
        
        def worker():
            # Compute heavy analysis in background
            # Extract payloads
            payloads = current_data.get("payloads", []) if current_data else []
            
            results = {
                'computed': True,
                'path': current_path,
                'n_frames': len(payloads)
            }
            
            # Pre-compute expensive features
            if payloads:
                try:
                    # Compute geometric features (expensive)
                    results['geometric_features'] = self._precompute_geometric_features(payloads)
                except Exception as e:
                    print(f"[AnalysisPane] Failed to precompute geometric features: {e}")
            
            return results
        
        # Use priority executor if available
        executor_mgr = getattr(self._executor, '_executor_manager', None)
        if executor_mgr is not None and hasattr(executor_mgr, 'submit_analysis'):
            # Get the actual analysis executor from manager
            analysis_executor = getattr(executor_mgr, 'analysis_executor', None)
            if analysis_executor:
                from .priority_executor import TaskPriority
                self._update_future = analysis_executor.submit(worker, priority=TaskPriority.HIGH)
            else:
                self._update_future = self._executor.submit(worker)
        else:
            self._update_future = self._executor.submit(worker)
        
        def on_complete(future: Future) -> None:
            if future.cancelled():
                return
            try:
                results = future.result()
                
                # NEW: Cache in smart cache system
                from .smart_cache import get_cache_manager
                cache_mgr = get_cache_manager()
                cache_key = cache_mgr.get_cache_key(path=current_path, params={'n_frames': results.get('n_frames', 0)})
                cache_mgr.analysis_cache.put(cache_key, results)
                
                # Store for use during UI update
                self._cached_results = results
                # Apply results on main thread
                self._on_main_thread(lambda: self._update_all_tabs())
            except Exception as exc:
                print(f"[AnalysisPane] Async update failed: {exc}")
                import traceback
                traceback.print_exc()
        
        self._update_future.add_done_callback(on_complete)
    
    def _precompute_geometric_features(self, payloads: List[FramePayload]) -> Dict[str, Any]:
        """Precompute expensive geometric features in background thread."""
        geom_features = {}
        
        # Get arena dimensions if available
        arena_center = self.current_data.get('display_center', (0, 0)) if self.current_data else (0, 0)
        
        prev_positions = {}
        prev_velocities = {}
        
        for frame_idx, payload in enumerate(payloads):
            if not isinstance(payload, FramePayload):
                continue
            
            # Track all mouse centroids for this frame
            frame_centroids = {}
            
            for mouse_id, group in payload.mouse_groups.items():
                if mouse_id not in geom_features:
                    geom_features[mouse_id] = {
                        'convex_hull_areas': [],
                        'body_orientations': [],
                        'pose_spread': [],
                        'ellipse_aspect_ratios': [],
                        'body_compactness': [],
                        'velocity_magnitudes': [],
                        'acceleration_magnitudes': [],
                        'jerk_magnitudes': [],
                        'distance_to_arena_center': [],
                    }
                
                valid_points = [p for p in group.points if not np.isnan(p).any()]
                if len(valid_points) < 3:
                    continue
                
                points_array = np.array(valid_points)
                centroid = np.mean(points_array, axis=0)
                frame_centroids[mouse_id] = centroid
                
                # Convex hull area
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(points_array)
                    geom_features[mouse_id]['convex_hull_areas'].append(hull.volume)
                except:
                    pass
                
                # Body orientation
                if len(points_array) > 1:
                    centered = points_array - centroid
                    cov = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eig(cov)
                    
                    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
                    orientation = np.arctan2(principal_axis[1], principal_axis[0])
                    geom_features[mouse_id]['body_orientations'].append(orientation)
                    
                    eig_sorted = np.sort(eigenvalues)[::-1]
                    if eig_sorted[1] > 1e-10:
                        aspect_ratio = eig_sorted[0] / eig_sorted[1]
                        geom_features[mouse_id]['ellipse_aspect_ratios'].append(aspect_ratio)
                
                # Pose spread
                distances = np.linalg.norm(points_array - centroid, axis=1)
                geom_features[mouse_id]['pose_spread'].append(np.std(distances))
                
                # Body compactness
                if len(distances) > 0:
                    mean_radius = np.mean(distances)
                    if mean_radius > 1e-10:
                        compactness = 1.0 / mean_radius
                        geom_features[mouse_id]['body_compactness'].append(compactness)
                
                # Distance to arena center
                if arena_center:
                    dist_to_center = np.linalg.norm(centroid - np.array(arena_center))
                    geom_features[mouse_id]['distance_to_arena_center'].append(dist_to_center)
                
                # Velocity, acceleration, jerk
                if mouse_id in prev_positions:
                    velocity = centroid - prev_positions[mouse_id]
                    vel_mag = np.linalg.norm(velocity)
                    geom_features[mouse_id]['velocity_magnitudes'].append(vel_mag)
                    
                    if mouse_id in prev_velocities:
                        acceleration = velocity - prev_velocities[mouse_id]
                        acc_mag = np.linalg.norm(acceleration)
                        geom_features[mouse_id]['acceleration_magnitudes'].append(acc_mag)
                    
                    prev_velocities[mouse_id] = velocity
                
                prev_positions[mouse_id] = centroid
        
        return geom_features
    
    def _update_all_tabs_from_cache(self) -> None:
        """Update all tabs using cached results."""
        # Just update UI with cached data
        self._update_all_tabs()
    
    def _update_all_tabs(self) -> None:
        """Update all tabs (must be called on main thread for UI updates)."""
        print(f"[AnalysisPane] _update_all_tabs starting")
        try:
            self._update_overview()
            print(f"[AnalysisPane] Overview updated")
            self._update_statistics()
            print(f"[AnalysisPane] Statistics updated")
            # Trajectories moved to Graphs tab
            # self._update_trajectories()
            # print(f"[AnalysisPane] Trajectories updated")
            self._update_behaviors()
            print(f"[AnalysisPane] Behaviors updated")
            self._update_features()
            print(f"[AnalysisPane] Features updated")
            self._update_feature_discovery()
            print(f"[AnalysisPane] Feature discovery updated")
        finally:
            # Emit signal when done
            print(f"[AnalysisPane] All tabs updated, emitting analysis_complete signal")
            self.analysis_complete.emit()
    
    def _clear_all(self) -> None:
        """Clear all analysis displays."""
        # Clear overview
        while self.overview_layout.count():
            child = self.overview_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Clear other tabs
        self.stats_table.set_data([], [])
        self.behavior_content.clear()
        self.features_table.set_data([], [])
        self.geom_features_table.set_data([], [])
        # Trajectory view removed - now in graphs.py
    
    def _update_overview(self) -> None:
        """Update the overview tab with dataset summary."""
        # Clear existing content
        while self.overview_layout.count():
            child = self.overview_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if not self.current_data:
            return
        
        # Extract basic info
        payloads = self.current_data.get("payloads", [])
        metadata = self.current_data.get("metadata", {})
        
        # File info
        file_label = QtWidgets.QLabel(f"<b>File:</b> {self.current_path.name if self.current_path else 'Unknown'}")
        file_label.setStyleSheet(f"color: {UI_TEXT_PRIMARY};")
        self.overview_layout.addWidget(file_label)
        
        # Frame count
        frame_count = len(payloads)
        frame_label = QtWidgets.QLabel(f"<b>Total Frames:</b> {frame_count:,}")
        frame_label.setStyleSheet(f"color: {UI_TEXT_PRIMARY};")
        self.overview_layout.addWidget(frame_label)
        
        # Collect mouse IDs
        all_mice = set()
        for payload in payloads:
            if isinstance(payload, FramePayload):
                all_mice.update(payload.mouse_groups.keys())
        
        mice_label = QtWidgets.QLabel(f"<b>Mice Detected:</b> {len(all_mice)} ({', '.join(sorted(all_mice))})")
        mice_label.setStyleSheet(f"color: {UI_TEXT_PRIMARY};")
        self.overview_layout.addWidget(mice_label)
        
        # Video dimensions - check multiple possible sources
        video_width = "Unknown"
        video_height = "Unknown"
        
        if metadata:
            video_width = metadata.get("video_width", metadata.get("width", "Unknown"))
            video_height = metadata.get("video_height", metadata.get("height", "Unknown"))
        
        # Also check video_size_px in main data dict
        if video_width == "Unknown" and "video_size_px" in self.current_data:
            video_size = self.current_data["video_size_px"]
            if isinstance(video_size, (list, tuple)) and len(video_size) >= 2:
                video_width = video_size[0]
                video_height = video_size[1]
        
        dims_label = QtWidgets.QLabel(f"<b>Video Dimensions:</b> {video_width} × {video_height} px")
        dims_label.setStyleSheet(f"color: {UI_TEXT_PRIMARY};")
        self.overview_layout.addWidget(dims_label)
        
        # Dataset context information
        if self._dataset_stats and self._dataset_stats._loaded and self.current_path:
            self.overview_layout.addSpacing(20)
            context_title = QtWidgets.QLabel("<b>Dataset Context</b>")
            context_title.setStyleSheet(f"color: {UI_TEXT_PRIMARY}; font-size: 14px;")
            self.overview_layout.addWidget(context_title)
            
            # Lab information
            lab_name = self._dataset_stats.get_lab_for_file(self.current_path)
            if lab_name:
                lab_files = self._dataset_stats.get_files_in_lab(lab_name)
                lab_label = QtWidgets.QLabel(
                    f"<b>Lab:</b> {lab_name} "
                    f"({len(lab_files)} files total in this lab)"
                )
                lab_label.setStyleSheet(f"color: {UI_TEXT_PRIMARY};")
                self.overview_layout.addWidget(lab_label)
            
            # Dataset overview
            lab_stats = self._dataset_stats.get_lab_stats()
            total_files = sum(lab_stats.values())
            total_labs = len(lab_stats)
            
            dataset_label = QtWidgets.QLabel(
                f"<b>Full Dataset:</b> {total_files} files across {total_labs} labs"
            )
            dataset_label.setStyleSheet(f"color: {UI_TEXT_MUTED};")
            self.overview_layout.addWidget(dataset_label)
        
        # Check for behaviors
        has_behaviors = self.current_data.get("has_behaviors", False)
        behavior_label = QtWidgets.QLabel(
            f"<b>Behavior Annotations:</b> {'✓ Present' if has_behaviors else '✗ Not Available'}"
        )
        behavior_label.setStyleSheet(f"color: {'#4ade80' if has_behaviors else UI_TEXT_MUTED};")
        self.overview_layout.addWidget(behavior_label)
        
        # Data quality metrics
        self.overview_layout.addSpacing(20)
        quality_title = QtWidgets.QLabel("<b>Data Quality Metrics</b>")
        quality_title.setStyleSheet(f"color: {UI_TEXT_PRIMARY}; font-size: 14px;")
        self.overview_layout.addWidget(quality_title)
        
        # Calculate missing data percentage
        total_points = 0
        missing_points = 0
        for payload in payloads:
            if isinstance(payload, FramePayload):
                for mouse_id, group in payload.mouse_groups.items():
                    total_points += len(group.points)
                    for point in group.points:
                        if np.isnan(point).any():
                            missing_points += 1
        
        if total_points > 0:
            missing_pct = (missing_points / total_points) * 100
            quality_label = QtWidgets.QLabel(
                f"Missing/Invalid Points: {missing_points:,} / {total_points:,} ({missing_pct:.2f}%)"
            )
            quality_label.setStyleSheet(f"color: {UI_TEXT_PRIMARY};")
            self.overview_layout.addWidget(quality_label)
        
        self.overview_layout.addStretch()
    
    def _update_statistics(self) -> None:
        """Update statistics tab with detailed metrics."""
        if not self.current_data:
            self.stats_table.clear()
            return
        
        payloads = self.current_data.get("payloads", [])
        
        # Collect all coordinates
        coords_by_mouse: Dict[str, List[Tuple[float, float]]] = {}
        
        for payload in payloads:
            if isinstance(payload, FramePayload):
                for mouse_id, group in payload.mouse_groups.items():
                    if mouse_id not in coords_by_mouse:
                        coords_by_mouse[mouse_id] = []
                    for point in group.points:
                        if not np.isnan(point).any():
                            coords_by_mouse[mouse_id].append((float(point[0]), float(point[1])))
        
        # Build statistics data for virtual table
        headers = ["Mouse ID", "Points", "X Mean", "X Std", "Y Mean", "Y Std", "X Range", "Y Range"]
        table_data = []
        
        for mouse_id, coords in sorted(coords_by_mouse.items()):
            if not coords:
                continue
            
            x_vals = np.array([c[0] for c in coords])
            y_vals = np.array([c[1] for c in coords])
            
            row = [
                str(mouse_id),
                f"{len(coords):,}",
                f"{np.mean(x_vals):.2f}",
                f"{np.std(x_vals):.2f}",
                f"{np.mean(y_vals):.2f}",
                f"{np.std(y_vals):.2f}",
                f"{np.min(x_vals):.1f} - {np.max(x_vals):.1f}",
                f"{np.min(y_vals):.1f} - {np.max(y_vals):.1f}"
            ]
            table_data.append(row)
        
        # Set data using virtual table's efficient method
        self.stats_table.set_data(table_data, headers)
    
    # REMOVED: Trajectory update - moved to graphs.py
    # def _update_trajectories(self) -> None:
    #     """Update trajectory visualization using Vispy."""
    #     # This functionality is now in the Graphs tab
    #     pass
    
    def _update_behaviors(self) -> None:
        """Update behavior analysis for annotated data with scope support."""
        if not self.current_data:
            self.behavior_content.clear()
            return
        
        scope = self.behavior_scope.currentText()
        
        # Determine if we need to load additional files
        if scope == "Current File Only":
            self._update_behaviors_current_file()
        elif scope == "All Files in Lab":
            self._update_behaviors_lab_scope()
        elif scope == "Entire Dataset":
            self._update_behaviors_dataset_scope()
    
    def _update_behaviors_current_file(self) -> None:
        """Update behaviors for current file only."""
        has_behaviors = self.current_data.get("has_behaviors", False)
        
        if not has_behaviors:
            self.behavior_content.setHtml(
                f"<p style='color: {UI_TEXT_MUTED};'>No behavior annotations found in this file.</p>"
                f"<p style='color: {UI_TEXT_MUTED};'>Behavior annotations are available in the "
                f"train_annotation/ directory.</p>"
            )
            return
        
        # Collect behavior statistics
        payloads = self.current_data.get("payloads", [])
        behavior_counts: Dict[str, int] = {}
        total_frames_with_behaviors = 0
        
        for payload in payloads:
            if isinstance(payload, FramePayload) and payload.behaviors:
                total_frames_with_behaviors += 1
                for mouse_id, behavior in payload.behaviors.items():
                    key = f"{mouse_id}: {behavior}"
                    behavior_counts[key] = behavior_counts.get(key, 0) + 1
        
        # Format output
        output = f"<h3 style='color: {UI_TEXT_PRIMARY};'>Behavior Annotation Summary (Current File)</h3>"
        output += f"<p style='color: {UI_TEXT_PRIMARY};'>Frames with behaviors: {total_frames_with_behaviors:,}</p>"
        output += "<hr>"
        output += f"<h4 style='color: {UI_TEXT_PRIMARY};'>Behavior Distribution:</h4>"
        output += "<table style='width: 100%; border-collapse: collapse;'>"
        output += f"<tr><th style='text-align: left; color: {UI_TEXT_PRIMARY}; padding: 8px;'>Mouse & Behavior</th>"
        output += f"<th style='text-align: right; color: {UI_TEXT_PRIMARY}; padding: 8px;'>Frames</th></tr>"
        
        for behavior, count in sorted(behavior_counts.items(), key=lambda x: x[1], reverse=True):
            output += f"<tr><td style='color: {UI_TEXT_PRIMARY}; padding: 8px;'>{behavior}</td>"
            output += f"<td style='color: {UI_TEXT_PRIMARY}; text-align: right; padding: 8px;'>{count:,}</td></tr>"
        
        output += "</table>"
        
        self.behavior_content.setHtml(output)
    
    def _update_behaviors_lab_scope(self) -> None:
        """Update behaviors for all files in the current lab."""
        if not hasattr(self, 'dataset_stats') or not self.dataset_stats:
            self.behavior_content.setHtml(
                f"<p style='color: {UI_TEXT_MUTED};'>Dataset statistics not available.</p>"
            )
            return
        
        # Get current file to determine lab
        current_file = self.current_data.get("file_path")
        if not current_file:
            self.behavior_content.setHtml(
                f"<p style='color: {UI_TEXT_MUTED};'>No file currently loaded.</p>"
            )
            return
        
        # Find lab for current file
        lab_name = self.dataset_stats.get_lab_for_file(current_file)
        if not lab_name:
            self.behavior_content.setHtml(
                f"<p style='color: {UI_TEXT_MUTED};'>Could not determine lab for current file.</p>"
            )
            return
        
        # Schedule async loading
        self._load_lab_behaviors_async(lab_name)
    
    def _update_behaviors_dataset_scope(self) -> None:
        """Update behaviors for entire dataset."""
        if not hasattr(self, 'dataset_stats') or not self.dataset_stats:
            self.behavior_content.setHtml(
                f"<p style='color: {UI_TEXT_MUTED};'>Dataset statistics not available.</p>"
            )
            return
        
        # Schedule async loading
        self._load_dataset_behaviors_async()
    
    def _load_lab_behaviors_async(self, lab_name: str) -> None:
        """Load behaviors from all files in a lab asynchronously."""
        def load_task():
            try:
                # Get all files in the lab
                lab_files = self.dataset_stats.get_files_in_lab(lab_name)
                
                # Find annotation directory
                base_dir = Path(self.current_data.get("file_path", "")).parents[2]
                annotation_dir = base_dir / "train_annotation" / lab_name
                
                if not annotation_dir.exists():
                    return {"error": f"Annotation directory not found: {annotation_dir}"}
                
                behavior_counts: Dict[str, int] = {}
                total_frames_with_behaviors = 0
                files_processed = 0
                
                # Load each annotation file
                annotation_files = list(annotation_dir.glob("*.parquet"))
                for ann_file in annotation_files:
                    try:
                        import pandas as pd
                        df = pd.read_parquet(ann_file)
                        
                        # Extract behaviors from annotation file
                        action_col = next((c for c in ["behavior", "action", "label"] if c in df.columns), None)
                        agent_col = next((c for c in ["agent_id", "subject_id", "animal_id"] if c in df.columns), None)
                        
                        if action_col and agent_col:
                            for _, row in df.iterrows():
                                behavior = row[action_col]
                                mouse_id = row[agent_col]
                                if pd.notna(behavior) and pd.notna(mouse_id):
                                    key = f"{mouse_id}: {behavior}"
                                    behavior_counts[key] = behavior_counts.get(key, 0) + 1
                                    total_frames_with_behaviors += 1
                        
                        files_processed += 1
                    except Exception as exc:
                        print(f"[AnalysisPane] Failed to load {ann_file}: {exc}")
                        continue
                
                return {
                    "behavior_counts": behavior_counts,
                    "total_frames": total_frames_with_behaviors,
                    "files_processed": files_processed,
                    "lab_name": lab_name
                }
            except Exception as exc:
                import traceback
                traceback.print_exc()
                return {"error": str(exc)}
        
        def on_complete(result):
            if "error" in result:
                self.behavior_content.setHtml(
                    f"<p style='color: {UI_TEXT_MUTED};'>Error loading lab behaviors: {result['error']}</p>"
                )
                return
            
            # Format output
            output = f"<h3 style='color: {UI_TEXT_PRIMARY};'>Behavior Annotation Summary (Lab: {result['lab_name']})</h3>"
            output += f"<p style='color: {UI_TEXT_PRIMARY};'>Files processed: {result['files_processed']}</p>"
            output += f"<p style='color: {UI_TEXT_PRIMARY};'>Total behavior annotations: {result['total_frames']:,}</p>"
            output += "<hr>"
            output += f"<h4 style='color: {UI_TEXT_PRIMARY};'>Behavior Distribution:</h4>"
            output += "<table style='width: 100%; border-collapse: collapse;'>"
            output += f"<tr><th style='text-align: left; color: {UI_TEXT_PRIMARY}; padding: 8px;'>Mouse & Behavior</th>"
            output += f"<th style='text-align: right; color: {UI_TEXT_PRIMARY}; padding: 8px;'>Annotations</th></tr>"
            
            for behavior, count in sorted(result['behavior_counts'].items(), key=lambda x: x[1], reverse=True):
                output += f"<tr><td style='color: {UI_TEXT_PRIMARY}; padding: 8px;'>{behavior}</td>"
                output += f"<td style='color: {UI_TEXT_PRIMARY}; text-align: right; padding: 8px;'>{count:,}</td></tr>"
            
            output += "</table>"
            
            self.behavior_content.setHtml(output)
        
        # Show loading message
        self.behavior_content.setHtml(
            f"<p style='color: {UI_TEXT_MUTED};'>Loading behaviors for lab: {lab_name}...</p>"
        )
        
        # Submit task
        if self.executor:
            future = self.executor.submit(load_task)
            future.add_done_callback(lambda f: self.on_main_thread(on_complete, f.result()))
    
    def _load_dataset_behaviors_async(self) -> None:
        """Load behaviors from entire dataset asynchronously."""
        def load_task():
            try:
                # Find base annotation directory
                base_dir = Path(self.current_data.get("file_path", "")).parents[2]
                annotation_dir = base_dir / "train_annotation"
                
                if not annotation_dir.exists():
                    return {"error": f"Annotation directory not found: {annotation_dir}"}
                
                behavior_counts: Dict[str, int] = {}
                total_frames_with_behaviors = 0
                files_processed = 0
                labs_processed = set()
                
                # Iterate through all lab directories
                for lab_dir in annotation_dir.iterdir():
                    if not lab_dir.is_dir():
                        continue
                    
                    labs_processed.add(lab_dir.name)
                    
                    # Load each annotation file in the lab
                    for ann_file in lab_dir.glob("*.parquet"):
                        try:
                            import pandas as pd
                            df = pd.read_parquet(ann_file)
                            
                            # Extract behaviors from annotation file
                            action_col = next((c for c in ["behavior", "action", "label"] if c in df.columns), None)
                            agent_col = next((c for c in ["agent_id", "subject_id", "animal_id"] if c in df.columns), None)
                            
                            if action_col and agent_col:
                                for _, row in df.iterrows():
                                    behavior = row[action_col]
                                    mouse_id = row[agent_col]
                                    if pd.notna(behavior) and pd.notna(mouse_id):
                                        key = f"{mouse_id}: {behavior}"
                                        behavior_counts[key] = behavior_counts.get(key, 0) + 1
                                        total_frames_with_behaviors += 1
                            
                            files_processed += 1
                        except Exception as exc:
                            print(f"[AnalysisPane] Failed to load {ann_file}: {exc}")
                            continue
                
                return {
                    "behavior_counts": behavior_counts,
                    "total_frames": total_frames_with_behaviors,
                    "files_processed": files_processed,
                    "labs_processed": len(labs_processed)
                }
            except Exception as exc:
                import traceback
                traceback.print_exc()
                return {"error": str(exc)}
        
        def on_complete(result):
            if "error" in result:
                self.behavior_content.setHtml(
                    f"<p style='color: {UI_TEXT_MUTED};'>Error loading dataset behaviors: {result['error']}</p>"
                )
                return
            
            # Format output
            output = f"<h3 style='color: {UI_TEXT_PRIMARY};'>Behavior Annotation Summary (Entire Dataset)</h3>"
            output += f"<p style='color: {UI_TEXT_PRIMARY};'>Labs processed: {result['labs_processed']}</p>"
            output += f"<p style='color: {UI_TEXT_PRIMARY};'>Files processed: {result['files_processed']}</p>"
            output += f"<p style='color: {UI_TEXT_PRIMARY};'>Total behavior annotations: {result['total_frames']:,}</p>"
            output += "<hr>"
            output += f"<h4 style='color: {UI_TEXT_PRIMARY};'>Behavior Distribution:</h4>"
            output += "<table style='width: 100%; border-collapse: collapse;'>"
            output += f"<tr><th style='text-align: left; color: {UI_TEXT_PRIMARY}; padding: 8px;'>Mouse & Behavior</th>"
            output += f"<th style='text-align: right; color: {UI_TEXT_PRIMARY}; padding: 8px;'>Annotations</th></tr>"
            
            for behavior, count in sorted(result['behavior_counts'].items(), key=lambda x: x[1], reverse=True):
                output += f"<tr><td style='color: {UI_TEXT_PRIMARY}; padding: 8px;'>{behavior}</td>"
                output += f"<td style='color: {UI_TEXT_PRIMARY}; text-align: right; padding: 8px;'>{count:,}</td></tr>"
            
            output += "</table>"
            
            self.behavior_content.setHtml(output)
        
        # Show loading message
        self.behavior_content.setHtml(
            f"<p style='color: {UI_TEXT_MUTED};'>Loading behaviors from entire dataset...</p>"
        )
        
        # Submit task
        if self.executor:
            future = self.executor.submit(load_task)
            future.add_done_callback(lambda f: self.on_main_thread(on_complete, f.result()))
    
    def _update_features(self) -> None:
        """Update feature engineering analysis with comprehensive metrics."""
        if not self.current_data:
            self.features_table.clear()
            self.export_features_btn.setEnabled(False)
            return
        
        payloads = self.current_data.get("payloads", [])
        
        # Data structures for feature computation
        velocities_by_mouse: Dict[str, List[float]] = {}
        accelerations_by_mouse: Dict[str, List[float]] = {}
        angular_velocities_by_mouse: Dict[str, List[float]] = {}
        inter_mouse_distances: List[float] = []
        
        prev_positions: Dict[str, np.ndarray] = {}
        prev_velocities: Dict[str, float] = {}
        prev_prev_positions: Dict[str, np.ndarray] = {}
        
        # Compute features frame by frame
        for payload in payloads:
            if not isinstance(payload, FramePayload):
                continue
            
            current_positions: Dict[str, np.ndarray] = {}
            
            # First pass: compute centroids and velocities
            for mouse_id, group in payload.mouse_groups.items():
                centroid = _compute_centroid(group.points)
                if centroid is None:
                    continue
                
                current_positions[mouse_id] = centroid
                
                # Velocity
                if mouse_id in prev_positions:
                    velocity = _compute_velocity(prev_positions[mouse_id], centroid)
                    
                    if mouse_id not in velocities_by_mouse:
                        velocities_by_mouse[mouse_id] = []
                    velocities_by_mouse[mouse_id].append(velocity)
                    
                    # Acceleration
                    if mouse_id in prev_velocities:
                        acceleration = _compute_acceleration(prev_velocities[mouse_id], velocity)
                        if mouse_id not in accelerations_by_mouse:
                            accelerations_by_mouse[mouse_id] = []
                        accelerations_by_mouse[mouse_id].append(acceleration)
                    
                    prev_velocities[mouse_id] = velocity
                    
                    # Angular velocity
                    if mouse_id in prev_prev_positions:
                        ang_vel = _compute_angular_velocity(
                            prev_prev_positions[mouse_id],
                            prev_positions[mouse_id],
                            centroid
                        )
                        if mouse_id not in angular_velocities_by_mouse:
                            angular_velocities_by_mouse[mouse_id] = []
                        angular_velocities_by_mouse[mouse_id].append(ang_vel)
                
                # Update position history
                if mouse_id in prev_positions:
                    prev_prev_positions[mouse_id] = prev_positions[mouse_id]
                prev_positions[mouse_id] = centroid
            
            # Second pass: inter-mouse distances
            if len(current_positions) > 1:
                distances = _compute_distance_matrix(current_positions)
                inter_mouse_distances.extend(distances.values())
        
        # Build comprehensive features data for virtual table
        headers = [
            "Mouse ID",
            "Velocity Mean",
            "Velocity Std",
            "Velocity Max",
            "Accel Mean",
            "Accel Std",
            "Angular Vel Mean",
            "Angular Vel Std"
        ]
        
        all_mice = sorted(set(list(velocities_by_mouse.keys()) + 
                             list(accelerations_by_mouse.keys()) + 
                             list(angular_velocities_by_mouse.keys())))
        
        # Store computed features for export
        self._computed_features = {
            'velocities': velocities_by_mouse,
            'accelerations': accelerations_by_mouse,
            'angular_velocities': angular_velocities_by_mouse,
            'inter_mouse_distances': inter_mouse_distances
        }
        
        table_data = []
        
        for mouse_id in all_mice:
            row = [str(mouse_id)]
            
            # Velocity stats
            if mouse_id in velocities_by_mouse and velocities_by_mouse[mouse_id]:
                vel = np.array(velocities_by_mouse[mouse_id])
                row.extend([f"{np.mean(vel):.3f}", f"{np.std(vel):.3f}", f"{np.max(vel):.3f}"])
            else:
                row.extend(["N/A", "N/A", "N/A"])
            
            # Acceleration stats
            if mouse_id in accelerations_by_mouse and accelerations_by_mouse[mouse_id]:
                acc = np.array(accelerations_by_mouse[mouse_id])
                row.extend([f"{np.mean(acc):.3f}", f"{np.std(acc):.3f}"])
            else:
                row.extend(["N/A", "N/A"])
            
            # Angular velocity stats
            if mouse_id in angular_velocities_by_mouse and angular_velocities_by_mouse[mouse_id]:
                ang = np.array(angular_velocities_by_mouse[mouse_id])
                row.extend([f"{np.mean(ang):.3f}", f"{np.std(ang):.3f}"])
            else:
                row.extend(["N/A", "N/A"])
            
            table_data.append(row)
        
        # Add inter-mouse distance row
        if inter_mouse_distances:
            dist_array = np.array(inter_mouse_distances)
            dist_row = ["Inter-Mouse Dist", f"{np.mean(dist_array):.3f}", f"{np.std(dist_array):.3f}", 
                       f"{np.max(dist_array):.3f}", "", "", "", ""]
            table_data.append(dist_row)
        
        # Set data using virtual table
        self.features_table.set_data(table_data, headers)
        self.export_features_btn.setEnabled(True)
    
    def _export_features(self) -> None:
        """Export computed features to CSV file."""
        if not hasattr(self, '_computed_features') or not self.current_path:
            QtWidgets.QMessageBox.warning(
                self,
                "No Features",
                "No features available to export. Please load data first."
            )
            return
        
        # Ask for save location
        default_name = self.current_path.stem + "_features.csv"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Features to CSV",
            str(self.current_path.parent / default_name),
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            features = self._computed_features
            
            # Prepare DataFrame with all features
            data_rows = []
            
            # Per-mouse features
            all_mice = sorted(set(
                list(features['velocities'].keys()) +
                list(features['accelerations'].keys()) +
                list(features['angular_velocities'].keys())
            ))
            
            for mouse_id in all_mice:
                velocities = features['velocities'].get(mouse_id, [])
                accelerations = features['accelerations'].get(mouse_id, [])
                angular_vels = features['angular_velocities'].get(mouse_id, [])
                
                row = {
                    'mouse_id': mouse_id,
                    'velocity_mean': np.mean(velocities) if velocities else np.nan,
                    'velocity_std': np.std(velocities) if velocities else np.nan,
                    'velocity_max': np.max(velocities) if velocities else np.nan,
                    'velocity_min': np.min(velocities) if velocities else np.nan,
                    'acceleration_mean': np.mean(accelerations) if accelerations else np.nan,
                    'acceleration_std': np.std(accelerations) if accelerations else np.nan,
                    'acceleration_max': np.max(accelerations) if accelerations else np.nan,
                    'angular_velocity_mean': np.mean(angular_vels) if angular_vels else np.nan,
                    'angular_velocity_std': np.std(angular_vels) if angular_vels else np.nan,
                    'angular_velocity_max': np.max(angular_vels) if angular_vels else np.nan,
                }
                data_rows.append(row)
            
            # Add inter-mouse distance summary row
            if features['inter_mouse_distances']:
                distances = np.array(features['inter_mouse_distances'])
                data_rows.append({
                    'mouse_id': 'INTER_MOUSE_DISTANCE',
                    'velocity_mean': np.mean(distances),
                    'velocity_std': np.std(distances),
                    'velocity_max': np.max(distances),
                    'velocity_min': np.min(distances),
                    'acceleration_mean': np.nan,
                    'acceleration_std': np.nan,
                    'acceleration_max': np.nan,
                    'angular_velocity_mean': np.nan,
                    'angular_velocity_std': np.nan,
                    'angular_velocity_max': np.nan,
                })
            
            df = pd.DataFrame(data_rows)
            df.to_csv(file_path, index=False)
            
            QtWidgets.QMessageBox.information(
                self,
                "Export Successful",
                f"Features exported successfully to:\n{file_path}"
            )
            
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export features:\n{str(exc)}"
            )
    
    def _update_feature_discovery(self) -> None:
        """Update feature discovery tab with advanced ML features."""
        if not self.current_data:
            self.geom_features_table.set_data([], [])
            self.separability_content.clear()
            self.export_discovery_btn.setEnabled(False)
            return
        
        payloads = self.current_data.get("payloads", [])
        if not payloads:
            return
        
        # Compute geometric features
        self._compute_geometric_features(payloads)
        
        # Compute separability analysis
        self._compute_separability_analysis(payloads)
        
        # 3D point cloud moved to Graphs tab
        # self._update_3d_point_cloud(payloads)
        
        self.export_discovery_btn.setEnabled(True)
    
    def _compute_geometric_features(self, payloads: List[FramePayload]) -> None:
        """Compute advanced geometric and social features for MABe competition."""
        geom_features = {}
        social_features = []
        
        # Get arena dimensions if available
        arena_center = self.current_data.get('display_center', (0, 0))
        
        prev_positions = {}
        prev_velocities = {}
        
        for frame_idx, payload in enumerate(payloads):
            if not isinstance(payload, FramePayload):
                continue
            
            # Track all mouse centroids for this frame
            frame_centroids = {}
            
            for mouse_id, group in payload.mouse_groups.items():
                if mouse_id not in geom_features:
                    geom_features[mouse_id] = {
                        'convex_hull_areas': [],
                        'body_orientations': [],
                        'pose_spread': [],
                        'ellipse_aspect_ratios': [],
                        'body_compactness': [],
                        'velocity_magnitudes': [],
                        'acceleration_magnitudes': [],
                        'jerk_magnitudes': [],
                        'distance_to_arena_center': [],
                        'path_tortuosity': [],
                    }
                
                valid_points = [p for p in group.points if not np.isnan(p).any()]
                if len(valid_points) < 3:
                    continue
                
                points_array = np.array(valid_points)
                centroid = np.mean(points_array, axis=0)
                frame_centroids[mouse_id] = centroid
                
                # 1. Convex hull area (body size proxy)
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(points_array)
                    geom_features[mouse_id]['convex_hull_areas'].append(hull.volume)
                except:
                    pass
                
                # 2. Body orientation (PCA-based)
                if len(points_array) > 1:
                    centered = points_array - centroid
                    cov = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eig(cov)
                    
                    # Primary orientation
                    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
                    orientation = np.arctan2(principal_axis[1], principal_axis[0])
                    geom_features[mouse_id]['body_orientations'].append(orientation)
                    
                    # Ellipse aspect ratio (elongation)
                    eig_sorted = np.sort(eigenvalues)[::-1]
                    if eig_sorted[1] > 1e-10:
                        aspect_ratio = eig_sorted[0] / eig_sorted[1]
                        geom_features[mouse_id]['ellipse_aspect_ratios'].append(aspect_ratio)
                
                # 3. Pose spread (posture variability)
                distances = np.linalg.norm(points_array - centroid, axis=1)
                geom_features[mouse_id]['pose_spread'].append(np.std(distances))
                
                # 4. Body compactness (inverse of spread normalized by area)
                if len(distances) > 0:
                    mean_radius = np.mean(distances)
                    if mean_radius > 1e-10:
                        compactness = 1.0 / mean_radius
                        geom_features[mouse_id]['body_compactness'].append(compactness)
                
                # 5. Distance to arena center
                if arena_center:
                    dist_to_center = np.linalg.norm(centroid - np.array(arena_center))
                    geom_features[mouse_id]['distance_to_arena_center'].append(dist_to_center)
                
                # 6-8. Velocity, Acceleration, Jerk (temporal derivatives)
                if mouse_id in prev_positions:
                    velocity = centroid - prev_positions[mouse_id]
                    velocity_mag = np.linalg.norm(velocity)
                    geom_features[mouse_id]['velocity_magnitudes'].append(velocity_mag)
                    
                    if mouse_id in prev_velocities:
                        acceleration = velocity - prev_velocities[mouse_id]
                        accel_mag = np.linalg.norm(acceleration)
                        geom_features[mouse_id]['acceleration_magnitudes'].append(accel_mag)
                        
                        # Jerk (rate of change of acceleration)
                        if len(geom_features[mouse_id]['acceleration_magnitudes']) > 1:
                            prev_accel_mag = geom_features[mouse_id]['acceleration_magnitudes'][-2]
                            jerk = abs(accel_mag - prev_accel_mag)
                            geom_features[mouse_id]['jerk_magnitudes'].append(jerk)
                    
                    prev_velocities[mouse_id] = velocity
                
                prev_positions[mouse_id] = centroid
            
            # Social/Inter-mouse features
            if len(frame_centroids) >= 2:
                mice_list = list(frame_centroids.keys())
                for i in range(len(mice_list)):
                    for j in range(i + 1, len(mice_list)):
                        mouse_i, mouse_j = mice_list[i], mice_list[j]
                        cent_i, cent_j = frame_centroids[mouse_i], frame_centroids[mouse_j]
                        
                        # Inter-mouse distance
                        distance = np.linalg.norm(cent_i - cent_j)
                        social_features.append({
                            'frame': frame_idx,
                            'mouse_pair': f"{mouse_i}-{mouse_j}",
                            'distance': distance
                        })
        
        # Build enhanced feature table data
        headers = [
            "Mouse ID", 
            "Hull Area (μ±σ)",
            "Aspect Ratio (μ±σ)",
            "Compactness (μ±σ)",
            "Velocity (μ±σ)",
            "Acceleration (μ±σ)",
            "Jerk (μ±σ)",
            "Arena Distance (μ±σ)"
        ]
        
        self._discovered_geometric_features = geom_features
        self._social_features = social_features
        
        def format_stat(values):
            """Format mean ± std."""
            if not values:
                return "N/A"
            mean = np.mean(values)
            std = np.std(values)
            return f"{mean:.2f}±{std:.2f}"
        
        # Build table data for virtual table
        table_data = []
        for mouse_id, features in sorted(geom_features.items()):
            row = [
                str(mouse_id),
                format_stat(features['convex_hull_areas']),
                format_stat(features['ellipse_aspect_ratios']),
                format_stat(features['body_compactness']),
                format_stat(features['velocity_magnitudes']),
                format_stat(features['acceleration_magnitudes']),
                format_stat(features['jerk_magnitudes']),
                format_stat(features['distance_to_arena_center'])
            ]
            table_data.append(row)
        
        # Set data using virtual table
        self.geom_features_table.set_data(table_data, headers)
        print(f"[AnalysisPane] Computed {len(social_features)} social feature measurements")
    
    def _compute_separability_analysis(self, payloads: List[FramePayload]) -> None:
        """Analyze feature separability for behavior classification using actual ML metrics."""
        output = f"<h3 style='color: {UI_TEXT_PRIMARY};'>Feature Separability Analysis</h3>"
        output += f"<p style='color: {UI_TEXT_MUTED};'>Finding features that best separate different behaviors...</p>"
        
        # Check if we have behavior annotations
        has_behaviors = any(p.behaviors for p in payloads if isinstance(p, FramePayload))
        
        if not has_behaviors:
            output += f"<p style='color: {UI_TEXT_MUTED};'>No behavior annotations found. Load annotation data to analyze separability.</p>"
            self.separability_content.setHtml(output)
            return
        
        try:
            # Extract features and labels from payloads with behaviors
            feature_matrix = []
            labels = []
            frame_indices = []
            
            for frame_idx, payload in enumerate(payloads):
                if not isinstance(payload, FramePayload) or not payload.behaviors:
                    continue
                
                # Compute geometric features for this frame
                geom_features = self._compute_frame_geometric_features(payload)
                if geom_features is None:
                    continue
                
                # For each mouse with a behavior label
                for mouse_id, behavior in payload.behaviors.items():
                    # Enhanced feature vector with MABe-optimized features
                    feature_vector = [
                        geom_features.get('convex_hull_area', 0),
                        geom_features.get('body_orientation', 0),
                        geom_features.get('pose_spread', 0),
                        geom_features.get('aspect_ratio', 0),
                        geom_features.get('compactness', 0),
                        geom_features.get('min_inter_distance', 0),
                        geom_features.get('group_dispersion', 0),
                        geom_features.get('centroid_x', 0),
                        geom_features.get('centroid_y', 0),
                    ]
                    feature_matrix.append(feature_vector)
                    labels.append(behavior)
                    frame_indices.append(frame_idx)
            
            if len(feature_matrix) < 10:
                output += f"<p style='color: {UI_TEXT_MUTED};'>Insufficient labeled data for analysis (found {len(feature_matrix)} samples).</p>"
                self.separability_content.setHtml(output)
                return
            
            # Convert to numpy arrays
            X = np.array(feature_matrix)
            y = np.array(labels)
            
            # Compute feature importance using variance ratio and mutual information
            try:
                from sklearn.feature_selection import mutual_info_classif
                from sklearn.preprocessing import LabelEncoder
                
                # Encode labels
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                
                # Compute mutual information
                mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
                
                # Compute feature variance
                feature_variance = np.var(X, axis=0)
                
                # Normalize and combine scores
                mi_normalized = mi_scores / (np.max(mi_scores) + 1e-10)
                var_normalized = feature_variance / (np.max(feature_variance) + 1e-10)
                combined_scores = 0.7 * mi_normalized + 0.3 * var_normalized
                
                # Enhanced feature names for MABe competition
                feature_names = [
                    'Convex Hull Area',
                    'Body Orientation',
                    'Pose Spread',
                    'Ellipse Aspect Ratio',
                    'Body Compactness',
                    'Min Inter-Mouse Distance',
                    'Group Dispersion',
                    'Centroid X',
                    'Centroid Y'
                ]
                
                # Sort by score
                feature_rankings = sorted(
                    zip(feature_names, combined_scores, mi_scores, feature_variance),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Display results
                output += f"<h4 style='color: {UI_TEXT_PRIMARY};'>Feature Importance Rankings</h4>"
                output += f"<p style='color: {UI_TEXT_PRIMARY};'>Analyzed {len(feature_matrix)} labeled frames with {len(set(labels))} unique behaviors</p>"
                output += "<table style='width: 100%; border-collapse: collapse; font-family: monospace;'>"
                output += f"<tr style='border-bottom: 2px solid {UI_TEXT_MUTED};'>"
                output += f"<th style='text-align: left; color: {UI_TEXT_PRIMARY}; padding: 8px;'>Feature</th>"
                output += f"<th style='text-align: right; color: {UI_TEXT_PRIMARY}; padding: 8px;'>Combined Score</th>"
                output += f"<th style='text-align: right; color: {UI_TEXT_PRIMARY}; padding: 8px;'>Mutual Info</th>"
                output += f"<th style='text-align: right; color: {UI_TEXT_PRIMARY}; padding: 8px;'>Variance</th>"
                output += "</tr>"
                
                for feature_name, combined, mi, var in feature_rankings:
                    output += "<tr>"
                    output += f"<td style='color: {UI_TEXT_PRIMARY}; padding: 8px;'>{feature_name}</td>"
                    output += f"<td style='color: {UI_TEXT_PRIMARY}; text-align: right; padding: 8px;'>{combined:.4f}</td>"
                    output += f"<td style='color: {UI_TEXT_PRIMARY}; text-align: right; padding: 8px;'>{mi:.4f}</td>"
                    output += f"<td style='color: {UI_TEXT_PRIMARY}; text-align: right; padding: 8px;'>{var:.2f}</td>"
                    output += "</tr>"
                
                output += "</table>"
                
                # Add recommendations
                output += f"<h4 style='color: {UI_TEXT_PRIMARY};'>Recommendations</h4>"
                output += "<ul style='color: {UI_TEXT_PRIMARY};'>"
                
                top_features = [name for name, _, _, _ in feature_rankings[:3]]
                output += f"<li><b>Top 3 features for SVM:</b> {', '.join(top_features)}</li>"
                
                # Behavior distribution
                behavior_counts = {}
                for behavior in labels:
                    behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
                
                output += f"<li><b>Behavior distribution:</b> {len(behavior_counts)} classes"
                if len(behavior_counts) < 20:
                    output += " ("
                    output += ", ".join([f"{b}: {c}" for b, c in sorted(behavior_counts.items(), key=lambda x: x[1], reverse=True)[:5]])
                    output += ")"
                output += "</li>"
                
                # Class balance check
                min_count = min(behavior_counts.values())
                max_count = max(behavior_counts.values())
                if max_count / min_count > 5:
                    output += f"<li><b>Warning:</b> Imbalanced classes detected (ratio {max_count/min_count:.1f}:1). Consider SMOTE or class weights.</li>"
                else:
                    output += f"<li><b>Class balance:</b> Good (ratio {max_count/min_count:.1f}:1)</li>"
                
                output += "</ul>"
                
                output += f"<p style='color: {UI_TEXT_MUTED}; font-style: italic;'>"
                output += "Mutual Information measures the dependency between features and behavior labels. "
                output += "Higher scores indicate more useful features for classification."
                output += "</p>"
                
            except ImportError:
                output += f"<p style='color: {UI_TEXT_MUTED};'>sklearn not available. Install scikit-learn for advanced analysis:</p>"
                output += f"<pre style='color: {UI_TEXT_PRIMARY}; background: {UI_SURFACE}; padding: 10px;'>pip install scikit-learn</pre>"
                
                # Fall back to basic statistics
                output += f"<p style='color: {UI_TEXT_PRIMARY};'><b>Basic Feature Statistics:</b></p>"
                feature_names = ['Convex Hull Area', 'Body Orientation', 'Pose Spread', 'Centroid X', 'Centroid Y']
                output += "<ul style='color: {UI_TEXT_PRIMARY};'>"
                for i, name in enumerate(feature_names):
                    col = X[:, i]
                    output += f"<li>{name}: mean={np.mean(col):.2f}, std={np.std(col):.2f}, range=[{np.min(col):.2f}, {np.max(col):.2f}]</li>"
                output += "</ul>"
        
        except Exception as exc:
            output += f"<p style='color: {UI_TEXT_MUTED};'>Error during separability analysis: {exc}</p>"
            import traceback
            traceback.print_exc()
        
        self.separability_content.setHtml(output)
    
    def _compute_frame_geometric_features(self, payload: FramePayload) -> Optional[Dict[str, float]]:
        """Compute comprehensive geometric features for a single frame (MABe-optimized)."""
        try:
            all_points = []
            for mouse_id, group in payload.mouse_groups.items():
                for point in group.points:
                    if not np.isnan(point).any():
                        all_points.append(point)
            
            if len(all_points) < 3:
                return None
            
            points_array = np.array(all_points)
            centroid = np.mean(points_array, axis=0)
            
            # 1. Convex hull area (body size)
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points_array)
                hull_area = hull.volume  # 2D area
            except:
                hull_area = 0.0
            
            # 2. Body orientation (PCA)
            centered = points_array - centroid
            if len(centered) >= 2:
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                orientation = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                
                # 3. Ellipse aspect ratio (elongation)
                eig_sorted = np.sort(eigenvalues)[::-1]
                aspect_ratio = eig_sorted[0] / (eig_sorted[1] + 1e-10)
            else:
                orientation = 0.0
                aspect_ratio = 1.0
            
            # 4. Pose spread (posture variability)
            pose_spread = np.std(np.linalg.norm(centered, axis=1))
            
            # 5. Body compactness
            mean_radius = np.mean(np.linalg.norm(centered, axis=1))
            compactness = 1.0 / (mean_radius + 1e-10)
            
            # 6. Inter-mouse features (if multiple mice)
            if len(payload.mouse_groups) >= 2:
                mouse_centroids = []
                for mouse_id, group in payload.mouse_groups.items():
                    valid_pts = [p for p in group.points if not np.isnan(p).any()]
                    if valid_pts:
                        mouse_centroids.append(np.mean(valid_pts, axis=0))
                
                if len(mouse_centroids) >= 2:
                    # Min distance between any two mice
                    min_inter_distance = float('inf')
                    for i in range(len(mouse_centroids)):
                        for j in range(i + 1, len(mouse_centroids)):
                            dist = np.linalg.norm(mouse_centroids[i] - mouse_centroids[j])
                            min_inter_distance = min(min_inter_distance, dist)
                    
                    # Group dispersion (std of mouse positions)
                    group_dispersion = np.std([np.linalg.norm(c - centroid) for c in mouse_centroids])
                else:
                    min_inter_distance = 0.0
                    group_dispersion = 0.0
            else:
                min_inter_distance = 0.0
                group_dispersion = 0.0
            
            return {
                'convex_hull_area': float(hull_area),
                'body_orientation': float(orientation),
                'pose_spread': float(pose_spread),
                'centroid_x': float(centroid[0]),
                'centroid_y': float(centroid[1]),
                'aspect_ratio': float(aspect_ratio),
                'compactness': float(compactness),
                'min_inter_distance': float(min_inter_distance),
                'group_dispersion': float(group_dispersion),
            }
        except Exception:
            return None
    
    # REMOVED: 3D point cloud - moved to graphs.py
    # def _update_3d_point_cloud(self, payloads: List[FramePayload]) -> None:
    #     """Update 3D point cloud visualization with behavior-based coloring."""
    #     # This functionality is now in the Graphs tab
    #     pass
    
    def _export_discovered_features(self) -> None:
        """Export discovered geometric features to CSV."""
        if not hasattr(self, '_discovered_geometric_features') or not self.current_path:
            QtWidgets.QMessageBox.warning(
                self,
                "No Features",
                "No discovered features available to export."
            )
            return
        
        # Ask for save location
        default_name = self.current_path.stem + "_discovered_features.csv"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Discovered Features",
            str(self.current_path.parent / default_name),
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            data_rows = []
            
            for mouse_id, features in sorted(self._discovered_geometric_features.items()):
                row = {
                    'mouse_id': mouse_id,
                    'avg_convex_hull_area': np.mean(features['convex_hull_areas']) if features['convex_hull_areas'] else np.nan,
                    'std_convex_hull_area': np.std(features['convex_hull_areas']) if features['convex_hull_areas'] else np.nan,
                    'mean_body_orientation': np.mean(features['body_orientations']) if features['body_orientations'] else np.nan,
                    'std_body_orientation': np.std(features['body_orientations']) if features['body_orientations'] else np.nan,
                    'avg_pose_spread': np.mean(features['pose_spread']) if features['pose_spread'] else np.nan,
                    'std_pose_spread': np.std(features['pose_spread']) if features['pose_spread'] else np.nan,
                }
                data_rows.append(row)
            
            df = pd.DataFrame(data_rows)
            df.to_csv(file_path, index=False)
            
            QtWidgets.QMessageBox.information(
                self,
                "Export Successful",
                f"Discovered features exported to:\n{file_path}"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export:\n{str(exc)}"
            )
    
    def _export_geometric_features(self) -> None:
        """Export geometric features from discovery tab to CSV."""
        if not hasattr(self, '_discovered_geometric_features') or not self.current_path:
            QtWidgets.QMessageBox.warning(
                self,
                "No Features",
                "No geometric features available. Switch to Feature Discovery tab first."
            )
            return
        
        # Same as _export_discovered_features
        self._export_discovered_features()
    
    def _export_separability_report(self) -> None:
        """Export separability analysis report to text file."""
        if not self.separability_content.toPlainText():
            QtWidgets.QMessageBox.warning(
                self,
                "No Report",
                "No separability report available. Switch to Feature Discovery tab first."
            )
            return
        
        # Ask for save location
        default_name = self.current_path.stem + "_separability_report.txt" if self.current_path else "separability_report.txt"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Separability Report",
            str(self.current_path.parent / default_name) if self.current_path else default_name,
            "Text Files (*.txt);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Save the HTML content as plain text
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.separability_content.toPlainText())
            
            QtWidgets.QMessageBox.information(
                self,
                "Export Successful",
                f"Separability report exported to:\n{file_path}"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export:\n{str(exc)}"
            )

