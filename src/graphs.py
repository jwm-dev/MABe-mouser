"""Graphical analysis tab with visualizations for pose tracking data.

This module provides comprehensive visual analysis including trajectories,
3D point clouds, velocity plots, heatmaps, behavior transitions, and more.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt6 import QtCore, QtWidgets
from vispy import scene
from vispy.scene import visuals
from vispy.scene.cameras import ArcballCamera

from .constants import BEHAVIOR_COLORS, UI_ACCENT, UI_BACKGROUND, UI_TEXT_MUTED, UI_TEXT_PRIMARY
from .dataset_stats import DatasetStatsCache
from .models import FramePayload

# Try to import matplotlib for additional plots
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class GraphsPane(QtWidgets.QWidget):
    """Graphical analysis pane with visualizations."""
    
    # Signal emitted when graphs are done updating
    graphs_complete = QtCore.pyqtSignal()
    
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
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Initialize the UI components."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create tab widget for different graph views
        self.graphs_tabs = QtWidgets.QTabWidget(self)
        self.graphs_tabs.setStyleSheet(f"""
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
        
        # Add graph tabs
        self._create_trajectories_tab()
        self._create_3d_cloud_tab()
        self._create_velocity_tab()
        self._create_heatmap_tab()
        self._create_distance_tab()
        self._create_acceleration_tab()
        self._create_behavior_timeline_tab()
        self._create_keypoint_distances_tab()
        
        layout.addWidget(self.graphs_tabs)
    
    def _create_trajectories_tab(self) -> None:
        """Create 2D trajectories visualization tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header = QtWidgets.QLabel("2D Mouse Trajectories")
        header.setStyleSheet(f"color: {UI_TEXT_PRIMARY}; font-size: 16px; font-weight: 600;")
        layout.addWidget(header)
        
        desc = QtWidgets.QLabel(
            "Visualize mouse movement paths over time. "
            "Each mouse is shown in a different color."
        )
        desc.setStyleSheet(f"color: {UI_TEXT_MUTED}; font-size: 13px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        layout.addSpacing(8)
        
        # Vispy canvas for trajectories
        self.trajectory_canvas = scene.SceneCanvas(keys='interactive', show=False)
        self.trajectory_canvas.native.setMinimumHeight(400)
        self.trajectory_view = self.trajectory_canvas.central_widget.add_view()
        self.trajectory_view.camera = 'panzoom'
        layout.addWidget(self.trajectory_canvas.native)
        
        self.graphs_tabs.addTab(widget, "Trajectories")
    
    def _create_3d_cloud_tab(self) -> None:
        """Create 3D point cloud visualization tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header = QtWidgets.QLabel("3D Point Cloud (X, Y, Time)")
        header.setStyleSheet(f"color: {UI_TEXT_PRIMARY}; font-size: 16px; font-weight: 600;")
        layout.addWidget(header)
        
        desc = QtWidgets.QLabel(
            "View pose keypoints in 3D space (X, Y, Time) colored by behavior. "
            "This reveals spatial-temporal patterns and behavior transitions."
        )
        desc.setStyleSheet(f"color: {UI_TEXT_MUTED}; font-size: 13px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        layout.addSpacing(8)
        
        # Vispy 3D canvas
        self.cloud_canvas = scene.SceneCanvas(keys='interactive', show=False, bgcolor='#0d1220')
        self.cloud_canvas.native.setMinimumHeight(400)
        self.cloud_view = self.cloud_canvas.central_widget.add_view()
        
        # Use ArcballCamera for proper 3D interaction
        self.cloud_view.camera = ArcballCamera(fov=45, distance=500)
        
        # Grid for reference
        grid = scene.visuals.GridLines(parent=self.cloud_view.scene)
        
        layout.addWidget(self.cloud_canvas.native)
        
        self.graphs_tabs.addTab(widget, "3D Cloud")
    
    def _create_velocity_tab(self) -> None:
        """Create velocity and acceleration plots tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header = QtWidgets.QLabel("Velocity & Acceleration")
        header.setStyleSheet(f"color: {UI_TEXT_PRIMARY}; font-size: 16px; font-weight: 600;")
        layout.addWidget(header)
        
        desc = QtWidgets.QLabel(
            "Movement speed and acceleration over time. "
            "Spikes in acceleration often indicate behavior changes."
        )
        desc.setStyleSheet(f"color: {UI_TEXT_MUTED}; font-size: 13px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        layout.addSpacing(8)
        
        # Placeholder for matplotlib plot
        if HAS_MATPLOTLIB:
            self.velocity_figure = Figure(figsize=(8, 6), facecolor='#0d1220')
            self.velocity_canvas = FigureCanvasQTAgg(self.velocity_figure)
            layout.addWidget(self.velocity_canvas)
        else:
            placeholder = QtWidgets.QLabel("Install matplotlib for velocity plots")
            placeholder.setStyleSheet(f"color: {UI_TEXT_MUTED};")
            layout.addWidget(placeholder)
        
        layout.addStretch()
        
        self.graphs_tabs.addTab(widget, "Velocity")
    
    def _create_heatmap_tab(self) -> None:
        """Create spatial heatmap tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header = QtWidgets.QLabel("Spatial Heatmap")
        header.setStyleSheet(f"color: {UI_TEXT_PRIMARY}; font-size: 16px; font-weight: 600;")
        layout.addWidget(header)
        
        desc = QtWidgets.QLabel(
            "Heatmap showing arena usage frequency. "
            "Reveals preferred locations and spatial patterns."
        )
        desc.setStyleSheet(f"color: {UI_TEXT_MUTED}; font-size: 13px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        layout.addSpacing(8)
        
        # Placeholder for heatmap
        if HAS_MATPLOTLIB:
            self.heatmap_figure = Figure(figsize=(8, 6), facecolor='#0d1220')
            self.heatmap_canvas = FigureCanvasQTAgg(self.heatmap_figure)
            layout.addWidget(self.heatmap_canvas)
        else:
            placeholder = QtWidgets.QLabel("Install matplotlib for heatmaps")
            placeholder.setStyleSheet(f"color: {UI_TEXT_MUTED};")
            layout.addWidget(placeholder)
        
        layout.addStretch()
        
        self.graphs_tabs.addTab(widget, "Heatmap")
    
    def update_data(self, path: Optional[Path], data: Optional[Dict[str, Any]]) -> None:
        """Update graphs with new data (parallelized for performance)."""
        print(f"[GraphsPane] update_data called with path={path}, data keys={data.keys() if data else None}")
        
        self.current_path = path
        self.current_data = data
        
        if data is None or not data:
            self._clear_all()
            self.graphs_complete.emit()
            return
        
        # Extract payloads
        payloads = data.get("payloads", [])
        print(f"[GraphsPane] Got {len(payloads)} payloads - parallelizing graph updates")
        
        # Update vispy visualizations first (must be on main thread)
        self._update_trajectories(payloads)
        self._update_3d_cloud(payloads)
        
        # Parallelize matplotlib graph rendering
        if HAS_MATPLOTLIB and self._executor:
            from concurrent.futures import as_completed
            
            # Submit all matplotlib updates in parallel
            futures = []
            update_funcs = [
                ('velocity', self._update_velocity),
                ('heatmap', self._update_heatmap),
                ('distance', self._update_distance),
                ('acceleration', self._update_acceleration),
                ('behavior', self._update_behavior_timeline),
                ('keypoint', self._update_keypoint_distances),
            ]
            
            for name, func in update_funcs:
                try:
                    future = self._executor.submit(func, payloads)
                    futures.append((name, future))
                except Exception as e:
                    print(f"[GraphsPane] Failed to submit {name}: {e}")
            
            # Wait for all to complete
            for name, future in futures:
                try:
                    future.result(timeout=30)  # 30 second timeout per graph
                    print(f"[GraphsPane] {name} plot completed")
                except Exception as e:
                    print(f"[GraphsPane] Error updating {name}: {e}")
        elif HAS_MATPLOTLIB:
            # Fallback to sequential if no executor
            self._update_velocity(payloads)
            self._update_heatmap(payloads)
            self._update_distance(payloads)
            self._update_acceleration(payloads)
            self._update_behavior_timeline(payloads)
            self._update_keypoint_distances(payloads)
        
        print(f"[GraphsPane] All graphs updated successfully")
        self.graphs_complete.emit()
    
    def _prepare_data_for_cache(self, path: Path, data: Dict[str, Any]) -> None:
        """Prepare graph data in background for instant display later."""
        print(f"[GraphsPane] Preparing cache data for {path.name}")
        
        # Extract payloads
        payloads = data.get("payloads", [])
        if not payloads:
            print(f"[GraphsPane] No payloads to cache")
            return
        
        # Pre-calculate heavy computations that graphs will need
        # This runs in a background thread, results get cached
        
        # For trajectory data: pre-compute centroids
        mouse_centroids = {}
        for payload in payloads:
            if not isinstance(payload, FramePayload):
                continue
            for mouse_id, group in payload.mouse_groups.items():
                points_array = np.array(group.points)
                valid_mask = ~np.isnan(points_array).any(axis=1)
                valid_points = points_array[valid_mask]
                if len(valid_points) > 0:
                    centroid = np.mean(valid_points, axis=0)
                    if mouse_id not in mouse_centroids:
                        mouse_centroids[mouse_id] = []
                    mouse_centroids[mouse_id].append(centroid)
        
        # Store in smart cache
        from .smart_cache import get_cache_manager
        cache_mgr = get_cache_manager()
        cache_key = cache_mgr.get_cache_key(path=path, params={'type': 'graphs'})
        cache_mgr.analysis_cache.put(cache_key, {
            'centroids': mouse_centroids,
            'n_payloads': len(payloads),
            'prepared': True
        })
        
        print(f"[GraphsPane] Cache prepared for {path.name}")
    
    def _clear_all(self) -> None:
        """Clear all visualizations."""
        # Clear trajectories
        if hasattr(self, 'trajectory_view'):
            self.trajectory_view.scene.visuals = []
        
        # Clear 3D cloud
        if hasattr(self, 'cloud_view'):
            for visual in list(self.cloud_view.scene.children):
                if isinstance(visual, (visuals.Markers, visuals.Line)):
                    visual.parent = None
        
        # Clear matplotlib plots
        if HAS_MATPLOTLIB:
            for attr_name in ['velocity_figure', 'heatmap_figure', 'distance_figure', 
                             'accel_figure', 'behavior_figure', 'keypoint_figure']:
                if hasattr(self, attr_name):
                    fig = getattr(self, attr_name)
                    fig.clear()
                    # Get corresponding canvas
                    canvas_name = attr_name.replace('_figure', '_canvas')
                    if hasattr(self, canvas_name):
                        canvas = getattr(self, canvas_name)
                        canvas.draw()
    
    def _update_trajectories(self, payloads: List[FramePayload]) -> None:
        """Update 2D trajectory visualization (optimized with vectorization)."""
        if not payloads:
            return
        
        print(f"[GraphsPane] Updating trajectories with {len(payloads)} frames")
        
        # Clear existing trajectories
        for visual in list(self.trajectory_view.scene.children):
            if isinstance(visual, visuals.Line):
                visual.parent = None
        
        # Collect trajectories per mouse (vectorized)
        mouse_trajectories: Dict[str, List[np.ndarray]] = {}
        
        # Batch process frames for better performance
        for payload in payloads:
            if not isinstance(payload, FramePayload):
                continue
            
            for mouse_id, group in payload.mouse_groups.items():
                # Filter valid points vectorized
                points_array = np.array(group.points)
                valid_mask = ~np.isnan(points_array).any(axis=1)
                valid_points = points_array[valid_mask]
                
                if len(valid_points) == 0:
                    continue
                
                # Compute centroid (vectorized)
                centroid = np.mean(valid_points, axis=0)
                
                if mouse_id not in mouse_trajectories:
                    mouse_trajectories[mouse_id] = []
                mouse_trajectories[mouse_id].append(centroid)
        
        # Plot trajectories
        colors = [
            (1.0, 0.3, 0.3),  # Red
            (0.3, 0.6, 1.0),  # Blue
            (0.3, 1.0, 0.3),  # Green
            (1.0, 0.8, 0.2),  # Yellow
        ]
        
        for idx, (mouse_id, trajectory) in enumerate(mouse_trajectories.items()):
            if len(trajectory) < 2:
                continue
            
            points = np.array(trajectory)
            color = colors[idx % len(colors)]
            
            line = visuals.Line(
                pos=points,
                color=color,
                width=2,
                parent=self.trajectory_view.scene
            )
        
        # Auto-fit camera
        if mouse_trajectories:
            all_points = []
            for traj in mouse_trajectories.values():
                all_points.extend(traj)
            
            if all_points:
                all_points = np.array(all_points)
                x_min, y_min = np.min(all_points, axis=0)
                x_max, y_max = np.max(all_points, axis=0)
                
                # Add margin
                margin = 0.1 * max(x_max - x_min, y_max - y_min)
                self.trajectory_view.camera.set_range(
                    x=(x_min - margin, x_max + margin),
                    y=(y_min - margin, y_max + margin)
                )
    
    def _update_3d_cloud(self, payloads: List[FramePayload]) -> None:
        """Update 3D point cloud visualization."""
        if not payloads:
            return
        
        print(f"[GraphsPane] Updating 3D cloud with {len(payloads)} frames")
        
        # Clear existing visuals
        for visual in list(self.cloud_view.scene.children):
            if isinstance(visual, (visuals.Markers, visuals.Line)):
                visual.parent = None
        
        # Collect all points with behavior colors
        all_points = []
        all_colors = []
        
        # Sample frames to avoid too many points (every Nth frame)
        sample_rate = max(1, len(payloads) // 1000)  # Max 1000 frames
        
        for frame_idx in range(0, len(payloads), sample_rate):
            payload = payloads[frame_idx]
            if not isinstance(payload, FramePayload):
                continue
            
            # Get behavior color for this frame/mouse
            for mouse_id, group in payload.mouse_groups.items():
                # Get behavior for this specific mouse
                behavior_label = payload.behaviors.get(mouse_id, None)
                if behavior_label and behavior_label in BEHAVIOR_COLORS:
                    color = BEHAVIOR_COLORS[behavior_label]
                else:
                    color = (0.7, 0.7, 0.7)  # Gray for unknown
                
                # Add all keypoints from this mouse
                for point in group.points:
                    if not np.isnan(point).any():
                        # Create 3D point: (x, y, frame_idx)
                        point_3d = np.array([float(point[0]), float(point[1]), float(frame_idx)])
                        all_points.append(point_3d)
                        all_colors.append(color)
        
        if all_points:
            points_array = np.array(all_points)
            colors_array = np.array(all_colors)
            
            # Create scatter plot
            markers = visuals.Markers(parent=self.cloud_view.scene)
            markers.set_data(
                points_array,
                edge_color=None,
                face_color=colors_array,
                size=5
            )
            
            print(f"[GraphsPane] Rendered {len(all_points)} points in 3D cloud")
    
    def _update_velocity(self, payloads: List[FramePayload]) -> None:
        """Update velocity plots (optimized with vectorization)."""
        if not HAS_MATPLOTLIB or not payloads:
            return
        
        try:
            print(f"[GraphsPane] Updating velocity plots")
            
            self.velocity_figure.clear()
            ax = self.velocity_figure.add_subplot(111)
            self._style_axis(ax)
            
            # Compute velocities per mouse (vectorized)
            mouse_trajectories: Dict[str, np.ndarray] = {}
            
            # First pass: collect all points per mouse
            temp_trajectories: Dict[str, List[np.ndarray]] = {}
            for payload in payloads:
                if not isinstance(payload, FramePayload):
                    continue
                
                for mouse_id, group in payload.mouse_groups.items():
                    points_array = np.array(group.points)
                    valid_mask = ~np.isnan(points_array).any(axis=1)
                    valid_points = points_array[valid_mask]
                    
                    if len(valid_points) == 0:
                        continue
                    
                    centroid = np.mean(valid_points, axis=0)
                    
                    if mouse_id not in temp_trajectories:
                        temp_trajectories[mouse_id] = []
                    temp_trajectories[mouse_id].append(centroid)
            
            # Convert to numpy arrays for vectorized operations
            for mouse_id, traj_list in temp_trajectories.items():
                if len(traj_list) >= 2:
                    mouse_trajectories[mouse_id] = np.array(traj_list)
            
            # Compute and plot velocities (vectorized)
            colors = ['#ff4444', '#4488ff', '#44ff44', '#ffcc22']
            
            for idx, (mouse_id, positions) in enumerate(mouse_trajectories.items()):
                # Vectorized velocity calculation: ||p[i+1] - p[i]||
                velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                
                # Assume 30 fps
                time = np.arange(len(velocities)) / 30.0
                
                color = colors[idx % len(colors)]
                ax.plot(time, velocities, color=color, label=f"Mouse {mouse_id}", linewidth=1.5)
            
            ax.set_xlabel('Time (s)', color='#f1f5ff', fontsize=11)
            ax.set_ylabel('Speed (pixels/frame)', color='#f1f5ff', fontsize=11)
            ax.set_title('Mouse Movement Speed', color='#f1f5ff', fontsize=14)
            
            # Use loc='upper right' instead of 'best' for performance
            if len(mouse_trajectories) > 0:
                ax.legend(facecolor='#1a2332', edgecolor='#8b95a8', labelcolor='#f1f5ff', 
                         loc='upper right', framealpha=0.9)
            ax.grid(True, alpha=0.2, color='#8b95a8')
            
            self.velocity_canvas.draw()
            
        except Exception as e:
            print(f"[GraphsPane] Error updating velocity plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_heatmap(self, payloads: List[FramePayload]) -> None:
        """Update spatial heatmap."""
        if not HAS_MATPLOTLIB or not payloads:
            return
        
        try:
            print(f"[GraphsPane] Updating spatial heatmap")
            
            self.heatmap_figure.clear()
            ax = self.heatmap_figure.add_subplot(111)
            self._style_axis(ax)
            
            # Collect all positions (vectorized for performance)
            all_positions = []
            
            for payload in payloads:
                if not isinstance(payload, FramePayload):
                    continue
                
                for mouse_id, group in payload.mouse_groups.items():
                    points_array = np.array(group.points)
                    # Filter out NaN values
                    valid_mask = ~np.isnan(points_array).any(axis=1)
                    valid_points = points_array[valid_mask]
                    if len(valid_points) > 0:
                        all_positions.append(valid_points[:, :2])  # x, y only
            
            if not all_positions:
                ax.text(0.5, 0.5, 'No position data available', 
                       ha='center', va='center', color='#8b95a8', transform=ax.transAxes)
                self.heatmap_canvas.draw()
                return
            
            # Concatenate all positions
            positions = np.vstack(all_positions)
            
            # Create 2D histogram
            heatmap, xedges, yedges = np.histogram2d(
                positions[:, 0],
                positions[:, 1],
                bins=50
            )
            
            # Plot heatmap
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(
                heatmap.T,
                extent=extent,
                origin='lower',
                cmap='hot',
                aspect='auto',
                interpolation='bilinear'
            )
            
            ax.set_xlabel('X Position (pixels)', color='#f1f5ff', fontsize=11)
            ax.set_ylabel('Y Position (pixels)', color='#f1f5ff', fontsize=11)
            ax.set_title('Arena Usage Heatmap', color='#f1f5ff', fontsize=14, pad=12)
            ax.tick_params(colors='#8b95a8', labelsize=10)
            
            # Colorbar
            cbar = self.heatmap_figure.colorbar(im, ax=ax)
            cbar.set_label('Frequency', color='#f1f5ff')
            cbar.ax.tick_params(colors='#8b95a8')
            
            self.heatmap_canvas.draw()
            
        except Exception as e:
            print(f"[GraphsPane] Error updating heatmap: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_distance_tab(self) -> None:
        """Create inter-mouse distance plot tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header = QtWidgets.QLabel("Inter-Mouse Distance")
        header.setStyleSheet(f"color: {UI_TEXT_PRIMARY}; font-size: 16px; font-weight: 600;")
        layout.addWidget(header)
        
        desc = QtWidgets.QLabel(
            "Distance between mice over time. "
            "Useful for detecting social interactions and proximity events."
        )
        desc.setStyleSheet(f"color: {UI_TEXT_MUTED}; font-size: 13px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        layout.addSpacing(8)
        
        if HAS_MATPLOTLIB:
            self.distance_figure = Figure(figsize=(10, 6), facecolor='#0d1220')
            self.distance_canvas = FigureCanvasQTAgg(self.distance_figure)
            layout.addWidget(self.distance_canvas)
        else:
            placeholder = QtWidgets.QLabel("Install matplotlib for distance plots")
            placeholder.setStyleSheet(f"color: {UI_TEXT_MUTED};")
            layout.addWidget(placeholder)
        
        layout.addStretch()
        self.graphs_tabs.addTab(widget, "Distance")
    
    def _create_acceleration_tab(self) -> None:
        """Create acceleration plot tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header = QtWidgets.QLabel("Acceleration Profile")
        header.setStyleSheet(f"color: {UI_TEXT_PRIMARY}; font-size: 16px; font-weight: 600;")
        layout.addWidget(header)
        
        desc = QtWidgets.QLabel(
            "Movement acceleration (rate of velocity change). "
            "Sharp peaks often indicate behavior transitions or sudden movements."
        )
        desc.setStyleSheet(f"color: {UI_TEXT_MUTED}; font-size: 13px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        layout.addSpacing(8)
        
        if HAS_MATPLOTLIB:
            self.accel_figure = Figure(figsize=(10, 6), facecolor='#0d1220')
            self.accel_canvas = FigureCanvasQTAgg(self.accel_figure)
            layout.addWidget(self.accel_canvas)
        else:
            placeholder = QtWidgets.QLabel("Install matplotlib for acceleration plots")
            placeholder.setStyleSheet(f"color: {UI_TEXT_MUTED};")
            layout.addWidget(placeholder)
        
        layout.addStretch()
        self.graphs_tabs.addTab(widget, "Acceleration")
    
    def _create_behavior_timeline_tab(self) -> None:
        """Create behavior timeline visualization tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header = QtWidgets.QLabel("Behavior Timeline")
        header.setStyleSheet(f"color: {UI_TEXT_PRIMARY}; font-size: 16px; font-weight: 600;")
        layout.addWidget(header)
        
        desc = QtWidgets.QLabel(
            "Timeline showing behavior annotations over time. "
            "Each behavior is color-coded for easy identification."
        )
        desc.setStyleSheet(f"color: {UI_TEXT_MUTED}; font-size: 13px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        layout.addSpacing(8)
        
        if HAS_MATPLOTLIB:
            self.behavior_figure = Figure(figsize=(12, 4), facecolor='#0d1220')
            self.behavior_canvas = FigureCanvasQTAgg(self.behavior_figure)
            layout.addWidget(self.behavior_canvas)
        else:
            placeholder = QtWidgets.QLabel("Install matplotlib for behavior timeline")
            placeholder.setStyleSheet(f"color: {UI_TEXT_MUTED};")
            layout.addWidget(placeholder)
        
        layout.addStretch()
        self.graphs_tabs.addTab(widget, "Behaviors")
    
    def _create_keypoint_distances_tab(self) -> None:
        """Create keypoint distance distribution tab."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header = QtWidgets.QLabel("Keypoint Spread Analysis")
        header.setStyleSheet(f"color: {UI_TEXT_PRIMARY}; font-size: 16px; font-weight: 600;")
        layout.addWidget(header)
        
        desc = QtWidgets.QLabel(
            "Distribution of distances between keypoints (body part spread). "
            "Shows posture variation and body configuration patterns."
        )
        desc.setStyleSheet(f"color: {UI_TEXT_MUTED}; font-size: 13px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        layout.addSpacing(8)
        
        if HAS_MATPLOTLIB:
            self.keypoint_figure = Figure(figsize=(10, 6), facecolor='#0d1220')
            self.keypoint_canvas = FigureCanvasQTAgg(self.keypoint_figure)
            layout.addWidget(self.keypoint_canvas)
        else:
            placeholder = QtWidgets.QLabel("Install matplotlib for keypoint analysis")
            placeholder.setStyleSheet(f"color: {UI_TEXT_MUTED};")
            layout.addWidget(placeholder)
        
        layout.addStretch()
        self.graphs_tabs.addTab(widget, "Keypoints")
    
    def _update_distance(self, payloads: List[FramePayload]) -> None:
        """Update inter-mouse distance plot."""
        if not HAS_MATPLOTLIB or not payloads:
            return
        
        try:
            print(f"[GraphsPane] Updating distance plot")
            
            self.distance_figure.clear()
            ax = self.distance_figure.add_subplot(111)
            self._style_axis(ax)
            
            # Collect centroids per mouse per frame
            mouse_positions = {}  # {frame_idx: {mouse_id: position}}
            
            for frame_idx, payload in enumerate(payloads):
                if not isinstance(payload, FramePayload):
                    continue
                
                if frame_idx not in mouse_positions:
                    mouse_positions[frame_idx] = {}
                
                for mouse_id, group in payload.mouse_groups.items():
                    points_array = np.array(group.points)
                    valid_mask = ~np.isnan(points_array).any(axis=1)
                    valid_points = points_array[valid_mask]
                    if len(valid_points) > 0:
                        centroid = np.mean(valid_points, axis=0)
                        mouse_positions[frame_idx][mouse_id] = centroid
            
            # Compute pairwise distances
            mouse_ids = list(set(mid for frame in mouse_positions.values() for mid in frame.keys()))
            
            if len(mouse_ids) < 2:
                ax.text(0.5, 0.5, 'Need at least 2 mice for distance plot', 
                       ha='center', va='center', color='#8b95a8', transform=ax.transAxes)
                self.distance_canvas.draw()
                return
            
            # Calculate distances for each pair
            time = []
            distances = {f"{mouse_ids[i]}-{mouse_ids[j]}": [] 
                        for i in range(len(mouse_ids)) for j in range(i+1, len(mouse_ids))}
            
            for frame_idx in sorted(mouse_positions.keys()):
                positions = mouse_positions[frame_idx]
                time.append(frame_idx / 30.0)  # Assume 30 fps
                
                for i in range(len(mouse_ids)):
                    for j in range(i+1, len(mouse_ids)):
                        pair_key = f"{mouse_ids[i]}-{mouse_ids[j]}"
                        
                        if mouse_ids[i] in positions and mouse_ids[j] in positions:
                            pos_i = positions[mouse_ids[i]]
                            pos_j = positions[mouse_ids[j]]
                            dist = np.linalg.norm(pos_i - pos_j)
                            distances[pair_key].append(dist)
                        else:
                            distances[pair_key].append(np.nan)
            
            # Plot each pair
            colors = ['#ff4444', '#4488ff', '#44ff44', '#ffcc22', '#ff44ff', '#44ffff']
            for idx, (pair, dist_values) in enumerate(distances.items()):
                color = colors[idx % len(colors)]
                ax.plot(time, dist_values, color=color, label=f"Mice {pair}", linewidth=1.5, alpha=0.8)
            
            ax.set_xlabel('Time (s)', color='#f1f5ff', fontsize=11)
            ax.set_ylabel('Distance (pixels)', color='#f1f5ff', fontsize=11)
            ax.set_title('Inter-Mouse Distance Over Time', color='#f1f5ff', fontsize=14, pad=12)
            ax.legend(facecolor='#1a2332', edgecolor='#8b95a8', labelcolor='#f1f5ff', 
                     framealpha=0.9, loc='upper right')
            ax.grid(True, alpha=0.15, color='#8b95a8', linestyle='--')
            
            self.distance_canvas.draw()
            
        except Exception as e:
            print(f"[GraphsPane] Error updating distance plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_acceleration(self, payloads: List[FramePayload]) -> None:
        """Update acceleration plot."""
        if not HAS_MATPLOTLIB or not payloads:
            return
        
        print(f"[GraphsPane] Updating acceleration plot")
        
        try:
            self.accel_figure.clear()
            ax = self.accel_figure.add_subplot(111)
            self._style_axis(ax)
            
            # Compute accelerations per mouse (vectorized where possible)
            mouse_trajectories: Dict[str, List[np.ndarray]] = {}
            
            for payload in payloads:
                if not isinstance(payload, FramePayload):
                    continue
                
                for mouse_id, group in payload.mouse_groups.items():
                    # Vectorized filtering of valid points
                    points_array = np.array(group.points)
                    valid_mask = ~np.isnan(points_array).any(axis=1)
                    valid_points = points_array[valid_mask]
                    
                    if len(valid_points) == 0:
                        continue
                    
                    centroid = np.mean(valid_points, axis=0)
                    
                    if mouse_id not in mouse_trajectories:
                        mouse_trajectories[mouse_id] = []
                    mouse_trajectories[mouse_id].append(centroid)
            
            # Compute and plot accelerations
            colors = ['#ff4444', '#4488ff', '#44ff44', '#ffcc22']
            
            for idx, (mouse_id, trajectory) in enumerate(mouse_trajectories.items()):
                if len(trajectory) < 3:
                    continue
                
                positions = np.array(trajectory)
                velocities = np.diff(positions, axis=0)
                accelerations = np.diff(velocities, axis=0)
                accel_magnitudes = np.linalg.norm(accelerations, axis=1)
                
                # Assume 30 fps
                time = np.arange(len(accel_magnitudes)) / 30.0
                
                color = colors[idx % len(colors)]
                ax.plot(time, accel_magnitudes, color=color, label=f"Mouse {mouse_id}", 
                       linewidth=1.5, alpha=0.8)
                
                # Add smoothed version
                if len(accel_magnitudes) > 10:
                    from scipy.ndimage import uniform_filter1d
                    smoothed = uniform_filter1d(accel_magnitudes, size=10)
                    ax.plot(time, smoothed, color=color, linestyle='--', linewidth=2, alpha=0.4)
            
            ax.set_xlabel('Time (s)', color='#f1f5ff', fontsize=11)
            ax.set_ylabel('Acceleration (pixels/frame²)', color='#f1f5ff', fontsize=11)
            ax.set_title('Movement Acceleration (solid) with Smoothed Trend (dashed)', 
                        color='#f1f5ff', fontsize=14, pad=12)
            ax.legend(facecolor='#1a2332', edgecolor='#8b95a8', labelcolor='#f1f5ff', 
                     framealpha=0.9, loc='upper right')
            ax.grid(True, alpha=0.15, color='#8b95a8', linestyle='--')
            
            self.accel_canvas.draw()
        except Exception as e:
            print(f"Error updating acceleration: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_behavior_timeline(self, payloads: List[FramePayload]) -> None:
        """Update behavior timeline visualization."""
        if not HAS_MATPLOTLIB or not payloads:
            return
        
        print(f"[GraphsPane] Updating behavior timeline")
        
        try:
            self.behavior_figure.clear()
            ax = self.behavior_figure.add_subplot(111)
            self._style_axis(ax)
            
            # Collect behaviors
            behaviors_over_time = []
            time_points = []
            
            for frame_idx, payload in enumerate(payloads):
                behavior = getattr(payload, 'behavior_label', None)
                behaviors_over_time.append(behavior if behavior else 'unknown')
                time_points.append(frame_idx / 30.0)  # Assume 30 fps
            
            # Get unique behaviors
            unique_behaviors = list(set(behaviors_over_time))
            behavior_to_num = {b: idx for idx, b in enumerate(unique_behaviors)}
            
            # Convert to numeric for plotting
            behavior_nums = [behavior_to_num[b] for b in behaviors_over_time]
            
            # Create color map
            colors_map = []
            for behavior in behaviors_over_time:
                if behavior in BEHAVIOR_COLORS:
                    colors_map.append(BEHAVIOR_COLORS[behavior])
                else:
                    colors_map.append((0.5, 0.5, 0.5))
            
            # Plot as colored segments
            for i in range(len(time_points) - 1):
                ax.axvspan(time_points[i], time_points[i+1], 
                          facecolor=colors_map[i], alpha=0.7)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=BEHAVIOR_COLORS.get(b, (0.5, 0.5, 0.5)), 
                                    label=b, alpha=0.7) for b in unique_behaviors]
            ax.legend(handles=legend_elements, facecolor='#1a2332', edgecolor='#8b95a8', 
                     labelcolor='#f1f5ff', framealpha=0.9, loc='upper right')
            
            ax.set_xlabel('Time (s)', color='#f1f5ff', fontsize=11)
            ax.set_ylabel('Behavior', color='#f1f5ff', fontsize=11)
            ax.set_title('Behavior Annotations Timeline', color='#f1f5ff', fontsize=14, pad=12)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.grid(True, axis='x', alpha=0.15, color='#8b95a8', linestyle='--')
            
            self.behavior_canvas.draw()
        except Exception as e:
            print(f"Error updating behavior timeline: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_keypoint_distances(self, payloads: List[FramePayload]) -> None:
        """Update keypoint distance distribution."""
        if not HAS_MATPLOTLIB or not payloads:
            return
        
        print(f"[GraphsPane] Updating keypoint distances")
        
        try:
            self.keypoint_figure.clear()
            ax = self.keypoint_figure.add_subplot(111)
            self._style_axis(ax)
            
            # Collect all pairwise keypoint distances (vectorized where possible)
            all_distances = []
            
            for payload in payloads:
                if not isinstance(payload, FramePayload):
                    continue
                
                for mouse_id, group in payload.mouse_groups.items():
                    points = np.array(group.points)
                    if len(points) < 2:
                        continue
                    
                    # Filter out NaN points
                    valid_mask = ~np.isnan(points).any(axis=1)
                    valid_points = points[valid_mask]
                    
                    if len(valid_points) < 2:
                        continue
                    
                    # Vectorized pairwise distance computation
                    # For each point, compute distance to all other points
                    for i in range(len(valid_points)):
                        # Compute distances from point i to all points j > i
                        diffs = valid_points[i+1:] - valid_points[i]
                        dists = np.linalg.norm(diffs, axis=1)
                        all_distances.extend(dists.tolist())
            
            if not all_distances:
                ax.text(0.5, 0.5, 'No keypoint data available', 
                       ha='center', va='center', color='#8b95a8', transform=ax.transAxes)
                self.keypoint_canvas.draw()
                return
            
            # Create histogram
            ax.hist(all_distances, bins=50, color='#4488ff', alpha=0.7, edgecolor='#66aaff')
            
            # Add stats
            distances_array = np.array(all_distances)
            mean_dist = np.mean(distances_array)
            median_dist = np.median(distances_array)
            std_dist = np.std(distances_array)
            
            ax.axvline(mean_dist, color='#ff4444', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_dist:.1f}')
            ax.axvline(median_dist, color='#44ff44', linestyle='--', linewidth=2, 
                      label=f'Median: {median_dist:.1f}')
            
            ax.set_xlabel('Distance (pixels)', color='#f1f5ff', fontsize=11)
            ax.set_ylabel('Frequency', color='#f1f5ff', fontsize=11)
            ax.set_title(f'Keypoint Distance Distribution (σ={std_dist:.1f})', 
                        color='#f1f5ff', fontsize=14, pad=12)
            ax.legend(facecolor='#1a2332', edgecolor='#8b95a8', labelcolor='#f1f5ff', 
                     framealpha=0.9, loc='upper right')
            ax.grid(True, alpha=0.15, color='#8b95a8', linestyle='--')
            
            self.keypoint_canvas.draw()
        except Exception as e:
            print(f"Error updating keypoint distances: {e}")
            import traceback
            traceback.print_exc()
    
    def _style_axis(self, ax) -> None:
        """Apply consistent styling to matplotlib axis."""
        ax.set_facecolor('#0d1220')
        ax.tick_params(colors='#8b95a8', labelsize=10)
        ax.spines['bottom'].set_color('#8b95a8')
        ax.spines['left'].set_color('#8b95a8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
