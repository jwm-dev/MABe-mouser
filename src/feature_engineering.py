"""Advanced feature engineering for behavior prediction from pose tracking data.

This module extracts comprehensive features from pose data to enable
behavior classification and prediction. Features are designed to maximize
separability between different behaviors in the feature space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import ConvexHull, distance
from scipy.signal import find_peaks, welch

from .models import FramePayload


class FeatureExtractor:
    """Extract comprehensive behavioral features from pose tracking data."""
    
    @staticmethod
    def extract_all_features(
        payloads: List[FramePayload],
        arena_bounds: Optional[Tuple[float, float, float, float]] = None,
        fps: float = 30.0
    ) -> pd.DataFrame:
        """Extract all features from pose data.
        
        Args:
            payloads: List of frame payloads with pose data
            arena_bounds: (xmin, xmax, ymin, ymax) arena boundaries
            fps: Frames per second for temporal features
            
        Returns:
            DataFrame with one row per frame and columns for all features
        """
        if not payloads:
            return pd.DataFrame()
        
        features_list = []
        
        # Extract features for each frame
        for i, payload in enumerate(payloads):
            frame_features = {}
            frame_features['frame'] = i
            
            # Get mouse positions for this frame
            mouse_positions = {}
            if hasattr(payload, 'points') and payload.points:
                for mouse_id, points in payload.points.items():
                    if points and len(points) > 0:
                        mouse_positions[mouse_id] = points
            
            # Geometric features (per frame)
            geom_features = FeatureExtractor._extract_frame_geometric_features(
                mouse_positions, arena_bounds
            )
            frame_features.update(geom_features)
            
            # Behavior label if available
            if hasattr(payload, 'behavior_label'):
                frame_features['behavior'] = payload.behavior_label
            
            features_list.append(frame_features)
        
        df = pd.DataFrame(features_list)
        
        # Temporal features (require multiple frames)
        if len(df) > 1:
            temporal_features = FeatureExtractor._extract_temporal_features(df, fps)
            df = pd.concat([df, temporal_features], axis=1)
        
        # Statistical features (rolling windows)
        if len(df) > 10:
            statistical_features = FeatureExtractor._extract_statistical_features(df)
            df = pd.concat([df, statistical_features], axis=1)
        
        return df
    
    @staticmethod
    def _extract_frame_geometric_features(
        mouse_positions: Dict[str, np.ndarray],
        arena_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> Dict[str, float]:
        """Extract geometric features from a single frame.
        
        Args:
            mouse_positions: Dict mapping mouse_id to array of keypoints
            arena_bounds: (xmin, xmax, ymin, ymax)
            
        Returns:
            Dictionary of geometric features
        """
        features = {}
        
        if not mouse_positions:
            return features
        
        # Per-mouse features
        for mouse_id, points in mouse_positions.items():
            if len(points) == 0:
                continue
            
            prefix = f"mouse_{mouse_id}_"
            
            # Centroid
            centroid = np.mean(points, axis=0)
            features[f"{prefix}centroid_x"] = float(centroid[0])
            features[f"{prefix}centroid_y"] = float(centroid[1])
            
            # Bounding box
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            bbox_width = float(np.max(x_coords) - np.min(x_coords))
            bbox_height = float(np.max(y_coords) - np.min(y_coords))
            features[f"{prefix}bbox_width"] = bbox_width
            features[f"{prefix}bbox_height"] = bbox_height
            features[f"{prefix}bbox_aspect"] = bbox_width / max(bbox_height, 1e-6)
            features[f"{prefix}bbox_area"] = bbox_width * bbox_height
            
            # Convex hull (if enough points)
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    features[f"{prefix}hull_area"] = float(hull.area)
                    features[f"{prefix}hull_perimeter"] = float(hull.area)  # Approximate
                except Exception:
                    features[f"{prefix}hull_area"] = 0.0
                    features[f"{prefix}hull_perimeter"] = 0.0
            
            # Spread (variance of points)
            features[f"{prefix}spread_x"] = float(np.var(x_coords))
            features[f"{prefix}spread_y"] = float(np.var(y_coords))
            
            # Distance to arena features
            if arena_bounds:
                xmin, xmax, ymin, ymax = arena_bounds
                arena_center_x = (xmin + xmax) / 2
                arena_center_y = (ymin + ymax) / 2
                
                # Distance to center
                dist_to_center = np.sqrt(
                    (centroid[0] - arena_center_x)**2 + 
                    (centroid[1] - arena_center_y)**2
                )
                features[f"{prefix}dist_to_center"] = float(dist_to_center)
                
                # Distance to nearest wall
                dist_to_walls = [
                    abs(centroid[0] - xmin),  # left
                    abs(centroid[0] - xmax),  # right
                    abs(centroid[1] - ymin),  # bottom
                    abs(centroid[1] - ymax),  # top
                ]
                features[f"{prefix}dist_to_wall"] = float(min(dist_to_walls))
                
                # Corner proximity (distance to nearest corner)
                corners = [
                    (xmin, ymin), (xmin, ymax),
                    (xmax, ymin), (xmax, ymax)
                ]
                corner_dists = [
                    np.sqrt((centroid[0] - cx)**2 + (centroid[1] - cy)**2)
                    for cx, cy in corners
                ]
                features[f"{prefix}dist_to_corner"] = float(min(corner_dists))
        
        # Inter-mouse features (social)
        mouse_ids = list(mouse_positions.keys())
        if len(mouse_ids) >= 2:
            centroids = {}
            for mouse_id, points in mouse_positions.items():
                if len(points) > 0:
                    centroids[mouse_id] = np.mean(points, axis=0)
            
            # All pairwise distances
            for i, id1 in enumerate(mouse_ids):
                for id2 in mouse_ids[i+1:]:
                    if id1 in centroids and id2 in centroids:
                        dist = np.linalg.norm(centroids[id1] - centroids[id2])
                        features[f"dist_{id1}_{id2}"] = float(dist)
            
            # Nearest neighbor distance for each mouse
            for mouse_id in mouse_ids:
                if mouse_id not in centroids:
                    continue
                other_dists = []
                for other_id in mouse_ids:
                    if other_id != mouse_id and other_id in centroids:
                        dist = np.linalg.norm(centroids[mouse_id] - centroids[other_id])
                        other_dists.append(dist)
                if other_dists:
                    features[f"mouse_{mouse_id}_nearest_neighbor_dist"] = float(min(other_dists))
        
        return features
    
    @staticmethod
    def _extract_temporal_features(
        df: pd.DataFrame,
        fps: float = 30.0
    ) -> pd.DataFrame:
        """Extract temporal features (velocity, acceleration, etc).
        
        Args:
            df: DataFrame with geometric features per frame
            fps: Frames per second
            
        Returns:
            DataFrame with temporal features
        """
        temporal_features = pd.DataFrame(index=df.index)
        dt = 1.0 / fps
        
        # Find all centroid columns
        centroid_x_cols = [col for col in df.columns if 'centroid_x' in col]
        centroid_y_cols = [col for col in df.columns if 'centroid_y' in col]
        
        for x_col, y_col in zip(centroid_x_cols, centroid_y_cols):
            mouse_id = x_col.split('_')[1]  # Extract mouse ID
            prefix = f"mouse_{mouse_id}_"
            
            x = df[x_col].values
            y = df[y_col].values
            
            # Velocity
            vx = np.gradient(x, dt)
            vy = np.gradient(y, dt)
            speed = np.sqrt(vx**2 + vy**2)
            temporal_features[f"{prefix}velocity_x"] = vx
            temporal_features[f"{prefix}velocity_y"] = vy
            temporal_features[f"{prefix}speed"] = speed
            
            # Acceleration
            ax = np.gradient(vx, dt)
            ay = np.gradient(vy, dt)
            accel_mag = np.sqrt(ax**2 + ay**2)
            temporal_features[f"{prefix}accel_x"] = ax
            temporal_features[f"{prefix}accel_y"] = ay
            temporal_features[f"{prefix}accel_mag"] = accel_mag
            
            # Jerk (rate of change of acceleration)
            jerk_x = np.gradient(ax, dt)
            jerk_y = np.gradient(ay, dt)
            jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2)
            temporal_features[f"{prefix}jerk_mag"] = jerk_mag
            
            # Heading (direction of movement)
            heading = np.arctan2(vy, vx)
            temporal_features[f"{prefix}heading"] = heading
            
            # Heading change (angular velocity)
            heading_change = np.gradient(heading, dt)
            temporal_features[f"{prefix}angular_velocity"] = heading_change
        
        return temporal_features
    
    @staticmethod
    def _extract_statistical_features(
        df: pd.DataFrame,
        window_sizes: List[int] = [5, 10, 30]
    ) -> pd.DataFrame:
        """Extract statistical features using rolling windows.
        
        Args:
            df: DataFrame with geometric and temporal features
            window_sizes: List of window sizes (in frames) for rolling statistics
            
        Returns:
            DataFrame with statistical features
        """
        stat_features = pd.DataFrame(index=df.index)
        
        # Columns to compute rolling stats on
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if not col.startswith('frame')]
        
        for window in window_sizes:
            for col in feature_cols:
                if col in df.columns:
                    # Rolling mean
                    stat_features[f"{col}_rolling_mean_{window}"] = (
                        df[col].rolling(window=window, center=True).mean()
                    )
                    
                    # Rolling std
                    stat_features[f"{col}_rolling_std_{window}"] = (
                        df[col].rolling(window=window, center=True).std()
                    )
                    
                    # Rolling min/max
                    if window == window_sizes[-1]:  # Only for largest window
                        stat_features[f"{col}_rolling_min_{window}"] = (
                            df[col].rolling(window=window, center=True).min()
                        )
                        stat_features[f"{col}_rolling_max_{window}"] = (
                            df[col].rolling(window=window, center=True).max()
                        )
        
        return stat_features
    
    @staticmethod
    def compute_feature_separability(
        df: pd.DataFrame,
        behavior_column: str = 'behavior'
    ) -> pd.DataFrame:
        """Compute how well each feature separates different behaviors.
        
        Uses Fisher's discriminant ratio: between-class variance / within-class variance
        
        Args:
            df: DataFrame with features and behavior labels
            behavior_column: Name of column containing behavior labels
            
        Returns:
            DataFrame with feature names and separability scores
        """
        if behavior_column not in df.columns:
            return pd.DataFrame()
        
        # Get feature columns (exclude frame, behavior, etc.)
        feature_cols = [
            col for col in df.columns 
            if col not in [behavior_column, 'frame'] and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        
        separability_scores = []
        
        for col in feature_cols:
            # Skip columns with NaN or inf
            if df[col].isna().any() or np.isinf(df[col]).any():
                continue
            
            # Group by behavior
            groups = df.groupby(behavior_column)[col]
            
            # Compute between-class variance
            overall_mean = df[col].mean()
            between_var = sum(
                len(group) * (group.mean() - overall_mean)**2 
                for _, group in groups
            ) / len(df)
            
            # Compute within-class variance
            within_var = sum(
                group.var() * len(group) 
                for _, group in groups
            ) / len(df)
            
            # Fisher discriminant ratio
            if within_var > 1e-10:
                fisher_ratio = between_var / within_var
            else:
                fisher_ratio = 0.0
            
            separability_scores.append({
                'feature': col,
                'fisher_ratio': fisher_ratio,
                'between_var': between_var,
                'within_var': within_var
            })
        
        result = pd.DataFrame(separability_scores)
        if len(result) > 0:
            result = result.sort_values('fisher_ratio', ascending=False)
        
        return result
