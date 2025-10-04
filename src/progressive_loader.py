"""Progressive data loading for large files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .models import FramePayload, MouseGroup


class ProgressiveDataLoader:
    """Load large parquet files progressively in chunks."""
    
    def __init__(
        self,
        chunk_size: int = 1000,  # Load 1000 frames at a time
        update_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize progressive loader.
        
        Args:
            chunk_size: Number of frames to load per chunk
            update_callback: Called when each chunk is loaded
        """
        self.chunk_size = chunk_size
        self.update_callback = update_callback
    
    def load_progressive(
        self,
        path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Load file progressively, yielding partial results.
        
        Args:
            path: Path to parquet file
            progress_callback: Progress update callback
        
        Returns:
            Complete loaded data
        """
        import pyarrow.parquet as pq
        
        # Open parquet file
        parquet_file = pq.ParquetFile(path)
        total_rows = parquet_file.metadata.num_rows
        
        # Determine chunk boundaries
        num_chunks = max(1, (total_rows + self.chunk_size - 1) // self.chunk_size)
        
        all_payloads = []
        all_metadata = {}
        
        for chunk_idx in range(num_chunks):
            start_row = chunk_idx * self.chunk_size
            end_row = min(start_row + self.chunk_size, total_rows)
            
            # Update progress
            fraction = (chunk_idx + 1) / num_chunks
            if progress_callback:
                progress_callback(
                    fraction * 0.8,  # Reserve last 20% for processing
                    f"Loading chunk {chunk_idx + 1}/{num_chunks}…"
                )
            
            # Read chunk
            table = parquet_file.read_row_group(chunk_idx) if num_chunks == parquet_file.num_row_groups else parquet_file.read()
            df_chunk = table.to_pandas()
            
            # Process chunk into payloads
            chunk_payloads = self._process_chunk(df_chunk, start_row)
            all_payloads.extend(chunk_payloads)
            
            # If we have a callback, deliver partial results
            if self.update_callback and chunk_idx < num_chunks - 1:
                partial_data = {
                    'payloads': all_payloads.copy(),
                    'is_partial': True,
                    'progress': fraction
                }
                self.update_callback(partial_data)
        
        # Final processing
        if progress_callback:
            progress_callback(0.9, "Processing complete data…")
        
        # Build final result
        result = {
            'payloads': all_payloads,
            'is_partial': False,
            'progress': 1.0,
            'total_frames': len(all_payloads)
        }
        
        # Add metadata if available
        if all_payloads:
            result.update(self._extract_metadata(all_payloads))
        
        if progress_callback:
            progress_callback(1.0, f"Loaded {len(all_payloads)} frames")
        
        return result
    
    def _process_chunk(
        self,
        df: pd.DataFrame,
        start_frame: int
    ) -> List[FramePayload]:
        """Process a dataframe chunk into FramePayload objects."""
        payloads = []
        
        # Group by frame
        if 'frame' in df.columns:
            grouped = df.groupby('frame')
        else:
            # Assume sequential frames
            df['frame'] = df.index + start_frame
            grouped = df.groupby('frame')
        
        for frame_num, frame_df in grouped:
            mouse_groups = {}
            
            # Group by mouse/individual
            if 'mouse_id' in frame_df.columns:
                mouse_col = 'mouse_id'
            elif 'individual' in frame_df.columns:
                mouse_col = 'individual'
            else:
                mouse_col = None
            
            if mouse_col:
                for mouse_id, mouse_df in frame_df.groupby(mouse_col):
                    points = self._extract_points(mouse_df)
                    if points:
                        mouse_groups[str(mouse_id)] = MouseGroup(
                            points=points,
                            colors=None
                        )
            else:
                # Single mouse or ungrouped data
                points = self._extract_points(frame_df)
                if points:
                    mouse_groups['0'] = MouseGroup(points=points, colors=None)
            
            if mouse_groups:
                payloads.append(FramePayload(
                    frame_number=int(frame_num),
                    mouse_groups=mouse_groups
                ))
        
        return payloads
    
    def _extract_points(self, df: pd.DataFrame) -> List[np.ndarray]:
        """Extract keypoint coordinates from dataframe."""
        points = []
        
        # Look for x/y columns
        x_cols = [col for col in df.columns if col.endswith('_x') or col == 'x']
        y_cols = [col for col in df.columns if col.endswith('_y') or col == 'y']
        
        if x_cols and y_cols:
            for x_col in x_cols:
                # Find corresponding y column
                if x_col == 'x':
                    y_col = 'y'
                else:
                    y_col = x_col.replace('_x', '_y')
                
                if y_col in df.columns:
                    x_val = df[x_col].iloc[0] if len(df) > 0 else np.nan
                    y_val = df[y_col].iloc[0] if len(df) > 0 else np.nan
                    
                    if not (np.isnan(x_val) or np.isnan(y_val)):
                        points.append(np.array([x_val, y_val]))
        
        return points
    
    def _extract_metadata(self, payloads: List[FramePayload]) -> Dict[str, Any]:
        """Extract metadata from payloads."""
        metadata = {}
        
        if not payloads:
            return metadata
        
        # Get mouse IDs
        all_mouse_ids = set()
        for payload in payloads:
            all_mouse_ids.update(payload.mouse_groups.keys())
        
        metadata['mouse_ids'] = sorted(all_mouse_ids)
        metadata['num_mice'] = len(all_mouse_ids)
        
        # Calculate bounding box
        all_points = []
        for payload in payloads:
            for group in payload.mouse_groups.values():
                all_points.extend([p for p in group.points if not np.isnan(p).any()])
        
        if all_points:
            all_points_array = np.array(all_points)
            min_coords = np.min(all_points_array, axis=0)
            max_coords = np.max(all_points_array, axis=0)
            
            metadata['bounds'] = {
                'min_x': float(min_coords[0]),
                'min_y': float(min_coords[1]),
                'max_x': float(max_coords[0]),
                'max_y': float(max_coords[1])
            }
            
            # Calculate display center and aspect
            center_x = (min_coords[0] + max_coords[0]) / 2
            center_y = (min_coords[1] + max_coords[1]) / 2
            metadata['display_center'] = (float(center_x), float(center_y))
            
            width = max_coords[0] - min_coords[0]
            height = max_coords[1] - min_coords[1]
            if height > 0:
                metadata['display_aspect'] = float(width / height)
        
        return metadata


__all__ = ["ProgressiveDataLoader"]
