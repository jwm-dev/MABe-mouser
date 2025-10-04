"""Dataset-wide statistics cache for MABe competition analysis.

This module provides pre-computed statistics across the entire dataset
for comparative analysis and ML feature engineering.
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .models import FramePayload


class DatasetStatsCache:
    """Cache for dataset-wide statistics."""
    
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / ".dataset_stats_cache.pkl"
        self.stats: Dict[str, Any] = {}
        self._loaded = False
    
    def load_or_compute(self, dataset_root: Path, executor: Optional[ThreadPoolExecutor] = None) -> None:
        """Load cached stats or compute them."""
        # Try to load from cache first
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.stats = pickle.load(f)
                self._loaded = True
                print(f"[DatasetStats] Loaded cached statistics from {self.cache_file}")
                return
            except Exception as exc:
                print(f"[DatasetStats] Failed to load cache: {exc}")
        
        # Compute fresh statistics
        print(f"[DatasetStats] Computing dataset statistics (this may take a moment)...")
        self._compute_stats(dataset_root, executor)
        self._save_cache()
    
    def _compute_stats(self, dataset_root: Path, executor: Optional[ThreadPoolExecutor]) -> None:
        """Compute statistics across the dataset."""
        train_tracking = dataset_root / "train_tracking"
        if not train_tracking.exists():
            print(f"[DatasetStats] train_tracking not found at {train_tracking}")
            return
        
        # Find all parquet files organized by lab
        labs: Dict[str, List[Path]] = {}
        for lab_dir in train_tracking.iterdir():
            if lab_dir.is_dir() and not lab_dir.name.startswith('.'):
                parquet_files = list(lab_dir.glob("*.parquet"))
                if parquet_files:
                    labs[lab_dir.name] = parquet_files
        
        # Store basic dataset structure
        self.stats['labs'] = {
            lab: len(files) for lab, files in labs.items()
        }
        self.stats['total_files'] = sum(len(files) for files in labs.values())
        self.stats['lab_names'] = sorted(labs.keys())
        
        print(f"[DatasetStats] Found {self.stats['total_files']} files across {len(labs)} labs")
        
        # Store for comparative analysis
        self.stats['files_by_lab'] = {
            lab: [f.name for f in files] for lab, files in labs.items()
        }
        
        self._loaded = True
    
    def _save_cache(self) -> None:
        """Save statistics to cache file."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.stats, f)
            print(f"[DatasetStats] Saved statistics cache to {self.cache_file}")
        except Exception as exc:
            print(f"[DatasetStats] Failed to save cache: {exc}")
    
    def get_lab_for_file(self, file_path: Path) -> Optional[str]:
        """Get the lab name for a given file."""
        if not self._loaded:
            return None
        
        filename = file_path.name
        for lab, files in self.stats.get('files_by_lab', {}).items():
            if filename in files:
                return lab
        return None
    
    def get_files_in_lab(self, lab_name: str) -> List[str]:
        """Get all files in a specific lab."""
        if not self._loaded:
            return []
        return self.stats.get('files_by_lab', {}).get(lab_name, [])
    
    def get_lab_stats(self) -> Dict[str, int]:
        """Get file counts per lab."""
        if not self._loaded:
            return {}
        return self.stats.get('labs', {})
    
    def invalidate(self) -> None:
        """Clear the cache."""
        if self.cache_file.exists():
            self.cache_file.unlink()
        self.stats = {}
        self._loaded = False
