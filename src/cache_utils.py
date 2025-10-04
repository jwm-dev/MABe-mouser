"""Cache helper mixin for the pose viewer application."""

from __future__ import annotations

import gzip
import hashlib
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .constants import CACHE_VERSION


class PoseViewerCacheMixin:
    _cache_filename = "pose_cache.pkl.gz"
    _max_cached_entries = 50  # Increased from 5 to 50 for better hit rate
    _max_entry_size_mb = 100  # Don't cache files > 100MB

    def _get_cache_lock(self) -> threading.RLock:
        lock = getattr(self, "_cache_lock", None)
        if lock is None:
            lock = threading.RLock()
            setattr(self, "_cache_lock", lock)
        return lock

    def _get_lru_tracker(self) -> Dict[str, int]:
        """Get or create LRU access tracker."""
        tracker = getattr(self, "_lru_access_tracker", None)
        if tracker is None:
            tracker = {}
            setattr(self, "_lru_access_tracker", tracker)
        return tracker

    def _update_lru_access(self, cache_key: str) -> None:
        """Update LRU access time for a cache key."""
        import time
        tracker = self._get_lru_tracker()
        tracker[cache_key] = time.time()

    def _cache_signature(self, path: Path) -> str:
        try:
            stat = path.stat()
            return f"{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}::{CACHE_VERSION}"
        except FileNotFoundError:
            return f"{path.resolve()}::missing::{CACHE_VERSION}"

    def _cache_key(self, path: Path) -> str:
        return hashlib.sha1(self._cache_signature(path).encode("utf-8")).hexdigest()

    def _cache_file(self) -> Path:
        return self.cache_dir / self._cache_filename

    def _ensure_cache_entries(self) -> Dict[str, Dict[str, Any]]:
        entries = getattr(self, "_cache_entries", None)
        if entries is None:
            self._migrate_legacy_cache_files()
            entries = self._cache_entries = self._load_cache_entries()
            if self._prune_cache_entries(entries):
                self._flush_cache_entries(entries)
        return entries

    def _flush_cache_on_startup(self) -> None:
        if not getattr(self, "cache_enabled", True):
            return
        with self._get_cache_lock():
            entries = self._ensure_cache_entries()
            self._flush_cache_entries(entries)

    def _prune_cache_entries(self, entries: Dict[str, Dict[str, Any]]) -> bool:
        """Prune cache entries using LRU eviction."""
        limit = int(getattr(self, "_max_cached_entries", 50))
        if limit <= 0:
            entries.clear()
            return True
        modified = False
        
        # Remove entries exceeding size limit
        max_size_bytes = self._max_entry_size_mb * 1024 * 1024
        for key in list(entries.keys()):
            entry = entries.get(key)
            if isinstance(entry, dict):
                payload = entry.get("payload", {})
                # Estimate size of payload (rough approximation)
                try:
                    import sys
                    size = sys.getsizeof(payload)
                    # Deep size estimation for nested structures
                    if isinstance(payload, dict):
                        for v in payload.values():
                            size += sys.getsizeof(v)
                    if size > max_size_bytes:
                        entries.pop(key, None)
                        modified = True
                except Exception:
                    pass
        
        # LRU eviction if still over limit
        while len(entries) > limit:
            tracker = self._get_lru_tracker()
            # Find least recently used entry
            lru_key = None
            lru_time = float('inf')
            
            for key in entries.keys():
                access_time = tracker.get(key, 0)
                if access_time < lru_time:
                    lru_time = access_time
                    lru_key = key
            
            # Fallback to first entry if no LRU data
            if lru_key is None:
                try:
                    lru_key = next(iter(entries))
                except StopIteration:
                    break
            
            entries.pop(lru_key, None)
            tracker.pop(lru_key, None)
            modified = True
        return modified

    def _load_cache_entries(self) -> Dict[str, Dict[str, Any]]:
        cache_file = self._cache_file()
        if not cache_file.exists():
            return {}
        try:
            with gzip.open(cache_file, "rb") as fh:
                bundle = pickle.load(fh)
            if not isinstance(bundle, dict) or bundle.get("version") != CACHE_VERSION:
                raise ValueError("cache version mismatch")
            entries = bundle.get("entries", {})
            if not isinstance(entries, dict):
                raise ValueError("cache entries missing")
            return entries
        except Exception:
            self._safe_unlink(cache_file)
            return {}

    def _legacy_cache_directories(self) -> Iterable[Path]:
        base_dirs = [self.cache_dir]
        try:
            base_dirs.append(Path.home() / ".mabe_pose_cache")
        except Exception:
            pass
        try:
            module_dir = Path(__file__).resolve().parent
            base_dirs.append(module_dir / "__pycache__")
        except Exception:
            pass
        return base_dirs

    def _migrate_legacy_cache_files(self) -> None:
        if getattr(self, "_legacy_cache_migrated", False):
            return
        setattr(self, "_legacy_cache_migrated", True)
        for directory in self._legacy_cache_directories():
            if not directory.exists():
                continue
            try:
                for candidate in directory.iterdir():
                    if candidate.is_dir():
                        continue
                    if candidate.suffixes[-2:] == [".pkl", ".gz"] and candidate.name != self._cache_filename:
                        self._safe_unlink(candidate)
            except Exception:
                continue

    def _flush_cache_entries(self, entries: Dict[str, Dict[str, Any]]) -> None:
        cache_file = self._cache_file()
        temp_file = cache_file.with_name(f".{cache_file.name}.tmp")
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(temp_file, "wb") as fh:
                pickle.dump({"version": CACHE_VERSION, "entries": entries}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            temp_file.replace(cache_file)
        except Exception:
            self._safe_unlink(temp_file)
            self._safe_unlink(cache_file)
            self._cache_entries = {}

    def _load_cached_data(self, path: Path) -> Optional[Dict[str, object]]:
        if not self.cache_enabled:
            return None
        with self._get_cache_lock():
            entries = self._ensure_cache_entries()
            cache_key = self._cache_key(path)
            entry = entries.get(cache_key)
            if not isinstance(entry, dict):
                return None
            signature = entry.get("signature")
            if signature != self._cache_signature(path):
                entries.pop(cache_key, None)
                self._flush_cache_entries(entries)
                return None
            payload = entry.get("payload")
            if isinstance(payload, dict):
                # Update LRU access time on cache hit
                self._update_lru_access(cache_key)
                return payload
            entries.pop(cache_key, None)
            self._flush_cache_entries(entries)
            return None

    def _store_cached_data(self, path: Path, data: Dict[str, object]) -> None:
        if not self.cache_enabled:
            return
        with self._get_cache_lock():
            entries = self._ensure_cache_entries()
            resolved = str(path.resolve())
            signature = self._cache_signature(path)
            stale_keys = [key for key, entry in entries.items() if isinstance(entry, dict) and entry.get("path") == resolved]
            state_changed = False
            for stale in stale_keys:
                if entries.pop(stale, None) is not None:
                    state_changed = True
            cache_key = self._cache_key(path)
            existing = entries.get(cache_key)
            if (
                isinstance(existing, dict)
                and existing.get("signature") == signature
                and existing.get("path") == resolved
            ):
                if self._prune_cache_entries(entries):
                    state_changed = True
                if state_changed:
                    self._flush_cache_entries(entries)
                return
            entries[cache_key] = {
                "path": resolved,
                "signature": signature,
                "payload": data,
            }
            # Update LRU access time on cache write
            self._update_lru_access(cache_key)
            state_changed = True
            if self._prune_cache_entries(entries):
                state_changed = True
            if state_changed:
                self._flush_cache_entries(entries)

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except Exception:
            pass


__all__ = ["PoseViewerCacheMixin"]
