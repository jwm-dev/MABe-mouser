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

    def _get_cache_lock(self) -> threading.RLock:
        lock = getattr(self, "_cache_lock", None)
        if lock is None:
            lock = threading.RLock()
            setattr(self, "_cache_lock", lock)
        return lock

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
        return entries

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
            stale_keys = [key for key, entry in entries.items() if isinstance(entry, dict) and entry.get("path") == resolved]
            for stale in stale_keys:
                entries.pop(stale, None)
            cache_key = self._cache_key(path)
            entries[cache_key] = {
                "path": resolved,
                "signature": self._cache_signature(path),
                "payload": data,
            }
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
