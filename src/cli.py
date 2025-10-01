"""Command-line interface helpers for the pose viewer application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .app import PoseViewerApp, cudf
from .startup_checks import run_startup_checks
from .ui import create_tk_root, run_mainloop


def discover_parquet_files(root_dir: Path, max_files: Optional[int] = None) -> List[Path]:
    if root_dir.is_file() and root_dir.suffix.lower() == ".parquet":
        return [root_dir]
    files = sorted(p for p in root_dir.rglob("*.parquet") if p.is_file())
    if max_files is not None:
        files = files[:max_files]
    return files


def default_data_root() -> Path:
    candidates = [
        Path.cwd() / "MABe-mouse-behavior-detection" / "train_tracking",
        Path.cwd() / "MABe-mouse-behavior-detection" / "test_tracking",
        Path.cwd(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path.cwd()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Carousel pose viewer for MABe parquet files")
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Directory containing parquet files (default: autodetect in current workspace)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit the number of parquet files loaded into the carousel",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Attempt GPU-accelerated preprocessing with cuDF when available",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_startup_checks()
    args = parse_args(argv)
    root_dir = Path(args.root) if args.root else default_data_root()
    if not root_dir.exists():
        message = f"Cannot find parquet source: {root_dir}"
        raise SystemExit(message)

    root = create_tk_root()
    use_gpu = bool(args.gpu and cudf is not None)
    if args.gpu and not use_gpu:
        print("[pose_viewer_gui] cuDF not available; continuing on CPU", file=sys.stderr)
    if root_dir.is_file():
        if root_dir.suffix.lower() != ".parquet":
            raise SystemExit(f"Specified path is not a parquet file: {root_dir}")
        initial_files: Optional[Sequence[Path]] = [root_dir]
        discovery_root = root_dir.parent
    else:
        initial_files = None
        discovery_root = root_dir
    app = PoseViewerApp(
        root,
        initial_files,
        discovery_root=discovery_root,
        max_files=args.max_files,
        use_gpu=use_gpu,
    )
    try:
        run_mainloop(root)
    finally:
        app._force_pause_playback()


__all__ = [
    "discover_parquet_files",
    "default_data_root",
    "parse_args",
    "main",
]
