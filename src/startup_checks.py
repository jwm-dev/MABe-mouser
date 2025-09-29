"""Startup automation for dependency and dataset validation."""

from __future__ import annotations

import importlib
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from kaggle.api.kaggle_api_extended import KaggleApi

try:  # Python 3.10+
    import importlib.metadata as importlib_metadata
except ImportError:  # pragma: no cover - fallback for older interpreters
    import importlib_metadata  # type: ignore[import]  # noqa: F401
    import importlib_metadata as importlib_metadata  # type: ignore[assignment]


def _parse_requirement_lines(lines: Iterable[str]) -> List[str]:
    requirements: List[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r"):
            continue  # Nested requirements are not currently supported
        if " " in line:
            line = line.split(" ", 1)[0]
        requirements.append(line)
    return requirements


def _requirement_to_distribution_name(requirement: str) -> str:
    sanitized = requirement.split(";", 1)[0].strip()
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if separator in sanitized:
            sanitized = sanitized.split(separator, 1)[0].strip()
            break
    if "[" in sanitized:
        sanitized = sanitized.split("[", 1)[0].strip()
    return sanitized


def _missing_distributions(requirements: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for requirement in requirements:
        dist_name = _requirement_to_distribution_name(requirement)
        if not dist_name:
            continue
        normalized = dist_name.replace("_", "-")
        try:
            importlib_metadata.distribution(normalized)
        except importlib_metadata.PackageNotFoundError:
            missing.append(requirement)
    return missing


def _install_requirements(requirements_path: Path) -> None:
    print(f"[startup] Installing required packages defined in {requirements_path}")
    command = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
    subprocess.check_call(command)
    importlib.invalidate_caches()


def ensure_requirements_installed(requirements_path: Path) -> None:
    if not requirements_path.exists():
        print(f"[startup] Requirements file not found at {requirements_path}; skipping dependency check")
        return
    requirements = _parse_requirement_lines(requirements_path.read_text().splitlines())
    if not requirements:
        return
    missing = _missing_distributions(requirements)
    if missing:
        print(f"[startup] Missing distributions detected: {', '.join(missing)}")
        _install_requirements(requirements_path)
    else:
        print("[startup] All required Python packages already installed")


def _authenticate_kaggle() -> "KaggleApi":
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - environment-specific
        message = (
            "[startup] Kaggle authentication failed. Ensure your kaggle.json credentials are installed. "
            "Refer to the Kaggle API documentation: https://www.kaggle.com/docs/api"
        )
        print(message)
        raise RuntimeError(message) from exc
    return api


def _download_competition_archive(api: "KaggleApi", project_root: Path, zip_path: Path) -> None:
    print("[startup] Downloading MABe competition archive via Kaggle API")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    api.competition_download_files(
        "MABe-mouse-behavior-detection",
        path=str(project_root),
        quiet=False,
        force=True,
    )
    if not zip_path.exists():
        raise FileNotFoundError(
            "[startup] Expected Kaggle download to create MABe-mouse-behavior-detection.zip, but it was not found"
        )


def _extract_competition_archive(zip_path: Path, target_directory: Path) -> None:
    print(f"[startup] Extracting archive to {target_directory}")
    if target_directory.exists():
        raise FileExistsError(f"Target directory already exists: {target_directory}")
    target_directory.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_directory)
    zip_path.unlink(missing_ok=True)
    print("[startup] Dataset ready")


def ensure_competition_data(project_root: Path) -> None:
    dataset_dir = project_root / "MABe-mouse-behavior-detection"
    if dataset_dir.exists():
        print("[startup] Competition dataset already present")
        return
    api = _authenticate_kaggle()
    zip_path = project_root / "MABe-mouse-behavior-detection.zip"
    _download_competition_archive(api, project_root, zip_path)
    _extract_competition_archive(zip_path, dataset_dir)


def run_startup_checks() -> None:
    project_root = Path(__file__).resolve().parent.parent
    requirements_path = project_root / "requirements.txt"
    ensure_requirements_installed(requirements_path)
    ensure_competition_data(project_root)


__all__ = [
    "ensure_requirements_installed",
    "ensure_competition_data",
    "run_startup_checks",
]
