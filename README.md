<div align="center">

# MABe-mouser

Interactive pose viewer and tooling for the [Mouse Action Behavior (MABe)](https://www.kaggle.com/competitions/mabe-mouse-behavior-detection) dataset.

</div>

## Table of contents

1. [Project overview](#project-overview)
2. [Key features](#key-features)
3. [Getting started](#getting-started)
4. [Usage](#usage)
5. [Repository layout](#repository-layout)
6. [Development workflow](#development-workflow)
7. [Troubleshooting](#troubleshooting)
8. [Roadmap](#roadmap)
9. [License](#license)

## Project overview

MABe-mouser is a desktop application built with PyQt6 and Vispy for exploring the MABe pose-tracking dataset. It focuses on high-quality playback and annotation of tracked mouse skeletons, with tooling for visual diagnostics such as trails, whiskers, tails, and label overlays. The application is designed to help practitioners understand dataset nuances, debug tracking artifacts, and validate competition submissions quickly.

## Key features

- **Fast pose playback** – GPU-accelerated rendering of multi-mouse sessions with smooth camera controls.
- **Rich overlays** – Visualize body markers, trails, tails, hulls, and whisker segments in a single scene.
- **Dataset-aware loading** – Convenience loaders for the official `train`, `test`, and supplemental parquet bundles supplied by Kaggle.
- **Hover and label helpers** – On-demand metadata display and intelligent label layout with minimal redraw overhead.
- **Export-friendly architecture** – Rendering utilities organized in `src/render.py` for reuse in export pipelines (WIP).

## Getting started

### 1. Prerequisites

- Python 3.11+ (the project currently targets 3.11 / 3.12)
- Git
- A Kaggle account with API credentials (`kaggle.json`)
- GPU acceleration is optional but recommended for large sessions

### 2. Clone the repository

```bash
git clone git@github.com:jwm-dev/MABe-mouser.git
cd MABe-mouser
```

### 3. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Kaggle access

1. Follow the [Kaggle API setup guide](https://github.com/Kaggle/kaggle-api#api-credentials).
2. Place `kaggle.json` in `~/.kaggle/` (Linux/macOS) or `%HOMEPATH%/.kaggle/` (Windows).
3. Ensure the file permissions are `600` (`chmod 600 ~/.kaggle/kaggle.json`).

### 6. Sync competition data

The first launch of the app will attempt to pull required bundles automatically. You can also prefetch everything via:

```bash
kaggle competitions download -c mabe-mouse-behavior-detection -p MABe-mouse-behavior-detection
unzip -o MABe-mouse-behavior-detection/mabe-mouse-behavior-detection.zip -d MABe-mouse-behavior-detection
```

Ensure the extracted `train.csv`, `test.csv`, and parquet folders remain in `MABe-mouse-behavior-detection/` inside the project root.

## Usage

1. Compile Python bytecode (optional but helpful for catching syntax issues):

	```bash
	python -m compileall src
	```

2. Launch the pose viewer:

	```bash
	python -m src
	```

3. Within the UI:
	- Use the dataset dropdown to switch between sessions.
	- Scrub timelines with the playback controls or the keyboard (←/→ for frame step, space for play/pause).
	- Hover markers to inspect per-mouse metadata.
	- Toggle overlays and rendering options through the sidebar.

The application caches downloaded pose data under `MABe-mouse-behavior-detection/` and should reuse prior downloads automatically.

## Repository layout

```
MABe-mouser/
├── README.md
├── requirements.txt
├── src/
│   ├── app.py               # PyQt6 entry point and application wiring
│   ├── cli.py               # CLI helpers for batch tasks
│   ├── camera.py            # Camera mixin handling pan/zoom and view state
│   ├── plotting.py          # Scene management, label layout, hover logic
│   ├── render.py            # Shared rendering helpers for bodies, tails, whiskers
│   ├── playback.py          # Playback orchestration and state interpolation
│   ├── geometry.py          # Vector math and spline utilities
│   └── ...                  # Additional utilities (exporters, status, UI helpers)
└── MABe-mouse-behavior-detection/
	 ├── train.csv, test.csv
	 ├── train_tracking/, test_tracking/    # Per-mouse tracking parquet bundles
	 └── train_annotation/                  # Labeled behavior segments
```

## Development workflow

1. Keep the virtual environment active.
2. Run unit and integration checks as needed (currently manual):

	```bash
	python -m compileall src            # Quick syntax validation
	python -m src                       # Launch interactive viewer for smoke tests
	```

3. When contributing:
	- Work on a feature branch (`git checkout -b feature/my-change`).
	- Use `pre-commit` or similar linting tools if you introduce them.
	- Submit PRs with screenshots/gif captures for UI-facing changes.

### Coding guidelines

- Preserve the separation between `render.py` (pure drawing helpers) and `plotting.py` (scene orchestration).
- Prefer NumPy vectorization to Python loops for geometry-heavy paths.
- Keep Vispy state mutations localized; reuse visuals instead of recreating them each frame.

## Troubleshooting

| Issue | Possible fix |
| ----- | ------------ |
| `kaggle: command not found` | Verify the virtual environment is active and `pip install kaggle` succeeded. |
| `401 - Unauthorized` while downloading | Recreate `~/.kaggle/kaggle.json` and ensure file permissions are `600`. |
| Viewer opens but shows blank scene | Confirm dataset folders are present and readable. Run from project root so relative paths resolve. |
| Heavy stutter when labels change | Ensure you are on the latest `main`; label rendering now pools text visuals to minimize redraw cost. |
| PyQt6 fails to load platform plugins | Install `qt6-base` system packages (Linux) or ensure your Python install ships with Qt binaries (Windows/macOS).

## Roadmap

- Clean remaining legacy code and finish refactoring between `plotting.py`, `playback.py`, and `render.py`.
- Streamline geometry helpers and expose a consistent API for tail/whisker rendering.
- Restore exporting pipelines (image sequence / video).
- Improve dataset selector UX with searchable dropdowns.
- Add thumbnail preview for parquet/movie assets before loading.

## License

This project is currently distributed for research and competition support. Confirm licensing terms with the repository owner before redistribution.

---

Questions or suggestions? Open an issue or reach out via the repository discussions tab.