# MABe Pose Carousel Viewer

## Font Awesome setup for the custom title bar

The in-app title bar expects access to the Font Awesome 6 Free **Solid** font so it can render the cat glyph (`fa-solid fa-cat`). To enable it:

1. Download the Font Awesome Free kit from [fontawesome.com](https://fontawesome.com/download).
2. Copy the `Font Awesome 6 Free-Solid-900.otf` (or the `.ttf` variant) into `src/fonts/` so it sits alongside the Python modules. You can also point to any copy of the Solid font via the `POSE_VIEWER_FA_FONT` environment variable.

Once the font file is available, launch the viewer as usual. If the font cannot be loaded, the UI will fall back to a Unicode cat emoji so you still have a recognizable icon.

```bash
# Example using the environment variable
export POSE_VIEWER_FA_FONT=/path/to/Font\ Awesome\ 6\ Free-Solid-900.otf
python -m src
```
