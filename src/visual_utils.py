"""Shared visual utility helpers for pose viewer rendering."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

ColorInput = Union[str, Sequence[float], Sequence[int], np.ndarray]

_COLOR_NAME_MAP: dict[str, Tuple[float, float, float]] = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 0.5, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "orange": (1.0, 0.55, 0.0),
    "purple": (0.5, 0.0, 0.5),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
}


def _normalize_channel(value: Union[float, int]) -> float:
    val = float(value)
    if val > 1.0:
        val /= 255.0
    return max(0.0, min(1.0, val))


def _parse_hex_color(text: str) -> Tuple[float, float, float, Optional[float]]:
    hex_digits = text.lstrip("#").strip()
    if len(hex_digits) in {3, 4}:
        hex_digits = "".join(ch * 2 for ch in hex_digits)
    if len(hex_digits) == 6:
        r = int(hex_digits[0:2], 16)
        g = int(hex_digits[2:4], 16)
        b = int(hex_digits[4:6], 16)
        return (
            _normalize_channel(r),
            _normalize_channel(g),
            _normalize_channel(b),
            None,
        )
    if len(hex_digits) == 8:
        r = int(hex_digits[0:2], 16)
        g = int(hex_digits[2:4], 16)
        b = int(hex_digits[4:6], 16)
        a = int(hex_digits[6:8], 16)
        return (
            _normalize_channel(r),
            _normalize_channel(g),
            _normalize_channel(b),
            _normalize_channel(a),
        )
    raise ValueError(f"Unsupported hex color '{text}'")


def to_rgb(color: ColorInput) -> Tuple[float, float, float]:
    if isinstance(color, str):
        text = color.strip().lower()
        if text in {"none", "transparent"}:
            return (0.0, 0.0, 0.0)
        if text.startswith("#"):
            r, g, b, _ = _parse_hex_color(text)
            return (r, g, b)
        if text in _COLOR_NAME_MAP:
            return _COLOR_NAME_MAP[text]
        raise ValueError(f"Unrecognised color string '{color}'")

    if isinstance(color, np.ndarray):
        flat = color.flatten()
    else:
        flat = list(color)  # type: ignore[arg-type]

    if len(flat) == 4:
        flat = flat[:3]
    if len(flat) != 3:
        raise ValueError(f"Cannot convert value '{color}' to RGB")

    r, g, b = (_normalize_channel(chan) for chan in flat)
    return (float(r), float(g), float(b))


def to_rgba(color: ColorInput, *, alpha: Optional[float] = None) -> Tuple[float, float, float, float]:
    if isinstance(color, str):
        text = color.strip().lower()
        if text in {"none", "transparent"}:
            return (0.0, 0.0, 0.0, 0.0 if alpha is None else _normalize_channel(alpha))
        if text.startswith("#"):
            r, g, b, parsed_alpha = _parse_hex_color(text)
            return (
                r,
                g,
                b,
                float(
                    _normalize_channel(
                        parsed_alpha if parsed_alpha is not None else (1.0 if alpha is None else alpha)
                    )
                ),
            )
        if text in _COLOR_NAME_MAP:
            r, g, b = _COLOR_NAME_MAP[text]
            return (r, g, b, float(_normalize_channel(1.0 if alpha is None else alpha)))
        raise ValueError(f"Unrecognised color string '{color}'")

    if isinstance(color, np.ndarray):
        flat = color.flatten()
    else:
        flat = list(color)  # type: ignore[arg-type]

    if len(flat) == 4:
        r, g, b, a = flat
        return (
            float(_normalize_channel(r)),
            float(_normalize_channel(g)),
            float(_normalize_channel(b)),
            float(_normalize_channel(alpha if alpha is not None else a)),
        )
    if len(flat) == 3:
        r, g, b = flat
        return (
            float(_normalize_channel(r)),
            float(_normalize_channel(g)),
            float(_normalize_channel(b)),
            float(_normalize_channel(1.0 if alpha is None else alpha)),
        )
    raise ValueError(f"Cannot convert value '{color}' to RGBA")


def _ensure_3d(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] == 2:
        zeros = np.zeros((arr.shape[0], 1), dtype=np.float32)
        arr = np.concatenate((arr.astype(np.float32, copy=False), zeros), axis=1)
    elif arr.shape[1] == 1:
        zeros = np.zeros((arr.shape[0], 2), dtype=np.float32)
        arr = np.concatenate((arr.astype(np.float32, copy=False), zeros), axis=1)
    elif arr.shape[1] > 3:
        arr = arr[:, :3]
    return arr.astype(np.float32, copy=False)


def _rgba(color: ColorInput, alpha: Optional[float] = None) -> np.ndarray:
    return np.array(to_rgba(color, alpha=alpha), dtype=np.float32)


__all__ = [
    "ColorInput",
    "to_rgb",
    "to_rgba",
    "_ensure_3d",
    "_rgba",
]
