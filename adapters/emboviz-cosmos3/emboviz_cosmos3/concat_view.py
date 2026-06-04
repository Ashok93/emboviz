"""Split Cosmos's DROID ``concat_view`` back into the individual camera images.

Cosmos conditions on and generates a single stitched frame for DROID: the wrist
camera on top, and the two exterior cameras downscaled and placed side by side on
the bottom (cosmos-framework ``droid_lerobot_dataset.py::_load_concat_video``):

    result = cat([ wrist,                       # (h_w, w)
                   cat([left, right], width) ],  # (h_w//2, w)  -> [left | right]
                 height)                         # (h_w + h_w//2, w)

So the wrist occupies the **top two-thirds** of the height (``h_w : h_w//2`` = 2:1
for an even wrist height, the resolutions Cosmos uses), and the bottom third holds
``[exterior_left | exterior_right]`` split 50/50 across the width.

The closed-loop stress test needs this because the world model dreams the stitched
frame, but the policy under test consumes its individual cameras. This is a pure
geometric split (numpy only) — no torch, no GPU.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ConcatRegion = Literal["wrist", "exterior_left", "exterior_right"]


def build_concat_view(
    wrist: np.ndarray, exterior_left: np.ndarray, exterior_right: np.ndarray
) -> np.ndarray:
    """Stitch three camera frames into a DROID ``concat_view``.

    The inverse of :func:`split_concat_view`, reproducing
    ``_load_concat_video``: the wrist frame sets the size ``(h_w, w_w)``; the two
    exteriors are bilinearly resized to ``(h_w//2, w_w//2)`` and placed side by
    side beneath it. Returns ``(h_w + h_w//2, w_w, 3)`` uint8 RGB.
    """
    w = _as_rgb_u8(wrist, "wrist")
    left = _as_rgb_u8(exterior_left, "exterior_left")
    right = _as_rgb_u8(exterior_right, "exterior_right")

    h_w, w_w = int(w.shape[0]), int(w.shape[1])
    half_h, half_w = h_w // 2, w_w // 2
    if half_h < 1 or half_w < 1:
        raise ValueError(f"wrist frame too small to build a concat_view: {w.shape}.")

    left_r = _resize(left, half_h, half_w)
    right_r = _resize(right, half_h, half_w)
    bottom = np.concatenate([left_r, right_r], axis=1)  # (half_h, 2*half_w, 3)
    # Pad the bottom to the wrist width if 2*half_w < w_w (odd width).
    if bottom.shape[1] != w_w:
        pad = w_w - bottom.shape[1]
        bottom = np.concatenate([bottom, np.repeat(bottom[:, -1:], pad, axis=1)], axis=1)
    return np.ascontiguousarray(np.concatenate([w, bottom], axis=0))


def _as_rgb_u8(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"{name} must be (H, W, 3) uint8 RGB, got dtype={a.dtype} shape={a.shape}.")
    return a


def _resize(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(
        Image.fromarray(arr, mode="RGB").resize((width, height), Image.BILINEAR), dtype=np.uint8
    )


def split_concat_view(concat_image: np.ndarray) -> dict[ConcatRegion, np.ndarray]:
    """Split a DROID ``concat_view`` frame into its three camera images.

    ``concat_image`` is ``(H, W, 3)`` uint8 RGB. Returns ``wrist`` (top two-thirds,
    full width), ``exterior_left`` and ``exterior_right`` (bottom third, left/right
    halves). Each output is a contiguous view-derived copy in the same dtype.
    """
    arr = np.asarray(concat_image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"concat_view must be (H, W, 3) RGB, got shape {arr.shape}.")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h < 3 or w < 2:
        raise ValueError(f"concat_view too small to split: {arr.shape}.")

    split_row = round(2 * h / 3)          # wrist : exteriors height ratio is 2:1
    mid_col = w // 2
    wrist = arr[:split_row, :]
    bottom = arr[split_row:, :]
    return {
        "wrist": np.ascontiguousarray(wrist),
        "exterior_left": np.ascontiguousarray(bottom[:, :mid_col]),
        "exterior_right": np.ascontiguousarray(bottom[:, mid_col:]),
    }


__all__ = ["ConcatRegion", "split_concat_view"]
