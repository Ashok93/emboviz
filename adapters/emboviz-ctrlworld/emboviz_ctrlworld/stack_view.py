"""Build / split the Ctrl-World three-view vertical stack.

Ctrl-World predicts the three DROID cameras jointly: each view is encoded
separately at 320x192 (W x H) and the latents are stacked along the latent
height — ``(4, 72, 40)`` = three ``(4, 24, 40)`` views — in the fixed training
order ``[exterior_1, exterior_2, wrist]`` (``dataset/dataset_droid_exp33.py``
``cond_cam_id1..3`` over the ``extract_latent.py`` video list ``[exterior_1_left,
exterior_2_left, wrist_left]``). The pixel-space equivalent is a 320x576
vertical stack of three 320x192 views, which is the single image the closed-loop
driver and the wire carry (the same one-image currency the Cosmos ``concat_view``
uses). The adapter splits the stack and encodes each view separately — never the
stacked pixels in one VAE pass, which would bleed across view boundaries.

Resolution: 192x320 per view is the checkpoint's training resolution
(``config.py`` ``height = 192, width = 320``); the stack is always built at it.
The reference preprocessing resizes float frames in [-1, 1]
(``extract_latent.py`` ``interpolate(..., mode='bilinear')``); building the
stack from uint8 frames applies the same bilinear resize with uint8 rounding
(±0.5/255 per channel), below the VAE's posterior sampling noise.

This module is a pure geometric stitch/split (numpy + Pillow) — no torch.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

StackView = Literal["exterior_1", "exterior_2", "wrist"]

#: View order along the stack height — the training latent stack order.
STACK_VIEW_ORDER: tuple[StackView, ...] = ("exterior_1", "exterior_2", "wrist")

#: Per-view size (H, W) the Ctrl-World DROID checkpoint was trained on.
VIEW_HW: tuple[int, int] = (192, 320)


def build_stack_view(
    exterior_1: np.ndarray,
    exterior_2: np.ndarray,
    wrist: np.ndarray,
) -> np.ndarray:
    """Stitch the three camera frames into a Ctrl-World stack.

    Each view is bilinearly resized to ``VIEW_HW`` and stacked vertically in
    :data:`STACK_VIEW_ORDER`. Returns ``(3 * 192, 320, 3)`` uint8 RGB.
    """
    views = [
        _resize(_as_rgb_u8(img, name), *VIEW_HW)
        for name, img in (("exterior_1", exterior_1), ("exterior_2", exterior_2), ("wrist", wrist))
    ]
    return np.ascontiguousarray(np.concatenate(views, axis=0))


def split_stack_view(stack_image: np.ndarray) -> dict[StackView, np.ndarray]:
    """Split a Ctrl-World stack into its three camera images.

    ``stack_image`` must be ``(3 * H, W, 3)`` with equal-height thirds; the
    canonical size is ``(576, 320, 3)``. The split is exact (no resampling).
    """
    arr = np.asarray(stack_image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"stack view must be (H, W, 3) RGB, got shape {arr.shape}.")
    h = int(arr.shape[0])
    if h % 3 != 0:
        raise ValueError(
            f"stack view height {h} is not divisible by 3; a Ctrl-World stack is "
            "three equal-height views."
        )
    third = h // 3
    return {
        view: np.ascontiguousarray(arr[i * third : (i + 1) * third])
        for i, view in enumerate(STACK_VIEW_ORDER)
    }


def _as_rgb_u8(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(
            f"{name} must be (H, W, 3) uint8 RGB, got dtype={a.dtype} shape={a.shape}."
        )
    return a


def _resize(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    if arr.shape[:2] == (height, width):
        return arr
    from PIL import Image

    return np.asarray(
        Image.fromarray(arr, mode="RGB").resize((width, height), Image.BILINEAR),
        dtype=np.uint8,
    )


__all__ = ["STACK_VIEW_ORDER", "StackView", "VIEW_HW", "build_stack_view", "split_stack_view"]
