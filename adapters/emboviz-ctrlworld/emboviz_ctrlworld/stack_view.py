"""Build / split a Ctrl-World multi-view vertical stack.

Ctrl-World predicts its camera views jointly: each view is VAE-encoded
separately and the latents are stacked along the latent height
(``dataset/dataset_droid_exp33.py`` lines 183-186 in the vendored reference).
The pixel-space equivalent — equal-height views stitched vertically in the
profile's view order — is the single image the closed-loop driver and the wire
carry (the same one-image currency the Cosmos ``concat_view`` uses). The
adapter splits the stack and encodes each view separately — never the stacked
pixels in one VAE pass, which would bleed across view boundaries.

Which views, in which order, at which per-view size is the checkpoint's
contract and comes from its :class:`emboviz_ctrlworld.profiles.
CtrlWorldProfile` (``views`` / ``view_hw``); these functions are pure geometry
over those parameters. The reference preprocessing resizes float frames in
[-1, 1] (``extract_latent.py`` ``interpolate(..., mode='bilinear')``);
building the stack from uint8 frames applies the same bilinear resize with
uint8 rounding (±0.5/255 per channel), below the VAE's posterior sampling
noise.

This module is numpy + Pillow only — no torch.
"""

from __future__ import annotations

import numpy as np


def build_stack_view(
    images: dict[str, np.ndarray],
    *,
    views: tuple[str, ...],
    view_hw: tuple[int, int],
) -> np.ndarray:
    """Stitch per-view camera frames into a Ctrl-World stack.

    ``images`` maps view name -> ``(H, W, 3)`` uint8 RGB; every name in
    ``views`` must be present (extra keys are rejected — a silently dropped
    view would condition the dream on less than the caller intended). Each
    view is bilinearly resized to ``view_hw`` and stacked vertically in
    ``views`` order. Returns ``(len(views) * H, W, 3)`` uint8.
    """
    if not views:
        raise ValueError("build_stack_view: views must be non-empty.")
    missing = [v for v in views if v not in images]
    if missing:
        raise ValueError(
            f"build_stack_view: missing image(s) for view(s) {missing} "
            f"(have: {sorted(images)})."
        )
    extra = sorted(set(images) - set(views))
    if extra:
        raise ValueError(
            f"build_stack_view: got image(s) for unknown view(s) {extra}; "
            f"the profile's views are {list(views)}."
        )
    resized = [_resize(_as_rgb_u8(images[v], v), *view_hw) for v in views]
    return np.ascontiguousarray(np.concatenate(resized, axis=0))


def split_stack_view(
    stack_image: np.ndarray,
    *,
    views: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Split a Ctrl-World stack into its per-view camera images.

    ``stack_image`` must be ``(len(views) * H, W, 3)`` with equal-height
    thirds/halves/etc. The split is exact (no resampling).
    """
    if not views:
        raise ValueError("split_stack_view: views must be non-empty.")
    arr = np.asarray(stack_image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"stack view must be (H, W, 3) RGB, got shape {arr.shape}.")
    h = int(arr.shape[0])
    n = len(views)
    if h % n != 0:
        raise ValueError(
            f"stack view height {h} is not divisible by {n}; this profile's "
            f"stack is {n} equal-height views {list(views)}."
        )
    per = h // n
    return {
        view: np.ascontiguousarray(arr[i * per : (i + 1) * per])
        for i, view in enumerate(views)
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


__all__ = ["build_stack_view", "split_stack_view"]
