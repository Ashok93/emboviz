"""Rendering for the closed-loop dream.

Extracts per-frame image arrays from a rollout and encodes them to MP4. Pure
Pillow + imageio (both in core); no torch, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from emboviz_wire.types import Trajectory


def _frames_of(traj_or_list) -> list:
    """Accept a Trajectory or a plain list of Scenes; return the Scene list."""
    return traj_or_list.frames if isinstance(traj_or_list, Trajectory) else list(traj_or_list)


def _scene_image(scene, camera: str) -> np.ndarray:
    return np.asarray(scene.observations.images[camera].data, dtype=np.uint8)


def frames_to_arrays(traj_or_list, camera: str) -> list[np.ndarray]:
    """Extract per-frame ``(H, W, 3)`` uint8 arrays for ``camera`` from a
    Trajectory or Scene list."""
    return [_scene_image(s, camera) for s in _frames_of(traj_or_list)]


def save_video(frames: list[np.ndarray], path: Path, *, fps: float = 10.0) -> int:
    """Write ``frames`` (each ``(H, W, 3)`` uint8 RGB) to an MP4 at ``path``.

    Returns the number of frames written. Requires uniform frame shape across the
    list (a world model that rescales mid-rollout is a real inconsistency, raised
    rather than silently letterboxed). Odd height/width are trimmed by one pixel
    so the H.264 ``yuv420p`` encoder (which needs even dimensions) accepts them.
    """
    import imageio.v3 as iio

    arrs = [np.ascontiguousarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    if not arrs:
        raise ValueError("save_video: no frames to write.")
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(
            f"save_video: frames have differing shapes {sorted(shapes)}; cannot encode one video."
        )
    h, w = arrs[0].shape[:2]
    h2, w2 = h - (h % 2), w - (w % 2)
    if (h2, w2) != (h, w):
        arrs = [a[:h2, :w2] for a in arrs]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        path, np.stack(arrs), plugin="pyav", codec="libx264",
        fps=int(round(max(1.0, fps))), out_pixel_format="yuv420p",
    )
    return len(arrs)
