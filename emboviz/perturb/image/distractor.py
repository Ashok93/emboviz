"""Distractor-injection perturber — paste fake objects into the scene.

The lightweight version pastes colored rectangles in empty regions. A
follow-on adapter can use Stable-Diffusion-Inpaint for photorealistic
distractor insertion (kept under `perturb/image/recolor.py` if desired —
same external dep).

Multi-camera by default: distractors are injected into every camera per
variant. Each camera draws its OWN random rectangles (positions are
independent per camera) — that's a more honest simulation than copying
the same rectangle to every viewpoint.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw

from emboviz.core.types import PerturbedScene, Scene, resolve_cameras
from emboviz.core.seeding import deterministic_seed
from emboviz.perturb.base import Perturber
from emboviz.perturb.image._image_utils import (
    make_perturbed_multi_camera_scene,
    to_array,
    to_pil,
)


# Rough palette of "object-like" colors
_DISTRACTOR_COLORS = [
    (200, 30, 30),    # red
    (30, 30, 200),    # blue
    (30, 200, 30),    # green
    (220, 180, 30),   # yellow
    (160, 80, 200),   # purple
    (240, 130, 30),   # orange
]


class DistractorInjectionPerturber(Perturber):
    """Paste N coloured rectangles into the scene as fake distractors."""

    name = "distractor_inject"
    axis = "vision.distractor"
    affects = frozenset({"images.*"})

    def __init__(
        self,
        counts: list[int] | None = None,
        rect_size_frac: float = 0.08,
        cameras: Optional[list[str]] = None,
    ):
        self.counts = counts or [1, 3, 5]
        self.rect_size_frac = rect_size_frac
        self.cameras = cameras

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        cameras = resolve_cameras(scene, self.cameras)
        # Seed PER camera so rectangle positions are reproducible but
        # not duplicated across viewpoints.
        for n in self.counts:
            new_images = {}
            for cam in cameras:
                base = to_array(scene.observations.images[cam].data)
                H, W = base.shape[:2]
                rect_side = max(8, int(self.rect_size_frac * min(H, W)))
                seed = deterministic_seed(scene.scene_id, f"{self.name}:{cam}")
                rng = np.random.default_rng(seed)
                pil = to_pil(base).copy()
                draw = ImageDraw.Draw(pil)
                for k in range(n):
                    cx = int(rng.integers(rect_side, W - rect_side))
                    cy = int(rng.integers(int(0.3 * H), H - rect_side))
                    color = _DISTRACTOR_COLORS[(k * 17) % len(_DISTRACTOR_COLORS)]
                    draw.rectangle(
                        [cx - rect_side // 2, cy - rect_side // 2,
                         cx + rect_side // 2, cy + rect_side // 2],
                        fill=color, outline=(0, 0, 0),
                    )
                new_images[cam] = pil
            yield make_perturbed_multi_camera_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"n{n}",
                new_images_by_camera=new_images,
                description=f"+{n} colored distractor{'s' if n > 1 else ''} on {cameras}",
                parameters={"n_distractors": n, "cameras": cameras},
            )
