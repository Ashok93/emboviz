"""Distractor-injection perturber — paste fake objects into the scene.

The lightweight version pastes colored rectangles in empty regions. A
follow-on adapter can use Stable-Diffusion-Inpaint for photorealistic
distractor insertion (kept under `perturb/image/recolor.py` if desired —
same external dep).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from policylens.core.types import PerturbedScene, Scene
from policylens.core.seeding import deterministic_seed
from policylens.perturb.base import Perturber
from policylens.perturb.image._image_utils import (
    make_perturbed_image_scene,
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
    domain = "image"

    def __init__(
        self,
        counts: list[int] | None = None,
        rect_size_frac: float = 0.08,
    ):
        self.counts = counts or [1, 3, 5]
        self.rect_size_frac = rect_size_frac

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        base = to_array(scene.image)
        H, W = base.shape[:2]
        rect_side = max(8, int(self.rect_size_frac * min(H, W)))
        seed = deterministic_seed(scene.scene_id, self.name)
        rng = np.random.default_rng(seed)

        for n in self.counts:
            pil = to_pil(base).copy()
            draw = ImageDraw.Draw(pil)
            for k in range(n):
                cx = rng.integers(rect_side, W - rect_side)
                cy = rng.integers(int(0.3 * H), H - rect_side)
                color = _DISTRACTOR_COLORS[(k * 17) % len(_DISTRACTOR_COLORS)]
                draw.rectangle(
                    [cx - rect_side // 2, cy - rect_side // 2,
                     cx + rect_side // 2, cy + rect_side // 2],
                    fill=color, outline=(0, 0, 0),
                )
            yield make_perturbed_image_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"n{n}",
                new_image=pil,
                description=f"+{n} colored distractor{'s' if n > 1 else ''}",
                parameters={"n_distractors": n},
            )
