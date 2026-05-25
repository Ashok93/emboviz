"""Viewpoint-jitter perturber — homography proxy for camera pose change.

We don't have access to the 3D scene at runtime, so we approximate viewpoint
shift by applying small perspective/affine warps. Validated by the
LIBERO-Plus team as the single largest VLA failure axis.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

from emboviz.core.types import PerturbedScene, Scene
from emboviz.perturb.base import Perturber
from emboviz.perturb.image._image_utils import (
    make_perturbed_image_scene,
    to_array,
    to_pil,
)


class ViewpointJitterPerturber(Perturber):
    """Small perspective / rotation / translation jitter."""

    name = "viewpoint_jitter"
    axis = "vision.viewpoint"
    domain = "image"

    def __init__(
        self,
        angles_deg: list[float] | None = None,
        translations_px: list[int] | None = None,
        zooms: list[float] | None = None,
    ):
        self.angles = angles_deg or [-10, -5, 5, 10]
        self.translations = translations_px or [-20, 20]
        self.zooms = zooms or [0.9, 1.1]

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        pil = scene.image if isinstance(scene.image, Image.Image) else Image.fromarray(
            to_array(scene.image)
        )
        W, H = pil.size

        for ang in self.angles:
            warped = pil.rotate(ang, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
            yield make_perturbed_image_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"rot{int(ang):+d}",
                new_image=warped,
                description=f"rotate {ang:+}°",
                parameters={"kind": "rotation", "deg": ang},
            )

        for tx in self.translations:
            warped = pil.transform(
                pil.size, Image.AFFINE, (1, 0, tx, 0, 1, 0),
                resample=Image.BILINEAR, fillcolor=(0, 0, 0),
            )
            yield make_perturbed_image_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"shiftx{tx:+d}",
                new_image=warped,
                description=f"translate x={tx:+}px",
                parameters={"kind": "translation_x", "px": tx},
            )

        for z in self.zooms:
            new_w, new_h = int(W * z), int(H * z)
            scaled = pil.resize((new_w, new_h), Image.BILINEAR)
            canvas = Image.new("RGB", (W, H), (0, 0, 0))
            ox = (W - new_w) // 2
            oy = (H - new_h) // 2
            canvas.paste(scaled, (ox, oy))
            yield make_perturbed_image_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"zoom{int(z*100)}",
                new_image=canvas,
                description=f"zoom {z:.2f}×",
                parameters={"kind": "zoom", "scale": z},
            )
