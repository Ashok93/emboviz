"""Lighting / colour-shift perturber — gamma + HSV + brightness."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance

from policylens.core.types import PerturbedScene, Scene
from policylens.perturb.base import Perturber
from policylens.perturb.image._image_utils import make_perturbed_image_scene


class LightingShiftPerturber(Perturber):
    name = "lighting_shift"
    axis = "vision.lighting"
    domain = "image"

    def __init__(
        self,
        brightness_factors: list[float] | None = None,
        gammas: list[float] | None = None,
        saturation_factors: list[float] | None = None,
    ):
        self.brightness_factors = brightness_factors or [0.6, 1.4]
        self.gammas = gammas or [0.7, 1.4]
        self.saturation_factors = saturation_factors or [0.4, 1.6]

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        pil = scene.image if isinstance(scene.image, Image.Image) else Image.fromarray(
            np.asarray(scene.image)
        )
        pil = pil.convert("RGB")

        for f in self.brightness_factors:
            yield make_perturbed_image_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"bright{int(f*100):03d}",
                new_image=ImageEnhance.Brightness(pil).enhance(f),
                description=f"brightness ×{f:.2f}",
                parameters={"kind": "brightness", "factor": f},
            )

        for g in self.gammas:
            arr = np.asarray(pil).astype(np.float32) / 255.0
            adj = np.power(arr, g) * 255.0
            yield make_perturbed_image_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"gamma{int(g*100):03d}",
                new_image=Image.fromarray(np.clip(adj, 0, 255).astype(np.uint8)),
                description=f"gamma {g:.2f}",
                parameters={"kind": "gamma", "gamma": g},
            )

        for f in self.saturation_factors:
            yield make_perturbed_image_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"sat{int(f*100):03d}",
                new_image=ImageEnhance.Color(pil).enhance(f),
                description=f"saturation ×{f:.2f}",
                parameters={"kind": "saturation", "factor": f},
            )
