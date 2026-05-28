"""Lighting / colour-shift perturber — gamma + HSV + brightness.

Multi-camera by default: applies the same lighting transform to every
camera in the scene per variant.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageEnhance

from emboviz.core.types import PerturbedScene, Scene, resolve_cameras
from emboviz.perturb.base import Perturber
from emboviz.perturb.image._image_utils import make_perturbed_multi_camera_scene


def _as_pil(image_data) -> Image.Image:
    if isinstance(image_data, Image.Image):
        return image_data.convert("RGB")
    return Image.fromarray(np.asarray(image_data)).convert("RGB")


class LightingShiftPerturber(Perturber):
    name = "lighting_shift"
    axis = "vision.lighting"
    affects = frozenset({"images.*"})

    def __init__(
        self,
        brightness_factors: list[float] | None = None,
        gammas: list[float] | None = None,
        saturation_factors: list[float] | None = None,
        cameras: Optional[list[str]] = None,
    ):
        self.brightness_factors = brightness_factors or [0.6, 1.4]
        self.gammas = gammas or [0.7, 1.4]
        self.saturation_factors = saturation_factors or [0.4, 1.6]
        self.cameras = cameras

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        cameras = resolve_cameras(scene, self.cameras)
        pils = {cam: _as_pil(scene.observations.images[cam].data) for cam in cameras}

        for f in self.brightness_factors:
            new_images = {cam: ImageEnhance.Brightness(pil).enhance(f) for cam, pil in pils.items()}
            yield make_perturbed_multi_camera_scene(
                scene=scene, perturber_name=self.name, axis=self.axis,
                variant_id=f"bright{int(f*100):03d}",
                new_images_by_camera=new_images,
                description=f"brightness ×{f:.2f} on {cameras}",
                parameters={"kind": "brightness", "factor": f, "cameras": cameras},
            )

        for g in self.gammas:
            new_images = {}
            for cam, pil in pils.items():
                arr = np.asarray(pil).astype(np.float32) / 255.0
                adj = np.power(arr, g) * 255.0
                new_images[cam] = Image.fromarray(np.clip(adj, 0, 255).astype(np.uint8))
            yield make_perturbed_multi_camera_scene(
                scene=scene, perturber_name=self.name, axis=self.axis,
                variant_id=f"gamma{int(g*100):03d}",
                new_images_by_camera=new_images,
                description=f"gamma {g:.2f} on {cameras}",
                parameters={"kind": "gamma", "gamma": g, "cameras": cameras},
            )

        for f in self.saturation_factors:
            new_images = {cam: ImageEnhance.Color(pil).enhance(f) for cam, pil in pils.items()}
            yield make_perturbed_multi_camera_scene(
                scene=scene, perturber_name=self.name, axis=self.axis,
                variant_id=f"sat{int(f*100):03d}",
                new_images_by_camera=new_images,
                description=f"saturation ×{f:.2f} on {cameras}",
                parameters={"kind": "saturation", "factor": f, "cameras": cameras},
            )
