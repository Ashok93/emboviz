"""Gaussian noise / sensor-noise simulation.

Multi-camera by default: when ``cameras`` is None, every camera in the
scene gets the same noise variant applied simultaneously. To target
specific cameras, pass ``cameras=["primary"]`` etc.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from emboviz.core.types import PerturbedScene, Scene, resolve_cameras
from emboviz.core.seeding import deterministic_seed
from emboviz.perturb.base import Perturber
from emboviz.perturb.image._image_utils import (
    make_perturbed_multi_camera_scene,
    to_array,
    to_pil,
)


class GaussianNoisePerturber(Perturber):
    name = "gaussian_noise"
    axis = "vision.sensor_noise"
    affects = frozenset({"images.*"})

    def __init__(
        self,
        stddevs: list[float] | None = None,
        cameras: Optional[list[str]] = None,
    ):
        self.stddevs = stddevs or [5.0, 15.0, 30.0]   # pixel-value units (0-255)
        self.cameras = cameras

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        cameras = resolve_cameras(scene, self.cameras)
        seed = deterministic_seed(scene.scene_id, self.name)
        rng = np.random.default_rng(seed)
        for s in self.stddevs:
            new_images = {}
            for cam in cameras:
                arr = to_array(scene.observations.images[cam].data).astype(np.float32)
                noisy = arr + rng.normal(0, s, arr.shape)
                new_images[cam] = to_pil(noisy)
            yield make_perturbed_multi_camera_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"std{int(s):03d}",
                new_images_by_camera=new_images,
                description=f"gaussian noise σ={s:g} on {cameras}",
                parameters={"sigma": s, "cameras": cameras},
            )
