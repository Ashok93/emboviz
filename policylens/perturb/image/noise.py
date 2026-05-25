"""Gaussian noise / sensor-noise simulation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from policylens.core.types import PerturbedScene, Scene
from policylens.core.seeding import deterministic_seed
from policylens.perturb.base import Perturber
from policylens.perturb.image._image_utils import (
    make_perturbed_image_scene,
    to_array,
    to_pil,
)


class GaussianNoisePerturber(Perturber):
    name = "gaussian_noise"
    axis = "vision.sensor_noise"
    domain = "image"

    def __init__(self, stddevs: list[float] | None = None):
        self.stddevs = stddevs or [5.0, 15.0, 30.0]   # in pixel-value units (0-255)

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        arr = to_array(scene.image).astype(np.float32)
        seed = deterministic_seed(scene.scene_id, self.name)
        rng = np.random.default_rng(seed)
        for s in self.stddevs:
            noisy = arr + rng.normal(0, s, arr.shape)
            yield make_perturbed_image_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"std{int(s):03d}",
                new_image=to_pil(noisy),
                description=f"gaussian noise σ={s:g}",
                parameters={"sigma": s},
            )
