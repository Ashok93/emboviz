"""Shared image-perturbation helpers (no SAM/IP2P — those go in recolor.py)."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
from PIL import Image

from emboviz.core.observations import RGBImage
from emboviz.core.types import Observations, PerturbedScene, Scene


def to_array(image) -> np.ndarray:
    """Coerce a PIL image to uint8 HWC ndarray."""
    if isinstance(image, np.ndarray):
        return image
    return np.array(image.convert("RGB"))


def to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_perturbed_image_scene(
    scene: Scene,
    perturber_name: str,
    axis: str,
    variant_id: str,
    new_image,
    description: str = "",
    parameters: Optional[dict] = None,
    camera: str = "primary",
) -> PerturbedScene:
    """Build a PerturbedScene with one camera's image replaced.

    Multi-cam aware: only the named `camera` is replaced; other cameras
    in observations.images are preserved. Default is "primary".
    """
    new_images = dict(scene.observations.images)
    new_images[camera] = RGBImage(data=new_image, camera_id=camera)
    new_obs = replace(scene.observations, images=new_images)
    new_scene = replace(scene, observations=new_obs)
    return PerturbedScene(
        scene=new_scene,
        perturber_name=perturber_name,
        axis=axis,
        variant_id=variant_id,
        parameters=parameters or {},
        description=description or f"{perturber_name}:{variant_id}",
    )
