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

    Single-camera helper. For multi-camera perturbers use
    ``make_perturbed_multi_camera_scene``. Raises KeyError if ``camera``
    is not already in the scene — we never invent cameras silently.
    """
    new_scene = scene.with_image(new_image, camera=camera)
    return PerturbedScene(
        scene=new_scene,
        perturber_name=perturber_name,
        axis=axis,
        variant_id=variant_id,
        parameters=parameters or {},
        description=description or f"{perturber_name}:{variant_id}",
    )


def make_perturbed_multi_camera_scene(
    scene: Scene,
    perturber_name: str,
    axis: str,
    variant_id: str,
    new_images_by_camera: dict,
    description: str = "",
    parameters: Optional[dict] = None,
) -> PerturbedScene:
    """Build a PerturbedScene with multiple cameras replaced in one step.

    Every key in ``new_images_by_camera`` must already exist in the scene's
    ``observations.images`` — this never invents cameras. Use when a
    perturbation is applied identically across all (or several) cameras,
    e.g., uniform noise across every camera stream.
    """
    new_scene = scene.with_images(new_images_by_camera)
    affected_cameras = sorted(new_images_by_camera)
    enriched_params = dict(parameters or {})
    enriched_params.setdefault("cameras", affected_cameras)
    return PerturbedScene(
        scene=new_scene,
        perturber_name=perturber_name,
        axis=axis,
        variant_id=variant_id,
        parameters=enriched_params,
        description=description or f"{perturber_name}:{variant_id} on {affected_cameras}",
    )
