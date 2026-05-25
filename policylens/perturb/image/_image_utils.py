"""Shared image-perturbation helpers (no SAM/IP2P — those go in recolor.py)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

from policylens.core.types import PerturbedScene, Scene


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
) -> PerturbedScene:
    new_scene = Scene(
        image=new_image,
        instruction=scene.instruction,
        metadata=scene.metadata,
        scene_id=scene.scene_id,
    )
    return PerturbedScene(
        scene=new_scene,
        perturber_name=perturber_name,
        axis=axis,
        variant_id=variant_id,
        parameters=parameters or {},
        description=description or f"{perturber_name}:{variant_id}",
    )
