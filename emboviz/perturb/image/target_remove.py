"""Target-removal perturber — the memorization sniff test.

If we mask out the target object and the model STILL executes a coherent
trajectory toward where the object usually is, it's memorising. From
LIBERO-Pro this is one of the cleanest tests of "is this really vision-
conditioned policy?"

If a bounding box of the target isn't supplied (cheap mode), we approximate
by masking a centred patch sized to the typical target object — good enough
to detect memorization on most cases.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from emboviz.core.types import PerturbedScene, Scene
from emboviz.perturb.base import Perturber
from emboviz.perturb.image._image_utils import (
    make_perturbed_image_scene,
    to_array,
    to_pil,
)


class TargetRemovalPerturber(Perturber):
    """Mask out the target object (or its likely location)."""

    name = "target_remove"
    axis = "vision.memorization"
    domain = "image"

    def __init__(
        self,
        bbox: Optional[tuple[int, int, int, int]] = None,
        patch_size_frac: float = 0.25,
        fill: str = "channel_mean",  # or "black" / "white"
    ):
        self.bbox = bbox
        self.patch_size_frac = patch_size_frac
        self.fill = fill

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        arr = to_array(scene.image).copy()
        H, W = arr.shape[:2]
        if self.fill == "channel_mean":
            fill = arr.reshape(-1, 3).mean(axis=0)
        elif self.fill == "black":
            fill = np.zeros(3, dtype=arr.dtype)
        else:
            fill = np.full(3, 255, dtype=arr.dtype)
        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox
            arr[y0:y1, x0:x1] = fill
        else:
            side = max(8, int(self.patch_size_frac * min(H, W)))
            # Bridge target objects sit toward the bottom-centre of the frame
            cy, cx = int(0.65 * H), W // 2
            y0 = max(0, cy - side // 2)
            x0 = max(0, cx - side // 2)
            arr[y0:y0 + side, x0:x0 + side] = fill
        yield make_perturbed_image_scene(
            scene=scene,
            perturber_name=self.name,
            axis=self.axis,
            variant_id="target_masked",
            new_image=to_pil(arr),
            description="target region masked out (memorization probe)",
            parameters={
                "bbox": self.bbox,
                "patch_size_frac": self.patch_size_frac,
                "fill": self.fill,
            },
        )
