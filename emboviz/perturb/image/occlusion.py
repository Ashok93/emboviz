"""Patch-occlusion sweep — block out parts of the image and measure.

Multi-camera by default: when ``cameras`` is None, every camera in the
scene gets the same occlusion variant applied simultaneously. Pass
``cameras=["wrist"]`` to occlude only specific cameras. Note: when an
explicit ``bbox`` is given, the same (x0,y0,x1,y1) is applied to every
listed camera — only meaningful if those cameras share resolution and
intrinsics; pass per-camera occlusion via separate perturber instances
otherwise.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from emboviz.core.types import PerturbedScene, Scene, resolve_cameras
from emboviz.perturb.base import Perturber
from emboviz.perturb.image._image_utils import (
    make_perturbed_multi_camera_scene,
    to_array,
    to_pil,
)


class OcclusionPerturber(Perturber):
    """Mask a fraction of the image with the channel mean.

    Region: by default, a centred square sized to the requested coverage. A
    bounding box can be passed to occlude a specific region (e.g., from an
    object detector for "target occlusion").
    """

    name = "occlusion"
    axis = "vision.occlusion"
    affects = frozenset({"images.*"})

    def __init__(
        self,
        coverages: list[float] | None = None,
        bbox: Optional[tuple[int, int, int, int]] = None,
        cameras: Optional[list[str]] = None,
    ):
        self.coverages = coverages or [0.1, 0.25, 0.5, 0.75]
        self.bbox = bbox
        self.cameras = cameras

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        cameras = resolve_cameras(scene, self.cameras)
        for cov in self.coverages:
            new_images = {}
            for cam in cameras:
                arr = to_array(scene.observations.images[cam].data)
                H, W = arr.shape[:2]
                mean = arr.reshape(-1, 3).mean(axis=0)
                new = arr.copy()
                if self.bbox is not None:
                    x0, y0, x1, y1 = self.bbox
                    new[y0:y1, x0:x1] = mean
                else:
                    side = int(np.sqrt(cov) * min(H, W))
                    cy, cx = H // 2, W // 2
                    y0 = max(0, cy - side // 2)
                    x0 = max(0, cx - side // 2)
                    y1 = min(H, y0 + side)
                    x1 = min(W, x0 + side)
                    new[y0:y1, x0:x1] = mean
                new_images[cam] = to_pil(new)
            yield make_perturbed_multi_camera_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"cov{int(cov*100):03d}",
                new_images_by_camera=new_images,
                description=f"occlude {int(cov*100)}% of frame on {cameras}",
                parameters={"coverage": cov, "bbox": self.bbox, "cameras": cameras},
            )
