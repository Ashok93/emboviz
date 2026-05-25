"""Patch-occlusion sweep — block out parts of the image and measure.

Two flavours:
  • **Sweep mode** — N variants at increasing coverage levels (10/25/50/75 %),
    each masking the same fixed region or a centered region.
  • **Single-patch mode** — used by `SensitivityMap` to ablate one patch at a
    time over a grid.
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


class OcclusionPerturber(Perturber):
    """Mask a fraction of the image with the channel mean.

    Region: by default, a centred square sized to the requested coverage. A
    bounding box can be passed to occlude a specific region (e.g., from an
    object detector for "target occlusion").
    """

    name = "occlusion"
    axis = "vision.occlusion"
    affects = frozenset({"images.primary"})

    def __init__(
        self,
        coverages: list[float] | None = None,
        bbox: Optional[tuple[int, int, int, int]] = None,
    ):
        self.coverages = coverages or [0.1, 0.25, 0.5, 0.75]
        self.bbox = bbox

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        arr = to_array(scene.primary_image_data)
        H, W = arr.shape[:2]
        mean = arr.reshape(-1, 3).mean(axis=0)
        for cov in self.coverages:
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
            yield make_perturbed_image_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"cov{int(cov*100):03d}",
                new_image=to_pil(new),
                description=f"occlude {int(cov*100)}% of frame",
                parameters={"coverage": cov, "bbox": self.bbox},
            )
