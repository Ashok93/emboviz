"""Target-removal perturber — the memorization sniff test.

If we mask out the target object and the model STILL executes a coherent
trajectory toward where the object usually is, it's memorizing. From
LIBERO-Pro this is one of the cleanest tests of "is this really a vision-
conditioned policy?"

Multi-camera by default: the target is located independently in every
camera (each camera sees the scene from a different viewpoint, so the
target's bbox/mask differs per camera). If a detector cannot find the
target with confidence in a given camera, the perturber yields NO variant
for that camera rather than masking a fabricated region.

You must supply either:
  • an explicit ``bbox`` (applied to every requested camera; only sensible
    when those cameras share resolution and intrinsics), OR
  • a ``target_detector`` matching the TargetDetector protocol (e.g. the
    default ``GroundingDINOSAMDetector``).

We never fall back to "centred patch" — that masks the table or the
gripper and silently invalidates the diagnostic.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from emboviz.core.types import PerturbedScene, Scene, resolve_cameras
from emboviz.perturb._target_detection import (
    BBoxDetector,
    TargetDetector,
)
from emboviz.perturb.base import Perturber
from emboviz.perturb.image._image_utils import (
    make_perturbed_multi_camera_scene,
    to_array,
    to_pil,
)


class TargetRemovalPerturber(Perturber):
    """Mask out the target object across one or more cameras."""

    name = "target_remove"
    axis = "vision.memorization"
    affects = frozenset({"images.*"})

    def __init__(
        self,
        bbox: Optional[tuple[int, int, int, int]] = None,
        target_detector: Optional[TargetDetector] = None,
        fill: str = "channel_mean",       # or "black" / "white"
        cameras: Optional[list[str]] = None,
    ):
        if bbox is not None:
            self.detector: TargetDetector = BBoxDetector(bbox)
        elif target_detector is not None:
            self.detector = target_detector
        else:
            raise ValueError(
                "TargetRemovalPerturber needs one of ``bbox=(x0,y0,x1,y1)`` "
                "or ``target_detector=...`` (SAM3Detector / "
                "GroundingDINOSAMDetector / JSONAnnotationConnector / "
                "CocoAnnotationConnector / CallableConnector). The "
                "perturber refuses to invent a target — silent fallback "
                "to a centred patch would mask the table or the gripper "
                "and invalidate any downstream verdict."
            )
        self.fill = fill
        self.cameras = cameras

    def _fill_value(self, arr: np.ndarray) -> np.ndarray:
        if self.fill == "channel_mean":
            return arr.reshape(-1, 3).mean(axis=0).astype(arr.dtype)
        if self.fill == "black":
            return np.zeros(3, dtype=arr.dtype)
        if self.fill == "white":
            return np.full(3, 255, dtype=arr.dtype)
        raise ValueError(f"Unknown fill mode: {self.fill!r}")

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        cameras = resolve_cameras(scene, self.cameras)
        new_images: dict = {}
        per_cam_meta: dict = {}
        for cam in cameras:
            cam_scene = scene.with_image(scene.observations.images[cam].data, camera=cam)
            # Re-detect per camera by running detector on a scene whose "primary"
            # accessor points at THIS camera's image.
            tmp = scene.with_image(scene.observations.images[cam].data, camera="primary") \
                  if "primary" in scene.observations.images and cam != "primary" else cam_scene
            detection = self.detector(tmp)
            if detection is None:
                # Honest skip for this camera — caller will see fewer cameras
                # in the variant's `cameras` parameter than they requested.
                per_cam_meta[cam] = {"detection": None}
                continue
            arr = to_array(scene.observations.images[cam].data).copy()
            fill = self._fill_value(arr)
            if detection.mask is not None and detection.mask.shape == arr.shape[:2]:
                arr[detection.mask] = fill
                per_cam_meta[cam] = {
                    "label": detection.label, "confidence": detection.confidence,
                    "mask_used": True,
                }
            else:
                x0, y0, x1, y1 = detection.bbox
                x0 = max(0, x0); y0 = max(0, y0)
                x1 = min(arr.shape[1], x1); y1 = min(arr.shape[0], y1)
                arr[y0:y1, x0:x1] = fill
                per_cam_meta[cam] = {
                    "label": detection.label, "confidence": detection.confidence,
                    "mask_used": False, "bbox": (x0, y0, x1, y1),
                }
            new_images[cam] = to_pil(arr)
        if not new_images:
            # No camera had a confident detection; emit nothing rather than fake.
            return
        yield make_perturbed_multi_camera_scene(
            scene=scene,
            perturber_name=self.name,
            axis=self.axis,
            variant_id="target_masked",
            new_images_by_camera=new_images,
            description=(
                f"target region masked across {sorted(new_images)} "
                f"(memorization probe)"
            ),
            parameters={
                "fill": self.fill,
                "requested_cameras": cameras,
                "masked_cameras": sorted(new_images),
                "per_camera_detection": per_cam_meta,
            },
        )
