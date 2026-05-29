"""Viewpoint-jitter perturber — homography proxy for camera pose change.

We don't have access to the 3D scene at runtime, so we approximate viewpoint
shift by applying small perspective/affine warps. Validated by the
LIBERO-Plus team as the single largest VLA failure axis.

Multi-camera by default: every camera in the scene gets the same warp per
variant. For a real multi-camera setup the cameras are physically rigidly
linked, so a "same warp on every camera" model is a reasonable proxy;
isolate a single camera explicitly via ``cameras=["wrist"]`` if you only
want to perturb that one.
"""

from __future__ import annotations

from typing import Iterable, Optional

from PIL import Image

from emboviz.core.types import PerturbedScene, Scene, resolve_cameras
from emboviz.perturb.base import Perturber
from emboviz.perturb.image._image_utils import (
    make_perturbed_multi_camera_scene,
    to_array,
)


def _as_pil(image_data) -> Image.Image:
    if isinstance(image_data, Image.Image):
        return image_data
    return Image.fromarray(to_array(image_data))


class ViewpointJitterPerturber(Perturber):
    """Small perspective / rotation / translation jitter."""

    name = "viewpoint_jitter"
    axis = "vision.viewpoint"
    affects = frozenset({"images.*"})

    def __init__(
        self,
        angles_deg: list[float] | None = None,
        translations_px: list[int] | None = None,
        zooms: list[float] | None = None,
        cameras: Optional[list[str]] = None,
    ):
        self.angles = angles_deg or [-10, -5, 5, 10]
        self.translations = translations_px or [-20, 20]
        self.zooms = zooms or [0.9, 1.1]
        self.cameras = cameras

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        cameras = resolve_cameras(scene, self.cameras)
        pils = {cam: _as_pil(scene.observations.images[cam].data) for cam in cameras}

        for ang in self.angles:
            new_images = {
                cam: pil.rotate(ang, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
                for cam, pil in pils.items()
            }
            yield make_perturbed_multi_camera_scene(
                scene=scene, perturber_name=self.name, axis=self.axis,
                variant_id=f"rot{int(ang):+d}",
                new_images_by_camera=new_images,
                description=f"rotate {ang:+}° on {cameras}",
                parameters={"kind": "rotation", "deg": ang, "cameras": cameras},
            )

        for tx in self.translations:
            new_images = {}
            for cam, pil in pils.items():
                new_images[cam] = pil.transform(
                    pil.size, Image.AFFINE, (1, 0, tx, 0, 1, 0),
                    resample=Image.BILINEAR, fillcolor=(0, 0, 0),
                )
            yield make_perturbed_multi_camera_scene(
                scene=scene, perturber_name=self.name, axis=self.axis,
                variant_id=f"shiftx{tx:+d}",
                new_images_by_camera=new_images,
                description=f"translate x={tx:+}px on {cameras}",
                parameters={"kind": "translation_x", "px": tx, "cameras": cameras},
            )

        for z in self.zooms:
            new_images = {}
            for cam, pil in pils.items():
                W, H = pil.size
                new_w, new_h = int(W * z), int(H * z)
                scaled = pil.resize((new_w, new_h), Image.BILINEAR)
                canvas = Image.new("RGB", (W, H), (0, 0, 0))
                ox = (W - new_w) // 2
                oy = (H - new_h) // 2
                canvas.paste(scaled, (ox, oy))
                new_images[cam] = canvas
            yield make_perturbed_multi_camera_scene(
                scene=scene, perturber_name=self.name, axis=self.axis,
                variant_id=f"zoom{int(z*100)}",
                new_images_by_camera=new_images,
                description=f"zoom {z:.2f}× on {cameras}",
                parameters={"kind": "zoom", "scale": z, "cameras": cameras},
            )
