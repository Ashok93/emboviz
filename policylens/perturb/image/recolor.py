"""Object recolor perturber — the color-binding test.

Programmatically recolors a target object in the scene (text-prompted via
GroundingDINO+SAM2) and tests whether the VLA's action changes. If the
model is invariant to target color, it isn't using the color attribute
to disambiguate — exactly the failure documented in 'When Vision
Overrides Language' (arXiv 2602.17659) and the Confusion Benchmark.

Why HSV-rotation, not InstructPix2Pix?
  • Deterministic, fast (~1 s per variant), preserves scene structure
    perfectly. For a binding test we want precision, not photorealism.
  • IP2P is stochastic and adds artifacts that can confound the diagnostic.

Why SAM2 on a GroundingDINO bbox, not raw bbox?
  • SAM2 produces a pixel-accurate mask. HSV rotation on the bbox would
    recolor irrelevant background pixels. SAM mask isolates the object.

Heavy deps (transformers' GroundingDINO + SAM2) are lazy-imported and
module-cached: models load once and are reused across all perturber
instances + scenes. Falls back gracefully if libs aren't installed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image

from policylens.core.types import PerturbedScene, Scene
from policylens.perturb.base import Perturber
from policylens.perturb.image._image_utils import (
    make_perturbed_image_scene,
    to_array,
    to_pil,
)


# Color palette → target hue in HSV (0-360 deg).
COLOR_HUE: dict[str, int] = {
    "red": 0,
    "orange": 30,
    "yellow": 60,
    "green": 120,
    "cyan": 180,
    "blue": 240,
    "purple": 280,
    "magenta": 320,
}

DEFAULT_COLORS = ["red", "blue", "green", "yellow", "purple"]

# Detection-quality thresholds — calibrated against small-object detection
# (Bridge spoons are ~30 px wide; GroundingDINO scores cluster lower than for
# COCO-scale objects, so we keep the bar permissive).
_BOX_THRESHOLD = 0.25
_TEXT_THRESHOLD = 0.20
_MIN_MASK_PIXELS = 16  # below this we treat the detection as failed


# -----------------------------------------------------------------------------
# Module-level model caches — GroundingDINO and SAM load independently so one
# failure doesn't kill the other. Without SAM we still have rect-bbox masks.
# -----------------------------------------------------------------------------


_GROUNDING: Optional[tuple] = None    # (processor, model)
_SAM: Optional[tuple] = None          # (processor, model)
_LOCK = threading.Lock()
_GROUNDING_REPO = "IDEA-Research/grounding-dino-tiny"
# SAM v1 base model — broadly compatible with transformers' SamProcessor.
# SAM2 (facebook/sam2-hiera-tiny) requires `Sam2Processor` which is only in
# transformers ≥ 4.50; we deliberately stay one version back for stability.
_SAM_REPO = "facebook/sam-vit-base"


def _load_grounding(device: str = "cuda"):
    """Load GroundingDINO once. Raises if transformers isn't installed."""
    global _GROUNDING
    with _LOCK:
        if _GROUNDING is not None:
            return _GROUNDING
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )
        proc = AutoProcessor.from_pretrained(_GROUNDING_REPO)
        mod = AutoModelForZeroShotObjectDetection.from_pretrained(_GROUNDING_REPO).to(device)
        mod.eval()
        _GROUNDING = (proc, mod)
        return _GROUNDING


def _load_sam(device: str = "cuda"):
    """Load SAM (v1) once. Returns None if SAM isn't installable on this env."""
    global _SAM
    with _LOCK:
        if _SAM is not None:
            return _SAM
        try:
            from transformers import SamModel, SamProcessor
            proc = SamProcessor.from_pretrained(_SAM_REPO)
            mod = SamModel.from_pretrained(_SAM_REPO).to(device)
            mod.eval()
            _SAM = (proc, mod)
            return _SAM
        except Exception as e:
            # SAM is OPTIONAL — we degrade to rectangular masks if it can't
            # load. Caching the failure (None tuple) avoids re-trying.
            _SAM = (None, None)
            return _SAM


# -----------------------------------------------------------------------------
# Detection + masking primitives
# -----------------------------------------------------------------------------


def _detect_bbox(image: Image.Image, text: str, device: str = "cuda") -> Optional[tuple]:
    """Text-prompted detection. Returns (x0, y0, x1, y1) in pixel coords.

    Handles a few transformers-version quirks:
      • param name changed `box_threshold` → `threshold` (depr ≥4.51)
      • output dict may have `text_labels` instead of `labels`
      • `scores` may not exist if 0 boxes pass thresholds
    """
    import inspect
    import torch

    proc, model = _load_grounding(device)
    inputs = proc(
        images=image,
        text=f"{text}.",
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]]).to(device)

    post_proc = (
        getattr(proc, "post_process_grounded_object_detection", None)
        or getattr(proc, "post_process_object_detection", None)
    )
    if post_proc is None:
        return None

    # Pick the right threshold kwarg for the installed version.
    sig = inspect.signature(post_proc).parameters
    kwargs: dict = {"target_sizes": target_sizes}
    if "input_ids" in sig:
        kwargs["input_ids"] = inputs["input_ids"]
    if "threshold" in sig:
        kwargs["threshold"] = _BOX_THRESHOLD
    elif "box_threshold" in sig:
        kwargs["box_threshold"] = _BOX_THRESHOLD
    if "text_threshold" in sig:
        kwargs["text_threshold"] = _TEXT_THRESHOLD

    results = post_proc(outputs, **kwargs)[0]
    boxes = results.get("boxes")
    scores = results.get("scores")
    if boxes is None or len(boxes) == 0 or scores is None or len(scores) == 0:
        return None
    best_idx = int(torch.as_tensor(scores).argmax())
    box = torch.as_tensor(boxes[best_idx]).detach().cpu().numpy()
    return tuple(int(round(float(v))) for v in box)


def _bbox_to_mask(image: Image.Image, bbox: tuple, device: str = "cuda") -> Optional[np.ndarray]:
    """SAM-refine a bbox into a pixel-accurate boolean mask.

    Returns None if SAM isn't available; callers fall back to a rect mask.
    """
    import torch
    proc, model = _load_sam(device)
    if proc is None or model is None:
        return None
    inputs = proc(
        images=image,
        input_boxes=[[list(bbox)]],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    masks = proc.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    return masks[0, 0].numpy().astype(bool)


def _rectangular_mask(image_shape: tuple, bbox: tuple) -> np.ndarray:
    """Fallback mask: a filled rectangle inside the bbox."""
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    x0, y0, x1, y1 = bbox
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    mask[y0:y1, x0:x1] = True
    return mask


def _recolor_with_mask(
    image: Image.Image, mask: np.ndarray, target_hue_deg: int,
    saturation_floor: int = 160,
) -> Image.Image:
    """HSV-rotate masked pixels to `target_hue_deg` (0-360). Bumps saturation
    inside the mask so the recolor is visually distinct."""
    pil_rgb = image.convert("RGB")
    hsv = np.array(pil_rgb.convert("HSV"))
    # PIL stores H in [0, 255] (mapped from [0, 360)).
    hue_255 = int((target_hue_deg % 360) / 360 * 255)
    hsv[..., 0] = np.where(mask, hue_255, hsv[..., 0])
    hsv[..., 1] = np.where(mask, np.maximum(hsv[..., 1], saturation_floor), hsv[..., 1])
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


# -----------------------------------------------------------------------------
# Public perturber
# -----------------------------------------------------------------------------


class ObjectRecolorPerturber(Perturber):
    """Recolor a target object to N colors; emit one variant per color.

    Two ways to specify the target:
      • `target` (text query)   — uses GroundingDINO to detect, SAM2 to mask
      • `target_bbox` (x0,y0,x1,y1) — skips GroundingDINO, uses SAM2 to refine

    If SAM2 isn't available, falls back to a rectangular-bbox mask.
    Variants are produced lazily; the mask is cached per `scene.scene_id`
    so we only run detection once per scene even when N colors are tested.
    """

    name = "object_recolor"
    axis = "vision.color_binding"
    domain = "image"

    def __init__(
        self,
        target: Optional[str] = None,
        target_bbox: Optional[tuple] = None,
        colors: Optional[list[str]] = None,
        device: str = "cuda",
        use_sam: bool = True,
    ):
        if target is None and target_bbox is None:
            raise ValueError(
                "Provide either `target` (text, e.g. 'spoon') or `target_bbox=(x0,y0,x1,y1)`."
            )
        self.target = target
        self.target_bbox = target_bbox
        self.colors = colors or DEFAULT_COLORS
        self.device = device
        self.use_sam = use_sam
        self._mask_cache: dict[str, Optional[np.ndarray]] = {}

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        mask = self._get_mask(scene)
        if mask is None or int(mask.sum()) < _MIN_MASK_PIXELS:
            return                                       # not applicable
        image = self._scene_pil(scene)
        for color in self.colors:
            hue = COLOR_HUE.get(color)
            if hue is None:
                continue
            recolored = _recolor_with_mask(image, mask, hue)
            yield make_perturbed_image_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=f"to_{color}",
                new_image=recolored,
                description=f"target {self.target or 'bbox'} → {color}",
                parameters={
                    "color": color,
                    "hue_deg": hue,
                    "mask_pixels": int(mask.sum()),
                    "target": self.target,
                },
            )

    # -- helpers --------------------------------------------------------------

    def _get_mask(self, scene: Scene) -> Optional[np.ndarray]:
        key = scene.scene_id or id(scene)
        if key in self._mask_cache:
            return self._mask_cache[key]

        image = self._scene_pil(scene)
        bbox = self.target_bbox
        if bbox is None and self.target:
            try:
                bbox = _detect_bbox(image, self.target, device=self.device)
            except Exception as e:
                print(f"[ObjectRecolorPerturber] detection error: {type(e).__name__}: {e}")
                bbox = None
        if bbox is None:
            self._mask_cache[key] = None
            return None

        if self.use_sam:
            try:
                mask = _bbox_to_mask(image, bbox, device=self.device)
            except Exception as e:
                print(f"[ObjectRecolorPerturber] SAM error ({type(e).__name__}: {e}); "
                      f"falling back to rectangular bbox mask.")
                mask = None
            if mask is None:
                mask = _rectangular_mask(np.array(image).shape, bbox)
        else:
            mask = _rectangular_mask(np.array(image).shape, bbox)

        self._mask_cache[key] = mask
        return mask

    @staticmethod
    def _scene_pil(scene: Scene) -> Image.Image:
        if isinstance(scene.image, Image.Image):
            return scene.image.convert("RGB")
        return Image.fromarray(to_array(scene.image)).convert("RGB")
