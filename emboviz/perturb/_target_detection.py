"""Target detection — the shared "where is the manipulated object" interface.

Multiple diagnostics need to know where the target object is in the frame:
  • MemorizationDiagnostic — to mask the target and check if action still moves
  • ObjectRecolorPerturber — to recolor the target via SAM mask
  • Future: per-target sensitivity maps, attention-target alignment, etc.

Detection lives behind a `TargetDetector` protocol so users can:
  1. Pass an explicit bbox/mask (their tracking already knows where the target is)
  2. Plug in their own fine-tuned detector (custom GroundingDINO / SAM / YOLO)
  3. Use the default GroundingDINO + SAM pipeline (extracts the noun from the
     instruction and runs zero-shot detection)

Honest principle: if detection confidence is low and no fallback is supplied,
return None. We do NOT default to a "centered bbox" hack — that's silently
wrong (might mask the gripper, the table, or empty space) and contaminates
downstream diagnostic scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from emboviz.core.types import Scene


@dataclass(frozen=True)
class TargetDetection:
    """A target localization result.

    `bbox` is (x0, y0, x1, y1) in pixel coordinates of the primary image.
    `mask` is an optional binary HxW array (True = target pixels) — populated
    by detectors that produce segmentation (SAM); bbox-only detectors leave it None.
    `label` is what the detector thought it found (for debugging / logging).
    `confidence` is the detector's score in [0, 1] if available.
    """

    bbox: tuple[int, int, int, int]
    mask: Optional[np.ndarray] = None
    label: str = ""
    confidence: float = 1.0


class TargetDetector(Protocol):
    """A callable that locates a target in a Scene.

    Implementations must return None if they cannot find the target with
    acceptable confidence — diagnostics treat None as "skip with reason"
    rather than fabricating an answer.
    """

    def __call__(self, scene: Scene) -> Optional[TargetDetection]: ...


class BBoxDetector:
    """Trivial detector that returns a fixed user-supplied bbox.

    Use when your tracking system already knows where the target is (motion
    capture, fiducials, prior frame's detection passed through, etc.).
    """

    def __init__(self, bbox: tuple[int, int, int, int], label: str = "user_supplied"):
        self._bbox = bbox
        self._label = label

    def __call__(self, scene: Scene) -> Optional[TargetDetection]:
        return TargetDetection(bbox=self._bbox, label=self._label, confidence=1.0)


class GroundingDINOSAMDetector:
    """Zero-shot target detection from the instruction.

    Pipeline:
      1. Parse the manipulated noun out of the instruction (via the same
         taxonomy NounSwapPerturber uses).
      2. Run GroundingDINO with that noun as the text prompt → bbox(es).
      3. (Optional) Refine the top bbox with SAM → segmentation mask.
      4. Return TargetDetection or None (low confidence / no detection).

    Heavy deps (transformers' GroundingDINO + SAM models) are lazy-imported.
    Users who don't run memorization or recolor diagnostics never load them.
    """

    def __init__(
        self,
        gd_repo: str = "IDEA-Research/grounding-dino-tiny",
        sam_repo: str = "facebook/sam-vit-base",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        min_confidence: float = 0.25,
        device: str = "cuda",
        use_sam: bool = True,
    ):
        self.gd_repo = gd_repo
        self.sam_repo = sam_repo
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_confidence = min_confidence
        self.device = device
        self.use_sam = use_sam
        self._gd = None  # (processor, model)
        self._sam = None

    def _ensure_loaded(self):
        if self._gd is None:
            try:
                import torch
                from transformers import (
                    AutoModelForZeroShotObjectDetection,
                    AutoProcessor,
                )
            except ImportError as e:
                raise ImportError(
                    "GroundingDINOSAMDetector requires `transformers`. "
                    "Install via your model adapter's optional deps."
                ) from e
            proc = AutoProcessor.from_pretrained(self.gd_repo)
            model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(self.gd_repo)
                .to(self.device).eval()
            )
            self._gd = (proc, model)
        if self.use_sam and self._sam is None:
            try:
                from transformers import SamModel, SamProcessor
                sam_proc = SamProcessor.from_pretrained(self.sam_repo)
                sam_model = SamModel.from_pretrained(self.sam_repo).to(self.device).eval()
                self._sam = (sam_proc, sam_model)
            except (ImportError, OSError, RuntimeError) as e:
                import warnings as _warnings
                _warnings.warn(
                    f"SAM ({self.sam_repo}) failed to load: "
                    f"{type(e).__name__}: {e}. Falling back to bbox-only "
                    "detection (no segmentation mask).",
                    stacklevel=2,
                )
                self._sam = None
                self.use_sam = False

    def _pick_noun(self, instruction: str) -> Optional[str]:
        """Extract the manipulated noun from the instruction.

        Reuses NounSwapPerturber's priority-ordered category lookup
        (utensil > food > toy > cloth > tool > container).
        """
        from emboviz.taxonomy.object_categories import OBJECT_CATEGORIES
        priority = ["utensil", "food", "toy", "cloth", "tool", "container"]
        words = (instruction or "").lower().split()
        for cat in priority:
            for w in words:
                stripped = w.strip(".,!?;:")
                if stripped in OBJECT_CATEGORIES.get(cat, []):
                    return stripped
        return None

    def __call__(self, scene: Scene) -> Optional[TargetDetection]:
        """Detect the target object on the scene's *primary* camera.

        Multi-camera diagnostics that need per-camera detection should
        construct a probe scene whose primary alias points at the camera
        they want to inspect (see MemorizationDiagnostic for the pattern).
        Returns None if the instruction has no recognised target noun or
        if detection confidence is below ``min_confidence`` — never
        fabricates a result.
        """
        import torch

        if scene.instruction is None or not scene.instruction.strip():
            return None  # No instruction → no target noun → honest skip
        noun = self._pick_noun(scene.instruction)
        if noun is None:
            return None  # No recognizable target noun — skip honestly

        self._ensure_loaded()
        proc, model = self._gd
        if "primary" not in scene.observations.images:
            raise ValueError(
                "GroundingDINOSAMDetector expects a 'primary' camera in the "
                f"scene (available: {sorted(scene.observations.images)}). "
                "Build a probe scene that aliases the camera you want to "
                "inspect under the name 'primary'."
            )
        pil = scene.observations.images["primary"].data
        # GroundingDINO expects period-separated phrases
        text = f"{noun}."
        inputs = proc(images=pil, text=text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = torch.tensor([pil.size[::-1]]).to(self.device)
        results = proc.post_process_grounded_object_detection(
            outputs, input_ids=inputs["input_ids"],
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )[0]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        if boxes.size == 0:
            return None
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score < self.min_confidence:
            return None
        x0, y0, x1, y1 = boxes[best].astype(int)
        bbox = (int(x0), int(y0), int(x1), int(y1))

        # SAM refinement
        mask = None
        if self.use_sam and self._sam is not None:
            try:
                sam_proc, sam_model = self._sam
                sam_inputs = sam_proc(pil, input_boxes=[[list(bbox)]], return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    sam_out = sam_model(**sam_inputs, multimask_output=False)
                masks = sam_proc.image_processor.post_process_masks(
                    sam_out.pred_masks.cpu(),
                    sam_inputs["original_sizes"].cpu(),
                    sam_inputs["reshaped_input_sizes"].cpu(),
                )
                if masks and len(masks) > 0:
                    m = masks[0][0][0].numpy()
                    mask = m.astype(bool)
            except (RuntimeError, ValueError, AttributeError) as e:
                import warnings as _warnings
                _warnings.warn(
                    f"SAM segmentation failed (post-bbox): "
                    f"{type(e).__name__}: {e}. Returning bbox-only "
                    "detection for this scene.",
                    stacklevel=2,
                )
                mask = None

        return TargetDetection(bbox=bbox, mask=mask, label=noun, confidence=score)
