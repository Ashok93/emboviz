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
    """Zero-shot open-vocabulary target detection via GroundingDINO + SAM.

    Pipeline (per LITERATURE.md §1):
      1. Pick the query phrase:
         • If ``target_text`` is set → use it as the GroundingDINO query.
           Example: ``GroundingDINOSAMDetector(target_text="the pipe")``.
         • Else → use ``scene.instruction`` directly as the phrase-grounding
           query. GroundingDINO is trained on phrase-grounding and will
           localize the referent of an instruction like "pick up the mug"
           or "move the bottom-right tip of the duvet to the left."
         • If neither is available (no target_text AND no instruction) →
           return None ("inconclusive: no phrase to ground").
      2. Run GroundingDINO with that phrase → bbox(es) + confidence scores.
      3. SAM refines the top bbox → pixel-accurate mask. SAM is REQUIRED
         (no bbox-only fallback); without it the intervention is too
         coarse to give an interpretable memorization verdict.
      4. Return TargetDetection or None (low confidence / no detection).

    We DO NOT extract nouns from a fixed taxonomy — that was an antipattern
    that silently skipped any instruction outside a 6-category lookup
    (utensils / food / toys / cloth / tools / containers). Real robot
    tasks reference "the lid", "the pipe", "the recycling bin",
    "the bottom-right tip of the duvet" — none of these are in a static
    taxonomy. Open-vocabulary phrase grounding is the literature-backed
    approach (Liu et al. 2024, GroundingDINO; Xiao et al. 2024,
    Florence-2 for referring-expression-segmentation).

    Heavy deps (transformers' GroundingDINO + SAM models) are lazy-imported.
    """

    def __init__(
        self,
        target_text: str,
        gd_repo: str = "IDEA-Research/grounding-dino-tiny",
        sam_repo: str = "facebook/sam-vit-base",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        min_confidence: float = 0.25,
        device: str = "cuda",
    ):
        """Args:
            target_text: REQUIRED. The phrase to mask in each frame —
                the user must specify what their policy is supposed to
                manipulate. Examples: ``"the mug"``, ``"the lid"``,
                ``"the welding torch"``, ``"the red pipe on the left"``.
                Memorization is a USER-SCOPED test (which object do you
                want to check the policy isn't ignoring?); we never
                guess.
            box_threshold, text_threshold: GroundingDINO confidence
                thresholds (defaults from the GD paper).
            min_confidence: detections with the top box's score below
                this are returned as None (the diagnostic then skips
                that frame with a clear reason).
        """
        if target_text is None or not str(target_text).strip():
            raise ValueError(
                "GroundingDINOSAMDetector requires a non-empty "
                "``target_text`` at construction. This is the phrase "
                "to mask — the user must say what their policy is "
                "supposed to manipulate (e.g. \"the mug\")."
            )
        self.gd_repo = gd_repo
        self.sam_repo = sam_repo
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_confidence = min_confidence
        self.device = device
        self.target_text = target_text
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
        if self._sam is None:
            # SAM is REQUIRED — no bbox-only fallback. A coarse bbox covers
            # the target plus background; the resulting "mask" intervention
            # is much weaker than intended and gives uninterpretable
            # memorization verdicts. If SAM can't load, the user needs to
            # know (install the right deps) rather than have results
            # silently degrade.
            try:
                from transformers import SamModel, SamProcessor
                sam_proc = SamProcessor.from_pretrained(self.sam_repo)
                sam_model = SamModel.from_pretrained(self.sam_repo).to(self.device).eval()
                self._sam = (sam_proc, sam_model)
            except (ImportError, OSError, RuntimeError) as e:
                raise RuntimeError(
                    f"GroundingDINOSAMDetector requires SAM ({self.sam_repo}) "
                    f"but it failed to load: {type(e).__name__}: {e}. SAM "
                    "provides the pixel-accurate masks the memorization "
                    "diagnostic needs — bbox-only masking is too coarse "
                    "and produces uninterpretable verdicts. Install the "
                    "SAM checkpoint or pass a different target_detector."
                ) from e

    def _query_phrase(self, scene: Scene) -> str:
        """Pick the GroundingDINO phrase-grounding query.

        ``target_text`` is **required** — the user must tell us what to
        mask. This is their use case and their model; only they know
        whether the test should obscure "the orange", "the spoon", "the
        welding torch", or "the recycling can". We do not guess by
        parsing the instruction and we do not fall back to feeding the
        full instruction to GroundingDINO — that conflates the test's
        scope ("what should be invisible to the policy?") with the
        policy's input ("what was the policy told to do?"), and gives
        different results depending on instruction phrasing.

        Raises:
            ValueError: if ``target_text`` was not set on the detector.
        """
        if self.target_text is None:
            raise ValueError(
                "GroundingDINOSAMDetector requires a non-empty "
                "``target_text`` — the phrase to mask in each frame. "
                "Memorization tests whether the policy is using vision "
                "for a specific object the user cares about (\"the mug\", "
                "\"the lid\", \"the welding torch\"). We do not guess "
                "the target from the policy's instruction; only the "
                "user knows what their model is supposed to manipulate. "
                "Set target_text when constructing the detector, e.g. "
                "GroundingDINOSAMDetector(target_text=\"the mug\")."
            )
        phrase = self.target_text.strip()
        if not phrase:
            raise ValueError(
                "GroundingDINOSAMDetector.target_text is set but empty."
            )
        return phrase

    def __call__(self, scene: Scene) -> Optional[TargetDetection]:
        """Detect the target object on the scene's *primary* camera.

        Multi-camera diagnostics that need per-camera detection should
        construct a probe scene whose primary alias points at the camera
        they want to inspect (see MemorizationDiagnostic for the pattern).
        Returns None when:
          • No phrase available (no target_text AND no instruction).
          • GroundingDINO returns no boxes above ``box_threshold``.
          • The top box's score is below ``min_confidence``.
        """
        import torch

        phrase = self._query_phrase(scene)
        if phrase is None:
            return None  # honest skip: nothing to ground

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
        # GroundingDINO expects the phrase to end with a period.
        text = phrase if phrase.endswith(".") else f"{phrase}."
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

        # SAM refinement — pixel-accurate mask. SAM was required at load
        # time, so any failure here is a real bug we want surfaced
        # (rather than papered over with a coarse bbox fill).
        sam_proc, sam_model = self._sam
        sam_inputs = sam_proc(
            pil, input_boxes=[[list(bbox)]], return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            sam_out = sam_model(**sam_inputs, multimask_output=False)
        masks = sam_proc.image_processor.post_process_masks(
            sam_out.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )
        if not masks or len(masks) == 0 or masks[0].shape[0] == 0:
            raise RuntimeError(
                f"SAM returned no mask for bbox {bbox} on a phrase that "
                f"GroundingDINO scored at {score:.3f}. Likely a SAM "
                "preprocessing edge case — investigate rather than fall "
                "back to bbox-only."
            )
        mask = masks[0][0][0].numpy().astype(bool)

        return TargetDetection(bbox=bbox, mask=mask, label=phrase, confidence=score)
