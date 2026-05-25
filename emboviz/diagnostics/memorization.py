"""Memorization-sniff diagnostic.

If we mask out the target and the model still produces a sizeable, coherent
action — it's running on memorized trajectories, not visual feedback.

This is the LIBERO-Pro-style test (Geng et al. 2025): VLAs trained on a
narrow trajectory distribution often learn to replay motor patterns when
the visual context resembles training, regardless of whether the target
object is actually present.

Honest target localization is required for this test. We use:
  1. User-supplied bbox or mask, OR
  2. User-supplied detector (any callable matching the TargetDetector protocol), OR
  3. Default GroundingDINO + SAM zero-shot detection

If no detector can confidently locate the target, the diagnostic
returns Severity.UNKNOWN with a clear reason — we do NOT fall back to
masking a central rectangle (which would silently mask the gripper or
empty space and produce meaningless scores).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Scene
from emboviz.diagnostics.base import Diagnostic
from emboviz.models.protocol import Capability, VLAModel
from emboviz.perturb._target_detection import (
    BBoxDetector,
    GroundingDINOSAMDetector,
    TargetDetector,
)
from emboviz.perturb.image._image_utils import to_array, to_pil


class MemorizationDiagnostic(Diagnostic):
    """Mask the target across every camera; check whether the model still
    executes a coherent action.

    Multi-camera contract:
      • Target is detected independently per camera (each viewpoint sees the
        scene differently).
      • If the detector cannot locate the target with confidence on a given
        camera, that camera is honestly skipped — we do not fabricate a
        bbox there.
      • If NO camera produces a confident detection, the diagnostic returns
        Severity.UNKNOWN.
      • All cameras with detections are masked simultaneously in the same
        perturbed scene — exposing the action to the same intervention the
        model would see if every camera lost sight of the object.

    Args:
        target_detector: how to locate the target. Default = GD+SAM.
        bbox: shortcut — if you already know the target's bbox (same in
            every camera). Use only when cameras share resolution/intrinsics.
        coherent_threshold: action magnitude / divergence threshold for
            calling a model "memorizing." Defaults match LIBERO-Pro paper.
        cameras: which cameras to operate on. None = every camera in the scene.
    """

    required_capabilities = Capability.INFERENCE

    def __init__(
        self,
        target_detector: Optional[TargetDetector] = None,
        bbox: Optional[tuple[int, int, int, int]] = None,
        coherent_threshold: float = 0.20,
        cameras: Optional[list[str]] = None,
    ):
        if bbox is not None:
            self.detector: TargetDetector = BBoxDetector(bbox)
        elif target_detector is not None:
            self.detector = target_detector
        else:
            self.detector = GroundingDINOSAMDetector()
        self.coherent_threshold = coherent_threshold
        self.cameras = cameras
        self.name = "memorization_test"
        self.axis = "vision.memorization"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        from emboviz.core.types import resolve_cameras
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks INFERENCE capability")

        cameras = resolve_cameras(scene, self.cameras)
        baseline = model.predict(scene)

        # Detect target per camera. The detector currently consumes
        # ``scene.primary_image_data`` so we hand it a scene whose primary
        # alias points at the camera we're testing.
        per_cam_detection: dict = {}
        per_cam_masked_array: dict = {}
        for cam in cameras:
            cam_image = scene.observations.images[cam].data
            # Build a "probe scene" whose primary camera == this cam's image.
            # If the scene already has a "primary" alias, override it.
            if "primary" in scene.observations.images:
                probe_scene = scene.with_image(cam_image, camera="primary")
            else:
                probe_scene = scene.with_image(cam_image, camera=cam)
            detection = self.detector(probe_scene)
            per_cam_detection[cam] = detection
            if detection is None:
                continue
            arr = to_array(cam_image)
            chan_mean = arr.reshape(-1, 3).mean(axis=0).astype(np.uint8)
            masked_arr = arr.copy()
            if detection.mask is not None and detection.mask.shape == arr.shape[:2]:
                masked_arr[detection.mask] = chan_mean
            else:
                x0, y0, x1, y1 = detection.bbox
                x0 = max(0, x0); y0 = max(0, y0)
                x1 = min(arr.shape[1], x1); y1 = min(arr.shape[0], y1)
                masked_arr[y0:y1, x0:x1] = chan_mean
            per_cam_masked_array[cam] = masked_arr

        if not per_cam_masked_array:
            return self._not_applicable(
                model, scene,
                "could not confidently locate the manipulated target in ANY "
                "of the scene's cameras "
                f"({sorted(scene.observations.images)}) — provide an explicit "
                "bbox or a custom TargetDetector",
            )

        # Apply masks simultaneously across every camera that had a detection.
        masked_pils = {cam: to_pil(arr) for cam, arr in per_cam_masked_array.items()}
        masked_scene = scene.with_images(masked_pils)
        action_no_target = model.predict(masked_scene)

        # Reference: blank EVERY camera (not just primary) — keeps the
        # action's "no visual at all" baseline comparable across multi-cam.
        blank_pils: dict = {}
        for cam in cameras:
            arr = to_array(scene.observations.images[cam].data)
            blank_pils[cam] = to_pil(np.full_like(arr, fill_value=int(arr.mean())))
        blank_scene = scene.with_images(blank_pils)
        action_blank = model.predict(blank_scene)

        diff_vs_blank = float(np.linalg.norm(action_no_target.action - action_blank.action))
        diff_vs_baseline = float(np.linalg.norm(action_no_target.action - baseline.action))
        action_magnitude = float(np.linalg.norm(action_no_target.action))

        detected_cams = sorted(per_cam_masked_array)
        skipped_cams = sorted(c for c in cameras if c not in per_cam_masked_array)
        labels = sorted({per_cam_detection[c].label for c in detected_cams})
        confs = {c: round(per_cam_detection[c].confidence, 3) for c in detected_cams}

        if diff_vs_baseline < self.coherent_threshold and action_magnitude > self.coherent_threshold:
            sev = Severity.CRITICAL
            verdict = (
                f"Target masked across {detected_cams} (labels={labels}, "
                f"confs={confs}). The model still produces a substantial action "
                f"(‖a‖={action_magnitude:.3f}) nearly identical to baseline "
                f"(Δ={diff_vs_baseline:.3f}). It's memorizing the trajectory "
                f"rather than reading the scene."
            )
        elif diff_vs_baseline < 2 * self.coherent_threshold:
            sev = Severity.MODERATE
            verdict = (
                f"With target masked across {detected_cams} (labels={labels}), "
                f"the action stays similar (Δ={diff_vs_baseline:.3f}). Partial "
                f"memorization."
            )
        else:
            sev = Severity.PASS
            verdict = (
                f"With target masked across {detected_cams} (labels={labels}), "
                f"the model's action changes substantially (Δ={diff_vs_baseline:.3f}) "
                f"— it's reading visual feedback."
            )

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=diff_vs_baseline,
            severity=sev,
            direction="lower_is_worse",
            explanation=verdict,
            per_variant={
                "diff_vs_baseline": diff_vs_baseline,
                "diff_vs_blank": diff_vs_blank,
                "action_magnitude": action_magnitude,
                **{f"detected:{c}": 1.0 for c in detected_cams},
                **{f"skipped:{c}": 0.0 for c in skipped_cams},
            },
            raw={
                "baseline_action": baseline.action.tolist(),
                "action_target_masked": action_no_target.action.tolist(),
                "action_blank_scene": action_blank.action.tolist(),
                "detected_cameras": detected_cams,
                "skipped_cameras": skipped_cams,
                "per_camera_detection": {
                    c: {
                        "label": per_cam_detection[c].label,
                        "bbox": list(per_cam_detection[c].bbox),
                        "confidence": per_cam_detection[c].confidence,
                        "mask_provided": per_cam_detection[c].mask is not None,
                    } for c in detected_cams
                },
            },
        )
