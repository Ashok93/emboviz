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

from emboviz.calibration import ModelCalibration, averaged_predict
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

    Calibration (recommended):
        When ``calibration`` is passed, the per-frame Δaction (raw L2 in the
        model's action units) is normalized into a 0-1 anchored score:
        ``score = max(0, raw_delta − noise_floor) / typical_action_magnitude``.
        Below the noise floor the score reads 0.0 (no signal); a score of
        ``>= grounded_threshold`` means the model is genuinely using vision.

    Args:
        target_detector: how to locate the target. Default = GD+SAM.
        bbox: shortcut — if you already know the target's bbox (same in
            every camera). Use only when cameras share resolution/intrinsics.
        noise_floor_score: anchored 0-1 score below which the model is
            "ignoring the intervention" (memorization signature).
        grounded_threshold: anchored 0-1 score above which the model is
            "genuinely reading the scene".
        cameras: which cameras to operate on. None = every camera in the scene.
        calibration: per-model anchors from ``emboviz.calibration.calibrate_model``.
            When None, the diagnostic falls back to raw L2 scores and a
            single hardcoded threshold — interpretable only against this
            model's own scale.
    """

    required_capabilities = Capability.INFERENCE

    def __init__(
        self,
        target_detector: Optional[TargetDetector] = None,
        bbox: Optional[tuple[int, int, int, int]] = None,
        noise_floor_score: float = 0.05,
        grounded_threshold: float = 0.30,
        cameras: Optional[list[str]] = None,
        calibration: Optional["ModelCalibration"] = None,
    ):
        if bbox is not None:
            self.detector: TargetDetector = BBoxDetector(bbox)
        elif target_detector is not None:
            self.detector = target_detector
        else:
            self.detector = GroundingDINOSAMDetector()
        self.noise_floor_score = noise_floor_score
        self.grounded_threshold = grounded_threshold
        self.cameras = cameras
        self.calibration = calibration
        self.name = "memorization_test"
        self.axis = "vision.memorization"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        from emboviz.core.types import resolve_cameras
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks INFERENCE capability")

        cameras = resolve_cameras(scene, self.cameras)
        n_samples = self.calibration.n_samples if self.calibration else 1
        baseline = averaged_predict(model, scene, n_samples)

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
        action_no_target = averaged_predict(model, masked_scene, n_samples)

        # Reference: blank EVERY camera (not just primary) — keeps the
        # action's "no visual at all" baseline comparable across multi-cam.
        blank_pils: dict = {}
        for cam in cameras:
            arr = to_array(scene.observations.images[cam].data)
            blank_pils[cam] = to_pil(np.full_like(arr, fill_value=int(arr.mean())))
        blank_scene = scene.with_images(blank_pils)
        action_blank = averaged_predict(model, blank_scene, n_samples)

        raw_diff_vs_blank = float(np.linalg.norm(action_no_target.action - action_blank.action))
        raw_diff_vs_baseline = float(np.linalg.norm(action_no_target.action - baseline.action))
        action_magnitude = float(np.linalg.norm(action_no_target.action))

        detected_cams = sorted(per_cam_masked_array)
        skipped_cams = sorted(c for c in cameras if c not in per_cam_masked_array)
        labels = sorted({per_cam_detection[c].label for c in detected_cams})
        confs = {c: round(per_cam_detection[c].confidence, 3) for c in detected_cams}

        # Anchored 0-1 score. With properly calibrated n_samples (computed
        # from the model's noise + a precision target by calibrate_model),
        # the averaged noise floor is bounded below precision_target × typical.
        # So the diagnostic always returns a confident answer:
        #   • score = 0 → model genuinely didn't respond (true memorization)
        #   • 0 < score < grounded_threshold → partial response
        #   • score >= grounded_threshold → real visual grounding
        # No UNKNOWN. If the user wants the diagnostic to run, calibration
        # ensured we can give them a real answer.
        if self.calibration is not None:
            score = self.calibration.normalize(raw_diff_vs_baseline)
            score_meaning = (
                f"normalized: {score:.3f}  (= max(0, raw_Δ {raw_diff_vs_baseline:.3f} "
                f"− noise_floor {self.calibration.noise_floor:.3f}) "
                f"/ typical_action_magnitude {self.calibration.typical_action_magnitude:.3f}, "
                f"n_samples={self.calibration.n_samples})"
            )
        else:
            score = raw_diff_vs_baseline
            score_meaning = (
                f"raw L2 (uncalibrated): {raw_diff_vs_baseline:.3f} — pass a "
                "ModelCalibration to anchor onto a 0-1 scale"
            )

        if score < self.noise_floor_score:
            sev = Severity.CRITICAL
            verdict = (
                f"Target masked across {detected_cams} (labels={labels}, "
                f"confs={confs}). Normalized response {score:.3f} is below "
                f"the memorization threshold ({self.noise_floor_score}). "
                f"Strong memorization signature — model does not respond to "
                f"target removal. [{score_meaning}]"
            )
        elif score < self.grounded_threshold:
            sev = Severity.MODERATE
            verdict = (
                f"With target masked across {detected_cams} (labels={labels}), "
                f"normalized response {score:.3f} is between memorization "
                f"threshold ({self.noise_floor_score}) and grounded threshold "
                f"({self.grounded_threshold}). Partial visual grounding. "
                f"[{score_meaning}]"
            )
        else:
            sev = Severity.PASS
            verdict = (
                f"With target masked across {detected_cams} (labels={labels}), "
                f"normalized response {score:.3f} >= grounded threshold "
                f"({self.grounded_threshold}). Model is reading visual "
                f"feedback. "
                f"[{score_meaning}]"
            )

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=score,
            severity=sev,
            direction="lower_is_worse",
            explanation=verdict,
            per_variant={
                "score_normalized":   score,
                "raw_diff_vs_baseline": raw_diff_vs_baseline,
                "raw_diff_vs_blank":  raw_diff_vs_blank,
                "action_magnitude":   action_magnitude,
                **{f"detected:{c}": 1.0 for c in detected_cams},
                **{f"skipped:{c}": 0.0 for c in skipped_cams},
            },
            raw={
                "baseline_action":         baseline.action.tolist(),
                "action_target_masked":    action_no_target.action.tolist(),
                "action_blank_scene":      action_blank.action.tolist(),
                "raw_diff_vs_baseline":    raw_diff_vs_baseline,
                "raw_diff_vs_blank":       raw_diff_vs_blank,
                "score_normalized":        score,
                "calibration_used":        self.calibration.to_summary() if self.calibration else None,
                "detected_cameras":        detected_cams,
                "skipped_cameras":         skipped_cams,
                "per_camera_detection": {
                    c: {
                        "label":         per_cam_detection[c].label,
                        "bbox":          list(per_cam_detection[c].bbox),
                        "confidence":    per_cam_detection[c].confidence,
                        "mask_provided": per_cam_detection[c].mask is not None,
                    } for c in detected_cams
                },
            },
        )
