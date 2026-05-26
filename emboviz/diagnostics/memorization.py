"""Memorization-sniff diagnostic — does the policy USE its visual input?

If we mask out the target and the model still produces an action close to
the unmasked baseline, it's running on memorized trajectories conditioned
on non-visual signals (proprio, instruction, action history), not on
visual feedback. This is the LIBERO-Pro signature (Geng et al. 2025,
arXiv:2510.03827) and the BYOVLA visual-robustness probe inverted
(Hancock et al. 2024, arXiv:2410.01971).

Implementation principles (LITERATURE.md §1):

  1. **Phrase grounding, no taxonomy.** Target is located by passing
     ``scene.instruction`` (or an explicit ``target_text`` override) to
     GroundingDINO + SAM. We never extract a noun from a fixed
     taxonomy — that silently skips any out-of-taxonomy task.

  2. **Fill ensemble.** We mask with TWO independent fills (channel-mean
     and Gaussian blur) and require AGREEMENT across both fills before
     calling memorization. Single-fill is susceptible to "baseline
     blindness" (Sturmfels, Lundberg & Lee 2020): if the masked region
     matches the fill color the model treats it as informative noise.

  3. **Intervention magnitude reporting.** For every fill we report
     ``mask_contrast`` — the pixel-L2 difference between the masked
     region's original content and the fill, normalized to [0, 1].
     This is the "did the image actually change?" sanity gate from
     Vig et al. 2020 causal-mediation framing. If contrast is too low,
     the intervention didn't happen (e.g. fill ≈ target color); we
     refuse to emit a verdict.

  4. **Per-camera detection and masking.** Each camera in the scene is
     queried independently. We mask EVERY camera that produced a
     confident detection, simultaneously, in one perturbed scene.

  5. **N-sample averaging for stochastic policies.** π0, GR00T,
     diffusion policies need K samples per prediction averaged before
     comparison (handled via ``averaged_predict`` and the
     ``ModelCalibration.n_samples`` derivation).

  6. **No SAM-fallback.** SAM is required (no bbox-only fallback) —
     the GroundingDINOSAMDetector raises at load if SAM is unavailable.
     A coarse bbox covers target + background and is too weak.
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


# Minimum normalized contrast (in [0,1]) between fill colour and the
# original target region for the intervention to count as "real." Below
# this the mask is visually indistinguishable from the target and the
# Δaction is uninterpretable. Default 0.05 (≈5% of pixel range) is a
# conservative threshold — for a 256-grey image, 13 steps of difference.
DEFAULT_MIN_MASK_CONTRAST = 0.05


def _apply_fill(arr: np.ndarray, mask: np.ndarray, fill_mode: str) -> np.ndarray:
    """Apply one of the supported fills to the mask region.

    ``arr`` is HxWxC uint8. ``mask`` is HxW bool. Returns a new uint8
    array with the masked region replaced.

    Fills:
      • ``"channel_mean"``  — fill with the per-channel mean of the
        whole image (the Zeiler-Fergus 2014 baseline). Strong intervention
        but can suffer baseline-blindness if target is itself near-mean.
      • ``"gaussian_blur"`` — replace the masked region with a heavily-
        blurred version of ITSELF. Stays on-manifold (doesn't introduce
        an alien colour) while still destroying high-frequency target
        content. Sigma scaled with the mask's bbox diameter so the blur
        magnitude matches target size.
    """
    out = arr.copy()
    if fill_mode == "channel_mean":
        chan_mean = arr.reshape(-1, arr.shape[-1]).mean(axis=0).astype(np.uint8)
        out[mask] = chan_mean
        return out
    if fill_mode == "gaussian_blur":
        # Sigma proportional to mask diameter (bbox-based). For a 50x50
        # mask in a 480x640 image, sigma ≈ 50/3 ≈ 17 pixels — strong
        # blur that removes object identity but preserves rough colour.
        ys, xs = np.where(mask)
        if ys.size == 0:
            return out
        bbox_diag = float(
            np.hypot(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
        )
        sigma = max(2.0, bbox_diag / 6.0)
        # Lazy import scipy only when needed.
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError as e:
            raise ImportError(
                "memorization gaussian_blur fill requires scipy. "
                "Install scipy or pass fill_modes=['channel_mean'] only."
            ) from e
        # Blur the WHOLE image, then paste blurred pixels into masked area.
        blurred = np.stack([
            gaussian_filter(arr[..., c].astype(np.float32), sigma=sigma)
            for c in range(arr.shape[-1])
        ], axis=-1).astype(np.uint8)
        out[mask] = blurred[mask]
        return out
    raise ValueError(
        f"Unknown fill_mode={fill_mode!r}. Supported: 'channel_mean', "
        "'gaussian_blur'."
    )


def _mask_contrast(
    original: np.ndarray, masked: np.ndarray, mask: np.ndarray,
) -> float:
    """Pixel-L2 difference between original and masked image, restricted
    to the masked region, normalized to [0, 1].

    Approximates LPIPS (which would require a learned perceptual model)
    at a fraction of the cost. A value of 0 means the mask is identical
    to the original; 1 means maximum possible pixel difference.
    """
    if not mask.any():
        return 0.0
    diff = (original.astype(np.float32) - masked.astype(np.float32))
    if diff.ndim == 3:
        per_pixel = np.linalg.norm(diff, axis=-1)
    else:
        per_pixel = np.abs(diff)
    masked_pixels = per_pixel[mask]
    # Max possible per-pixel L2 difference for 3-channel uint8 = sqrt(3) * 255
    max_per_pixel = float(np.sqrt(diff.shape[-1] if diff.ndim == 3 else 1) * 255.0)
    return float(masked_pixels.mean() / max_per_pixel)


class MemorizationDiagnostic(Diagnostic):
    """Mask the target across every camera; check whether the model still
    executes a coherent action.

    Args:
        target_detector: how to locate the target. Default = GD+SAM with
            scene.instruction as the query phrase. Pass a custom
            ``GroundingDINOSAMDetector(target_text="the pipe")`` to
            scope to a specific referent in multi-object instructions.
        bbox: shortcut — fixed bbox in every camera (only valid when
            cameras share resolution/intrinsics).
        fill_modes: which fills to ensemble. Default both — agreement
            required for CRITICAL verdict. Pass ``["channel_mean"]`` to
            skip the blur fill if scipy is unavailable.
        min_mask_contrast: refuse to emit CRITICAL when ALL fills
            produce a normalized contrast below this on a frame. Default
            ``DEFAULT_MIN_MASK_CONTRAST`` (0.05).
        noise_floor_score: anchored 0-1 score below which the model is
            "ignoring the intervention" (memorization signature).
        grounded_threshold: anchored 0-1 score above which the model is
            "genuinely reading the scene."
        cameras: which cameras to operate on. None = every camera in the scene.
        calibration: per-model anchors from
            ``emboviz.calibration.calibrate_model``. Required for
            anchored thresholds to mean the same thing across models;
            without it the diagnostic reports raw L2 scores.
    """

    required_capabilities = Capability.INFERENCE

    def __init__(
        self,
        target_detector: Optional[TargetDetector] = None,
        bbox: Optional[tuple[int, int, int, int]] = None,
        fill_modes: Optional[list[str]] = None,
        min_mask_contrast: float = DEFAULT_MIN_MASK_CONTRAST,
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
        self.fill_modes = list(fill_modes) if fill_modes else ["channel_mean", "gaussian_blur"]
        self.min_mask_contrast = float(min_mask_contrast)
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

        # 1. Per-camera target detection.
        per_cam_detection: dict = {}
        per_cam_original: dict = {}
        for cam in cameras:
            cam_image = scene.observations.images[cam].data
            if "primary" in scene.observations.images:
                probe_scene = scene.with_image(cam_image, camera="primary")
            else:
                probe_scene = scene.with_image(cam_image, camera=cam)
            detection = self.detector(probe_scene)
            per_cam_detection[cam] = detection
            if detection is None:
                continue
            arr = to_array(cam_image)
            if detection.mask is None or detection.mask.shape != arr.shape[:2]:
                # Detector contract: mask must be HxW matching image.
                # GroundingDINOSAMDetector enforces this; a custom
                # detector that violates it must be caught.
                raise ValueError(
                    f"Target detector for camera '{cam}' returned a "
                    f"detection without a pixel-accurate mask (got "
                    f"mask={detection.mask!r}). Memorization requires a "
                    "pixel mask; bbox-only is too coarse. Use a detector "
                    "that produces a mask (GroundingDINOSAMDetector default)."
                )
            per_cam_original[cam] = arr

        if not per_cam_original:
            return self._not_applicable(
                model, scene,
                "could not confidently locate the manipulated target in ANY "
                f"of the scene's cameras ({sorted(scene.observations.images)}). "
                "Try providing an explicit ``target_text`` to "
                "GroundingDINOSAMDetector, or a custom TargetDetector. We "
                "never fabricate a centred rectangle.",
            )

        detected_cams = sorted(per_cam_original)
        labels = sorted({per_cam_detection[c].label for c in detected_cams})
        confs = {c: round(per_cam_detection[c].confidence, 3) for c in detected_cams}

        # 2. Fill ensemble: for each fill mode, build masked scene,
        # measure intervention magnitude (mask_contrast) and response
        # magnitude (Δaction vs baseline).
        per_fill_results: dict[str, dict] = {}
        for fill_mode in self.fill_modes:
            masked_arrays: dict[str, np.ndarray] = {}
            contrasts: dict[str, float] = {}
            for cam in detected_cams:
                arr = per_cam_original[cam]
                mask = per_cam_detection[cam].mask
                masked = _apply_fill(arr, mask, fill_mode)
                masked_arrays[cam] = masked
                contrasts[cam] = _mask_contrast(arr, masked, mask)
            mean_contrast = float(np.mean(list(contrasts.values())))

            masked_pils = {cam: to_pil(a) for cam, a in masked_arrays.items()}
            masked_scene = scene.with_images(masked_pils)
            action_masked = averaged_predict(model, masked_scene, n_samples)
            raw_delta = float(np.linalg.norm(action_masked.action - baseline.action))
            if self.calibration is not None:
                norm_delta = self.calibration.normalize(raw_delta)
            else:
                norm_delta = raw_delta

            per_fill_results[fill_mode] = {
                "mean_contrast":   mean_contrast,
                "per_cam_contrast": contrasts,
                "raw_delta":       raw_delta,
                "normalized_delta": norm_delta,
                "action_masked":   action_masked.action.tolist(),
            }

        # 3. Intervention validity gate. If EVERY fill produced contrast
        # below ``min_mask_contrast`` the intervention was effectively
        # invisible — refuse a verdict.
        max_contrast = max(r["mean_contrast"] for r in per_fill_results.values())
        if max_contrast < self.min_mask_contrast:
            return self._not_applicable(
                model, scene,
                f"intervention too weak to test memorization on "
                f"{detected_cams}: max fill contrast {max_contrast:.3f} < "
                f"{self.min_mask_contrast} on every fill mode "
                f"({list(per_fill_results)}). The target's pixels are "
                "themselves close to every available fill colour, so the "
                "masked image is visually indistinguishable from the "
                "original. Either supply a higher-contrast fill mode or "
                "skip this frame.",
            )

        # 4. Verdict — require agreement across all fills with sufficient
        # contrast. Strongest fill = highest contrast (most aggressive
        # intervention); we trust its Δaction as the headline signal,
        # but bin severity based on the worst-case (max) fill Δaction
        # to be conservative about "ignored" verdicts.
        deltas = [r["normalized_delta"] for r in per_fill_results.values()]
        min_delta = float(min(deltas))   # most conservative — worst signal across fills
        max_delta = float(max(deltas))
        mean_delta = float(np.mean(deltas))

        # The principle (LITERATURE.md §1 step 6):
        #   MEMORIZATION_SIGNATURE iff max(δ) < threshold  (all fills agree → small)
        #   VISUALLY_GROUNDED iff min(δ) > threshold        (all fills agree → large)
        #   else MIXED
        if max_delta < self.noise_floor_score:
            sev = Severity.CRITICAL
            verdict = (
                f"All fills ({list(per_fill_results)}) produce normalized "
                f"Δaction < {self.noise_floor_score} when target masked on "
                f"{detected_cams} (labels={labels}, confs={confs}). Strong "
                f"memorization signature: model does not respond to target "
                f"removal under any fill — max Δ={max_delta:.3f}, "
                f"intervention contrast={max_contrast:.3f}."
            )
        elif min_delta > self.grounded_threshold:
            sev = Severity.PASS
            verdict = (
                f"All fills produce normalized Δaction > "
                f"{self.grounded_threshold} when target masked. Model is "
                f"reading visual feedback: min Δ={min_delta:.3f}, "
                f"max Δ={max_delta:.3f} across fills."
            )
        elif max_delta < self.grounded_threshold:
            sev = Severity.MODERATE
            verdict = (
                f"Mixed response: max Δ={max_delta:.3f} sits between "
                f"noise floor ({self.noise_floor_score}) and grounded "
                f"threshold ({self.grounded_threshold}). Partial visual "
                f"grounding — model uses some target cues but not "
                f"decisively."
            )
        else:
            # Fills disagree (some pass, some critical-ish). Treat as mixed.
            sev = Severity.MODERATE
            verdict = (
                f"Fills disagree: min Δ={min_delta:.3f}, max Δ={max_delta:.3f}. "
                f"At least one fill ({max(per_fill_results, key=lambda k: per_fill_results[k]['normalized_delta'])}) "
                f"shows real visual sensitivity, another doesn't. Likely "
                "fill-specific artifacts; recommend manual review of the "
                "per-fill raw output."
            )

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=mean_delta,
            severity=sev,
            direction="lower_is_worse",
            explanation=verdict,
            per_variant={
                "mean_delta_across_fills": mean_delta,
                "min_delta_across_fills":  min_delta,
                "max_delta_across_fills":  max_delta,
                "mean_contrast_across_fills": max_contrast,
                **{
                    f"fill:{fm}:normalized_delta": r["normalized_delta"]
                    for fm, r in per_fill_results.items()
                },
                **{
                    f"fill:{fm}:mean_contrast": r["mean_contrast"]
                    for fm, r in per_fill_results.items()
                },
                **{f"detected:{c}": 1.0 for c in detected_cams},
            },
            raw={
                "baseline_action":      baseline.action.tolist(),
                "per_fill":             per_fill_results,
                "calibration_used":     self.calibration.to_summary() if self.calibration else None,
                "min_mask_contrast":    self.min_mask_contrast,
                "noise_floor_score":    self.noise_floor_score,
                "grounded_threshold":   self.grounded_threshold,
                "detected_cameras":     detected_cams,
                "skipped_cameras":      sorted(
                    c for c in cameras if c not in per_cam_original
                ),
                "per_camera_detection": {
                    c: {
                        "label":      per_cam_detection[c].label,
                        "bbox":       list(per_cam_detection[c].bbox),
                        "confidence": per_cam_detection[c].confidence,
                    } for c in detected_cams
                },
            },
        )
