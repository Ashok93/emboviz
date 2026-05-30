"""BYOVLA-style per-region sensitivity map.

For each camera the diagnostic iterates an N×N grid, masks one cell at a
time with the channel mean, and measures the resulting Δaction. The
resulting per-camera heatmap shows which scene regions causally drive
the policy. Useful for: distinguishing "model focuses on target" from
"model focuses on background or distractor", and (with multi-camera)
"model relies on the wrist camera but ignores the head camera".

Multi-camera contract:
  • ``cameras=None`` (default) → run the grid per camera in the scene.
  • ``cameras=["primary"]`` → only that camera.
  • Per-camera concentration scores live in ``per_variant`` keyed by camera.
  • Aggregate ``scalar_score`` = the MEAN concentration across the cameras
    the model actually consumed (cameras whose grid response never clears
    the noise floor are reported separately as ``ignored_cameras``, not
    folded into the headline number where they'd read as misleading zeros).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.calibration import ModelCalibration, averaged_predict, averaged_predict_batch
from emboviz.core.results import DiagnosticResult, Finding, Severity
from emboviz.core.types import ActionResult, Scene, resolve_cameras
from emboviz.diagnostics.base import Diagnostic
from emboviz.metrics.action_divergence import ActionDivergenceMetric
from emboviz.models.protocol import Capability, VLAModel
from emboviz.perturb.image._image_utils import to_array, to_pil


class SensitivityMapDiagnostic(Diagnostic):
    """For each camera, mask each grid cell one-at-a-time and measure |Δaction|."""

    required_capabilities = Capability.INFERENCE

    def __init__(
        self,
        grid_side: int = 8,
        metric: Optional[ActionDivergenceMetric] = None,
        cameras: Optional[list[str]] = None,
        calibration: Optional[ModelCalibration] = None,
    ):
        self.grid_side = grid_side
        self._metric_override = metric
        self.cameras = cameras
        self.calibration = calibration
        self.name = f"sensitivity_map_{grid_side}x{grid_side}"
        self.axis = "vision.scene_sensitivity"

    def run(
        self, model: VLAModel, scene: Scene,
        *, baseline: Optional[ActionResult] = None,
    ) -> DiagnosticResult:
        """Run the sensitivity grid for ``scene``.

        ``baseline`` is an optional precomputed unperturbed prediction —
        the runner computes it once per frame and shares it across all
        diagnostics. Without this we'd waste ``n_samples`` forward passes
        per diagnostic per frame on the same baseline.
        """
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks INFERENCE capability")

        cameras = resolve_cameras(scene, self.cameras)
        metric = self._metric_override or ActionDivergenceMetric(model=model)
        n_samples = self.calibration.n_samples if self.calibration else 1
        if baseline is None:
            baseline = averaged_predict(model, scene, n_samples)

        per_camera_grid: dict[str, np.ndarray] = {}
        per_camera_top_k: dict[str, float] = {}
        per_camera_consumed: dict[str, bool] = {}
        per_camera_image_shape: dict[str, tuple[int, int]] = {}

        # Calibration-aware "consumed" threshold. A camera is considered
        # consumed only when the per-cell Δaction sits ABOVE the model's
        # measured noise floor — comparing against ``1e-9`` is a bug
        # because real noise floors are ~1e-4 to 1e-2 in action units, so
        # noise-only grids always cleared the old threshold. We subtract
        # the noise floor from each cell (the same "max(0, raw - noise)"
        # operation ``ModelCalibration.normalize`` performs), and only
        # mark a camera consumed when the maximum signal-above-noise per
        # cell exceeds a meaningful fraction of the typical action.
        if self.calibration is not None:
            cell_noise_floor = self.calibration.noise_floor
            cell_signal_threshold = (
                self.calibration.typical_action_magnitude * 0.05
            )
        else:
            cell_noise_floor = 0.0
            cell_signal_threshold = 1e-6

        # Phase 1 — build EVERY masked scene (all cameras × grid cells) up
        # front, each masking one patch with that camera's channel mean.
        # These forward passes are mutually independent, so we submit them
        # as ONE batch (Phase 2) instead of grid² sequential round-trips
        # per camera — the GPU runs them in parallel.
        cam_hw: dict[str, tuple[int, int]] = {}
        pert_scenes: list[Scene] = []
        pert_index: list[tuple[str, int, int]] = []   # (cam, gi, gj) per scene
        for cam in cameras:
            arr = to_array(scene.observations.images[cam].data)
            H, W = arr.shape[:2]
            chan_mean = arr.reshape(-1, 3).mean(axis=0)
            ph = H // self.grid_side
            pw = W // self.grid_side
            cam_hw[cam] = (H, W)
            for gi in range(self.grid_side):
                for gj in range(self.grid_side):
                    masked = arr.copy()
                    y0, x0 = gi * ph, gj * pw
                    masked[y0:y0 + ph, x0:x0 + pw] = chan_mean
                    pert_scenes.append(scene.with_image(to_pil(masked), camera=cam))
                    pert_index.append((cam, gi, gj))

        # Phase 2 — one batched prediction for every masked cell.
        preds = averaged_predict_batch(model, pert_scenes, n_samples)

        # Phase 3 — scatter Δaction into each camera's grid, then the
        # per-camera signal / concentration logic (unchanged).
        drops_by_cam = {
            cam: np.zeros((self.grid_side, self.grid_side), dtype=np.float32)
            for cam in cameras
        }
        for (cam, gi, gj), pert in zip(pert_index, preds):
            drops_by_cam[cam][gi, gj] = metric.compute(baseline, pert)

        for cam in cameras:
            H, W = cam_hw[cam]
            # Subtract noise floor — what's left is real signal above noise.
            signal = np.maximum(drops_by_cam[cam] - cell_noise_floor, 0.0)
            per_camera_grid[cam] = signal
            per_camera_image_shape[cam] = (H, W)
            flat = signal.flatten()
            total = float(flat.sum())
            max_cell = float(flat.max()) if flat.size else 0.0
            # A camera is "consumed" iff the strongest single-cell signal
            # above noise exceeds ``cell_signal_threshold`` (default: 5% of
            # typical action magnitude). Otherwise the entire grid is at
            # or below noise — model is not responding to spatial masking.
            if max_cell < cell_signal_threshold or total < cell_noise_floor:
                per_camera_top_k[cam] = 0.0
                per_camera_consumed[cam] = False
            else:
                per_camera_top_k[cam] = float(
                    np.sort(flat)[-self.grid_side:].sum() / total
                )
                per_camera_consumed[cam] = True

        consumed_cams = [c for c in cameras if per_camera_consumed[c]]
        ignored_cams = [c for c in cameras if not per_camera_consumed[c]]

        # Scalar: mean concentration across CONSUMED cameras only. The fact
        # that ignored cameras exist is reported separately in the verdict
        # (not folded into the headline number as a misleading min()).
        if consumed_cams:
            scalar = float(np.mean([per_camera_top_k[c] for c in consumed_cams]))
        else:
            scalar = 0.0

        raw_numbers = {
            "concentration":           scalar,
            "consumed_cameras":        consumed_cams,
            "ignored_cameras":         ignored_cams,
            "per_camera_concentration": per_camera_top_k,
            "grid_side":               self.grid_side,
            "n_grid_cells_per_camera": self.grid_side * self.grid_side,
        }

        if not consumed_cams:
            sev = Severity.UNKNOWN
            finding = Finding(
                observed=(
                    f"We covered a {self.grid_side}×{self.grid_side} grid "
                    f"of patches across {cameras} and the model's action "
                    f"barely changed for any patch — every cell's "
                    f"response was below the model's per-call sampling "
                    f"noise floor."
                ),
                meaning=(
                    "We cannot tell on this frame whether the model is "
                    "ignoring all visual input, or whether the response "
                    "is just lost in its sampling jitter. This is common "
                    "on quiescent frames where the action is "
                    "well-determined regardless of the image."
                ),
                next_step=(
                    "Pick a more dynamic mid-trajectory frame, or "
                    "increase averaging (model n_samples) to tighten "
                    "the noise floor."
                ),
                raw_numbers=raw_numbers,
            )
        elif scalar > 0.5:
            sev = Severity.PASS
            finding = Finding(
                observed=(
                    f"Across cameras {consumed_cams}, the top "
                    f"{self.grid_side} of {self.grid_side*self.grid_side} "
                    f"image patches capture {scalar:.0%} of the model's "
                    f"sensitivity. Each camera has a clear handful of "
                    f"hot regions."
                ),
                meaning=(
                    "Vision is focused on specific regions — the kind "
                    "of behaviour you want from a grounded policy."
                ),
                next_step=(
                    "Check the Rerun sensitivity heatmap overlay to "
                    "confirm those hot regions sit on task-relevant "
                    "objects (target, gripper, distractors)."
                ),
                raw_numbers=raw_numbers,
            )
        elif scalar > 0.25:
            sev = Severity.INFO
            finding = Finding(
                observed=(
                    f"Across cameras {consumed_cams}, the top "
                    f"{self.grid_side} patches per camera capture "
                    f"{scalar:.0%} of sensitivity — visible regions of "
                    f"focus but with a long tail of weakly-influential "
                    f"patches."
                ),
                meaning=(
                    "The policy uses several distinct regions per "
                    "camera. Not a problem by itself, just diffuse."
                ),
                next_step=(
                    "Confirm via Rerun overlay that the top regions are "
                    "the task-relevant ones."
                ),
                raw_numbers=raw_numbers,
            )
        else:
            sev = Severity.MODERATE
            finding = Finding(
                observed=(
                    f"Sensitivity across cameras {consumed_cams} is "
                    f"diffuse — top {self.grid_side} patches capture "
                    f"only {scalar:.0%} of the response. Many small "
                    f"patches contribute roughly equally."
                ),
                meaning=(
                    "The model may be relying on background statistics "
                    "(global colour, scene gist) rather than on the "
                    "task-relevant object. This often correlates with "
                    "out-of-distribution failures."
                ),
                next_step=(
                    "Use the Rerun heatmap overlay to see WHERE the "
                    "sensitivity is going. If it's spread over irrelevant "
                    "regions, your policy's vision is not well grounded."
                ),
                raw_numbers=raw_numbers,
            )
        if ignored_cams:
            finding = Finding(
                observed=finding.observed + (
                    f" Cameras with no measurable response: {ignored_cams}."
                ),
                meaning=finding.meaning + (
                    f" The unresponsive cameras {ignored_cams} are not "
                    f"being consumed at all on this frame."
                ),
                next_step=finding.next_step,
                raw_numbers=finding.raw_numbers,
            )
        verdict = f"{finding.observed} {finding.meaning} {finding.next_step}"

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=scalar,
            severity=sev,
            # scalar = top-K concentration. HIGHER concentration = MORE
            # focused = BETTER (model uses focused regions). LOWER
            # concentration = MORE diffuse = WORSE (model relies on
            # background cues). So lower_is_worse.
            direction="lower_is_worse",
            explanation=verdict,
            finding=finding,
            per_variant={
                **{f"cam:{cam}:concentration": v for cam, v in per_camera_top_k.items()},
                **{f"cam:{cam}:consumed": float(per_camera_consumed[cam]) for cam in cameras},
            },
            raw={
                "sensitivity_grid_per_camera": {
                    cam: g.tolist() for cam, g in per_camera_grid.items()
                },
                "top_k_concentration_per_camera": per_camera_top_k,
                "per_camera_consumed":            per_camera_consumed,
                "consumed_cameras":               consumed_cams,
                "ignored_cameras":                ignored_cams,
                "image_shape_per_camera": {
                    cam: list(shape) for cam, shape in per_camera_image_shape.items()
                },
                "grid_side":          self.grid_side,
                "cameras_evaluated":  cameras,
            },
        )
