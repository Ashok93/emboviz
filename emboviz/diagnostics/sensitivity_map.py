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
  • Aggregate ``scalar_score`` = the MINIMUM concentration across cameras
    (lowest concentration = most diffuse = worst signal). Going by the
    worst camera, not the average, surfaces "model uses one camera well
    but ignores another."
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Scene, resolve_cameras
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
    ):
        self.grid_side = grid_side
        self._metric_override = metric
        self.cameras = cameras
        self.name = f"sensitivity_map_{grid_side}x{grid_side}"
        self.axis = "vision.scene_sensitivity"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks INFERENCE capability")

        cameras = resolve_cameras(scene, self.cameras)
        metric = self._metric_override or ActionDivergenceMetric(model=model)
        baseline = model.predict(scene)

        per_camera_grid: dict[str, np.ndarray] = {}
        per_camera_top_k: dict[str, float] = {}
        per_camera_consumed: dict[str, bool] = {}
        per_camera_image_shape: dict[str, tuple[int, int]] = {}

        for cam in cameras:
            arr = to_array(scene.observations.images[cam].data)
            H, W = arr.shape[:2]
            chan_mean = arr.reshape(-1, 3).mean(axis=0)
            ph = H // self.grid_side
            pw = W // self.grid_side
            drops = np.zeros((self.grid_side, self.grid_side), dtype=np.float32)
            for gi in range(self.grid_side):
                for gj in range(self.grid_side):
                    masked = arr.copy()
                    y0, x0 = gi * ph, gj * pw
                    masked[y0:y0 + ph, x0:x0 + pw] = chan_mean
                    pert_scene = scene.with_image(to_pil(masked), camera=cam)
                    pert = model.predict(pert_scene)
                    drops[gi, gj] = metric.compute(baseline, pert)
            per_camera_grid[cam] = drops
            per_camera_image_shape[cam] = (H, W)
            flat = drops.flatten()
            total = float(flat.sum())
            # "Consumed" = some intervention on this camera moved the action.
            # Cameras the model doesn't read produce flat-zero grids — we
            # mark those explicitly rather than reporting a misleading
            # concentration of 0/0.
            if total < 1e-9:
                per_camera_top_k[cam] = 0.0
                per_camera_consumed[cam] = False
            else:
                per_camera_top_k[cam] = float(np.sort(flat)[-self.grid_side:].sum() / total)
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

        if not consumed_cams:
            sev = Severity.UNKNOWN
            verdict = (
                f"Model did not respond to per-cell masking on ANY of "
                f"{cameras}. Either the model genuinely ignores all "
                "visual input on this scene, or the perturbation magnitudes "
                "are below the model's noise floor."
            )
        elif scalar > 0.5:
            sev = Severity.PASS
            verdict = (
                f"Mean concentration across consumed cameras {consumed_cams} "
                f"is {scalar:.1%} (top {self.grid_side} cells per cam capture "
                f">50% of sensitivity). Model uses focused regions."
            )
        elif scalar > 0.25:
            sev = Severity.INFO
            verdict = (
                f"Mean concentration across consumed cameras {consumed_cams} "
                f"is {scalar:.1%}. Sensitivity moderately distributed; "
                f"model uses several regions per camera."
            )
        else:
            sev = Severity.MODERATE
            verdict = (
                f"Mean concentration across consumed cameras {consumed_cams} "
                f"is {scalar:.1%}. Sensitivity diffuse — model may be relying "
                f"on background cues."
            )
        if ignored_cams:
            verdict += (
                f" Cameras with zero sensitivity (model does not consume): "
                f"{ignored_cams}."
            )

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=scalar,
            severity=sev,
            direction="higher_is_worse",   # diffuse sensitivity = worse
            explanation=verdict,
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
