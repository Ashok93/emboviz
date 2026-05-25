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
            top_k = float(np.sort(flat)[-self.grid_side:].sum() / max(flat.sum(), 1e-9))
            per_camera_top_k[cam] = top_k

        # Aggregate: the worst (most-diffuse) camera drives the verdict —
        # a model that uses one camera well but ignores another should
        # still raise a flag here.
        worst_cam = min(per_camera_top_k, key=per_camera_top_k.get)
        scalar = float(per_camera_top_k[worst_cam])

        if scalar > 0.5:
            sev = Severity.PASS
            verdict = (
                f"All cameras' sensitivity is concentrated (worst={worst_cam} "
                f"at {scalar:.1%}). Model uses focused regions per camera."
            )
        elif scalar > 0.25:
            sev = Severity.INFO
            verdict = (
                f"Sensitivity moderately distributed; worst camera "
                f"{worst_cam}={scalar:.1%}. Model uses several regions on "
                f"at least one camera."
            )
        else:
            sev = Severity.MODERATE
            verdict = (
                f"Sensitivity diffuse on camera {worst_cam} ({scalar:.1%}); "
                f"model may be relying on background / distractor cues there, "
                f"or ignoring that camera entirely."
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
            per_variant={f"cam:{cam}": v for cam, v in per_camera_top_k.items()},
            raw={
                "sensitivity_grid_per_camera": {
                    cam: g.tolist() for cam, g in per_camera_grid.items()
                },
                "top_k_concentration_per_camera": per_camera_top_k,
                "image_shape_per_camera": {
                    cam: list(shape) for cam, shape in per_camera_image_shape.items()
                },
                "grid_side": self.grid_side,
                "cameras_evaluated": cameras,
                "worst_camera": worst_cam,
            },
        )
