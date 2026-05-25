"""BYOVLA-style per-region sensitivity map.

Mask each patch of an N×N grid one-at-a-time, measure how much the action
changes. The resulting heatmap shows which scene regions causally drive
the policy. Useful for: distinguishing "model focuses on target" from
"model focuses on background or arm position".
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Scene
from emboviz.diagnostics.base import Diagnostic
from emboviz.metrics.action_divergence import ActionDivergenceMetric
from emboviz.models.protocol import Capability, VLAModel
from emboviz.perturb.image._image_utils import to_array, to_pil


class SensitivityMapDiagnostic(Diagnostic):
    """For each grid cell, mask it out and measure |Δaction|."""

    required_capabilities = Capability.INFERENCE

    def __init__(
        self,
        grid_side: int = 8,
        metric: Optional[ActionDivergenceMetric] = None,
    ):
        self.grid_side = grid_side
        self._metric_override = metric
        self.name = f"sensitivity_map_{grid_side}x{grid_side}"
        self.axis = "vision.scene_sensitivity"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks INFERENCE capability")

        metric = self._metric_override or ActionDivergenceMetric(model=model)
        baseline = model.predict(scene)
        arr = to_array(scene.primary_image_data)
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
                pert = model.predict(scene.with_image(to_pil(masked)))
                drops[gi, gj] = metric.compute(baseline, pert)

        # Scalar: concentration of sensitivity — top-K cells / total
        # (high concentration = model relies on a small region).
        flat = drops.flatten()
        top_k_frac = float(np.sort(flat)[-self.grid_side:].sum() / max(flat.sum(), 1e-9))

        if top_k_frac > 0.5:
            sev = Severity.PASS
            verdict = (
                f"Sensitivity concentrated in {self.grid_side} top cells "
                f"({top_k_frac:.1%} of total) — model uses a focused region."
            )
        elif top_k_frac > 0.25:
            sev = Severity.INFO
            verdict = (
                f"Sensitivity moderately distributed ({top_k_frac:.1%} in top {self.grid_side}); "
                f"model uses several regions."
            )
        else:
            sev = Severity.MODERATE
            verdict = (
                f"Sensitivity diffuse ({top_k_frac:.1%} in top {self.grid_side}); "
                f"model may be relying on background / distractor cues."
            )

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=top_k_frac,
            severity=sev,
            direction="higher_is_worse",   # diffuse sensitivity = worse
            explanation=verdict,
            per_variant={},
            raw={
                "sensitivity_grid": drops.tolist(),
                "grid_side": self.grid_side,
                "image_shape": (H, W),
            },
        )
