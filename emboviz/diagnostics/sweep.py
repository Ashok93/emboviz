"""Sweep diagnostic — for parametric perturbations like occlusion %.

Same as CounterfactualDiagnostic but emphasises the *curve* over the
discrete level parameter (10/25/50/75 % coverage, 1/3/5 distractors, etc.).
The scalar score is the area under the divergence-vs-level curve.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Scene
from emboviz.diagnostics.base import Diagnostic
from emboviz.metrics.action_divergence import ActionDivergenceMetric
from emboviz.models.protocol import Capability, VLAModel
from emboviz.perturb.base import Perturber


class SweepDiagnostic(Diagnostic):
    """Run a perturber whose variants form a parametric sweep."""

    required_capabilities = Capability.INFERENCE

    def __init__(
        self,
        perturber: Perturber,
        level_param_key: str,
        metric: Optional[ActionDivergenceMetric] = None,
        # Thresholds calibrated for normalized-L2 action distance
        # (mean divergence ≥ 1.5 ≈ clear sensitivity along this axis).
        critical_auc: float = 1.5,
        moderate_auc: float = 0.5,
    ):
        self.perturber = perturber
        self.level_param_key = level_param_key
        self._metric_override = metric
        self.critical_auc = critical_auc
        self.moderate_auc = moderate_auc
        self.name = f"sweep.{perturber.name}"
        self.axis = perturber.axis

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks INFERENCE capability")

        metric = self._metric_override or ActionDivergenceMetric(model=model)
        baseline = model.predict(scene.image, scene.instruction)

        levels: list[float] = []
        divergences: list[float] = []
        records: list[dict] = []

        for variant in self.perturber.variants(scene):
            level = float(variant.parameters.get(self.level_param_key, 0.0))
            perturbed = model.predict_with_image(
                variant.scene.image, variant.scene.instruction,
            )
            d = metric.compute(baseline, perturbed)
            levels.append(level)
            divergences.append(d)
            records.append({
                "variant_id": variant.variant_id,
                "level": level,
                "divergence": d,
                "parameters": variant.parameters,
            })

        if not levels:
            return self._not_applicable(model, scene,
                f"{self.perturber.name} produced no variants")

        order = np.argsort(levels)
        lvl = np.array(levels)[order]
        div = np.array(divergences)[order]
        # AUC vs level — normalized by the level range so different
        # perturbers are comparable.
        if lvl[-1] > lvl[0]:
            auc = float(np.trapz(div, lvl) / (lvl[-1] - lvl[0]))
        else:
            auc = float(div.mean())

        if auc >= self.critical_auc:
            sev = Severity.CRITICAL
            verdict = f"AUC over sweep = {auc:.3f} ≥ critical ({self.critical_auc}); strong sensitivity along {self.perturber.axis}."
        elif auc >= self.moderate_auc:
            sev = Severity.MODERATE
            verdict = f"AUC over sweep = {auc:.3f}; moderate sensitivity along {self.perturber.axis}."
        else:
            sev = Severity.PASS
            verdict = f"AUC over sweep = {auc:.3f}; robust along {self.perturber.axis}."

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=auc,
            severity=sev,
            direction="higher_is_worse",
            explanation=verdict,
            per_variant={r["variant_id"]: r["divergence"] for r in records},
            raw={
                "levels": lvl.tolist(),
                "divergences": div.tolist(),
                "records": records,
            },
        )
