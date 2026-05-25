"""Counterfactual diagnostic — the workhorse.

Pattern: get baseline prediction → run perturber → for each variant get
prediction → measure divergence → aggregate into one DiagnosticResult.

Used by every counterfactual perturber (noun_swap, color_swap, occlusion,
viewpoint, distractor, lighting, ...). Same code, different perturber.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from policylens.core.results import DiagnosticResult, Severity
from policylens.core.types import Scene
from policylens.diagnostics.base import Diagnostic
from policylens.metrics.action_divergence import ActionDivergenceMetric
from policylens.models.protocol import Capability, VLAModel
from policylens.perturb.base import Perturber


# Default severity thresholds (in normalized action-space units).
# Calibrated for normalized_l2 distance — each action dim is divided by its
# (q99 − q01) range from the model's adapter, so "1 unit" ≈ "1 typical
# action range." Adapters can override via the `noise_floor` / `grounded`
# kwargs if their action space is exotic.
#
# Empirical calibration on OpenVLA + BridgeV2 episode 0:
#   noun_swap (spoon→fork) ≈ 0.70   (model partially uses language)
#   empty instruction       ≈ 0.94   (small change from baseline)
#   occlusion 50%           ≈ 1.39   (clearly sensitive)
#   OOD task                ≈ 2.50+  (clearly different action)
DEFAULT_NOISE_FLOOR = 0.5
DEFAULT_GROUNDED = 2.0


class CounterfactualDiagnostic(Diagnostic):
    """Perturb the scene N ways; measure how much the action moves."""

    required_capabilities = Capability.INFERENCE

    def __init__(
        self,
        perturber: Perturber,
        noise_floor: float = DEFAULT_NOISE_FLOOR,
        grounded_threshold: float = DEFAULT_GROUNDED,
        # Optional metric override; default uses model.compare_actions
        metric: Optional[ActionDivergenceMetric] = None,
    ):
        self.perturber = perturber
        self.noise_floor = noise_floor
        self.grounded_threshold = grounded_threshold
        self._metric_override = metric

        # Derive name + axis from the perturber for uniform reporting.
        self.name = f"counterfactual.{perturber.name}"
        self.axis = perturber.axis

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks INFERENCE capability")

        metric = self._metric_override or ActionDivergenceMetric(model=model)
        baseline = model.predict(scene.image, scene.instruction)

        per_variant: dict[str, float] = {}
        variant_records: list[dict] = []

        for variant in self.perturber.variants(scene):
            v_scene = variant.scene
            perturbed = (
                model.predict_with_image(v_scene.image, v_scene.instruction)
                if self.perturber.domain == "image"
                else model.predict(v_scene.image, v_scene.instruction)
            )
            d = metric.compute(baseline, perturbed)
            per_variant[variant.variant_id] = d
            variant_records.append({
                "variant_id": variant.variant_id,
                "axis": variant.axis,
                "description": variant.description,
                "instruction": v_scene.instruction,
                "parameters": variant.parameters,
                "divergence": d,
                "baseline_action": baseline.action.tolist(),
                "perturbed_action": perturbed.action.tolist(),
            })

        if not per_variant:
            return self._not_applicable(model, scene,
                f"{self.perturber.name} produced no variants for this scene")

        scores = np.array(list(per_variant.values()))
        mean = float(scores.mean())

        if mean < self.noise_floor:
            sev = Severity.CRITICAL
            verdict = (
                f"Action divergence under {self.perturber.name} averages {mean:.3f}, "
                f"below the noise floor ({self.noise_floor}). The model produces nearly "
                f"identical actions across variants — it isn't using the {self.perturber.axis} cue."
            )
        elif mean < self.grounded_threshold:
            sev = Severity.MODERATE
            verdict = (
                f"Mean divergence {mean:.3f} is between noise ({self.noise_floor}) "
                f"and grounded ({self.grounded_threshold}). Partial sensitivity to "
                f"{self.perturber.axis}."
            )
        else:
            sev = Severity.PASS
            verdict = (
                f"Mean divergence {mean:.3f} ≥ grounded threshold ({self.grounded_threshold}). "
                f"The model's action tracks {self.perturber.axis}."
            )

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=mean,
            severity=sev,
            direction="lower_is_worse",
            explanation=verdict,
            per_variant=per_variant,
            raw={
                "variants": variant_records,
                "baseline_instruction": scene.instruction,
                "noise_floor": self.noise_floor,
                "grounded_threshold": self.grounded_threshold,
                "perturber_axis": self.perturber.axis,
                "perturber_domain": self.perturber.domain,
            },
        )
