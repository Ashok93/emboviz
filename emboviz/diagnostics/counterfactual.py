"""Counterfactual diagnostic — the workhorse.

Pattern: get baseline prediction → run perturber → for each variant get
prediction → measure divergence → aggregate into one DiagnosticResult.

Used by every counterfactual perturber (noun_swap, color_swap, occlusion,
viewpoint, distractor, lighting, ...). Same code, different perturber.

Input-side gating: if the perturber mutates an input modality the model
doesn't consume (e.g., gripper_flip against a model that doesn't read
gripper), the diagnostic returns Severity.UNKNOWN with a clear reason
instead of running a meaningless test.
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
        # Capability check
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks INFERENCE capability")

        # Input-side gating: skip cleanly if the model doesn't consume what
        # this perturber mutates.
        if self.perturber.affects and not any(
            model.required_inputs.consumes(a) for a in self.perturber.affects
        ):
            affects_list = ", ".join(sorted(self.perturber.affects))
            return self._not_applicable(
                model, scene,
                f"model {model.model_id} does not consume input modalities "
                f"affected by {self.perturber.name} (perturber affects: {affects_list})",
            )

        # Scene-side validation: does the scene actually have what the model needs?
        reason = model.required_inputs.validate(scene)
        if reason:
            return self._not_applicable(
                model, scene,
                f"scene does not satisfy model.required_inputs: {reason}",
            )

        metric = self._metric_override or ActionDivergenceMetric(model=model)
        baseline = model.predict(scene)

        per_variant: dict[str, float] = {}
        variant_records: list[dict] = []

        for variant in self.perturber.variants(scene):
            v_scene = variant.scene
            perturbed = model.predict(v_scene)
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
            return self._not_applicable(
                model, scene,
                f"{self.perturber.name} produced no variants for this scene",
            )

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
                "perturber_affects": sorted(self.perturber.affects),
            },
        )
