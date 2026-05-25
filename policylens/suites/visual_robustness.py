"""Vision-axis diagnostics — occlusion, viewpoint, lighting, distractor,
target-removal, sensitivity-map."""

from __future__ import annotations

from policylens.diagnostics.counterfactual import CounterfactualDiagnostic
from policylens.diagnostics.memorization import MemorizationDiagnostic
from policylens.diagnostics.sensitivity_map import SensitivityMapDiagnostic
from policylens.diagnostics.sweep import SweepDiagnostic
from policylens.perturb.image import (
    DistractorInjectionPerturber,
    GaussianNoisePerturber,
    LightingShiftPerturber,
    OcclusionPerturber,
    ViewpointJitterPerturber,
)
from policylens.suites.base import Suite


def build_visual_robustness_suite() -> Suite:
    return Suite(
        name="visual_robustness",
        description=(
            "Vision-axis diagnostics — occlusion, viewpoint, lighting, distractor, "
            "noise, memorization probe, and per-region sensitivity map."
        ),
        diagnostics=[
            SweepDiagnostic(OcclusionPerturber(), level_param_key="coverage"),
            CounterfactualDiagnostic(ViewpointJitterPerturber()),
            CounterfactualDiagnostic(LightingShiftPerturber()),
            SweepDiagnostic(DistractorInjectionPerturber(), level_param_key="n_distractors"),
            CounterfactualDiagnostic(GaussianNoisePerturber()),
            MemorizationDiagnostic(),
            SensitivityMapDiagnostic(grid_side=8),
        ],
    )
