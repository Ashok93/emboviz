"""Vision-axis diagnostics — occlusion, viewpoint, lighting, distractor,
target-removal, sensitivity-map."""

from __future__ import annotations

from emboviz.diagnostics.counterfactual import CounterfactualDiagnostic
from emboviz.diagnostics.memorization import MemorizationDiagnostic
from emboviz.diagnostics.sensitivity_map import SensitivityMapDiagnostic
from emboviz.diagnostics.sweep import SweepDiagnostic
from emboviz.perturb.image import (
    DistractorInjectionPerturber,
    GaussianNoisePerturber,
    LightingShiftPerturber,
    OcclusionPerturber,
    ViewpointJitterPerturber,
)
from emboviz.suites.base import Suite


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
