"""Minimal smoke-test suite — runs in seconds, used for CI / sanity."""

from __future__ import annotations

from emboviz.diagnostics.counterfactual import CounterfactualDiagnostic
from emboviz.perturb.instruction import EmptyInstructionPerturber, NounSwapPerturber
from emboviz.perturb.image import OcclusionPerturber
from emboviz.suites.base import Suite


def build_quick_smoke() -> Suite:
    return Suite(
        name="quick_smoke",
        description="Three fastest diagnostics — sanity check the engine + adapter.",
        diagnostics=[
            CounterfactualDiagnostic(NounSwapPerturber(max_swaps=1)),
            CounterfactualDiagnostic(EmptyInstructionPerturber()),
            CounterfactualDiagnostic(OcclusionPerturber(coverages=[0.5])),
        ],
    )
