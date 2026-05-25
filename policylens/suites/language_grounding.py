"""All language-axis diagnostics in one preset."""

from __future__ import annotations

from policylens.diagnostics.counterfactual import CounterfactualDiagnostic
from policylens.perturb.instruction import (
    ColorSwapPerturber,
    CountSwapPerturber,
    EmptyInstructionPerturber,
    NegationPerturber,
    NounSwapPerturber,
    OODTaskPerturber,
    PrepositionSwapPerturber,
    RefusalPerturber,
)
from policylens.suites.base import Suite


def build_language_grounding_suite() -> Suite:
    return Suite(
        name="language_grounding",
        description=(
            "Eight counterfactual diagnostics that test whether the model is using "
            "the instruction at all and, if so, which linguistic features it uses."
        ),
        diagnostics=[
            CounterfactualDiagnostic(NounSwapPerturber()),
            CounterfactualDiagnostic(PrepositionSwapPerturber()),
            CounterfactualDiagnostic(ColorSwapPerturber()),
            CounterfactualDiagnostic(CountSwapPerturber()),
            CounterfactualDiagnostic(NegationPerturber()),
            CounterfactualDiagnostic(RefusalPerturber()),
            CounterfactualDiagnostic(EmptyInstructionPerturber()),
            CounterfactualDiagnostic(OODTaskPerturber()),
        ],
    )
