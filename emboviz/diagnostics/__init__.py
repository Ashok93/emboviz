"""Diagnostics — orchestrate Perturbers + Metrics + Model into DiagnosticResults."""

from emboviz.diagnostics.activation_patching import ActivationPatchingDiagnostic
from emboviz.diagnostics.attention import CrossModalAttentionDiagnostic
from emboviz.diagnostics.base import Diagnostic
from emboviz.diagnostics.concept_decomp import (
    ConceptDecompositionDiagnostic,
    find_anomalous_neurons,
)
from emboviz.diagnostics.counterfactual import CounterfactualDiagnostic
from emboviz.diagnostics.failure_prediction import FailurePredictionDiagnostic
from emboviz.diagnostics.memorization import MemorizationDiagnostic
from emboviz.diagnostics.probe import ProbeDiagnostic, ProbeVsActionDiagnostic
from emboviz.diagnostics.sensitivity_map import SensitivityMapDiagnostic
from emboviz.diagnostics.sweep import SweepDiagnostic
from emboviz.diagnostics.trajectory import (
    TrajectoryDiagnostic,
    TrajectoryDiagnosticResult,
)

__all__ = [
    "Diagnostic",
    "ActivationPatchingDiagnostic",
    "CounterfactualDiagnostic",
    "SweepDiagnostic",
    "CrossModalAttentionDiagnostic",
    "ConceptDecompositionDiagnostic",
    "find_anomalous_neurons",
    "FailurePredictionDiagnostic",
    "MemorizationDiagnostic",
    "ProbeDiagnostic",
    "ProbeVsActionDiagnostic",
    "SensitivityMapDiagnostic",
    "TrajectoryDiagnostic",
    "TrajectoryDiagnosticResult",
]
