"""Diagnostics — orchestrate Perturbers + Metrics + Model into DiagnosticResults."""

from policylens.diagnostics.activation_patching import ActivationPatchingDiagnostic
from policylens.diagnostics.attention import CrossModalAttentionDiagnostic
from policylens.diagnostics.base import Diagnostic
from policylens.diagnostics.concept_decomp import (
    ConceptDecompositionDiagnostic,
    find_anomalous_neurons,
)
from policylens.diagnostics.counterfactual import CounterfactualDiagnostic
from policylens.diagnostics.failure_prediction import FailurePredictionDiagnostic
from policylens.diagnostics.memorization import MemorizationDiagnostic
from policylens.diagnostics.probe import ProbeDiagnostic, ProbeVsActionDiagnostic
from policylens.diagnostics.sensitivity_map import SensitivityMapDiagnostic
from policylens.diagnostics.sweep import SweepDiagnostic
from policylens.diagnostics.trajectory import (
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
