"""Diagnostic ABC + shared utilities.

A Diagnostic is the orchestration layer: given a model and a scene, run
its specific algorithm (which usually composes a Perturber + a Metric)
and emit a DiagnosticResult.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from policylens.core.results import DiagnosticResult
from policylens.core.types import Scene
from policylens.models.protocol import Capability, NotSupported, VLAModel


class Diagnostic(ABC):
    """Base class for all diagnostics."""

    name: str
    axis: str
    required_capabilities: Capability = Capability.INFERENCE

    def applicable_to(self, model: VLAModel) -> bool:
        """Check whether this diagnostic's capability requirements are met."""
        return (model.capabilities & self.required_capabilities) == self.required_capabilities

    @abstractmethod
    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult: ...

    # Helper for diagnostics to produce a consistent NotApplicable result.
    def _not_applicable(self, model: VLAModel, scene: Scene, reason: str) -> DiagnosticResult:
        from policylens.core.results import Severity
        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=float("nan"),
            severity=Severity.UNKNOWN,
            direction="lower_is_worse",
            explanation=f"Diagnostic skipped: {reason}",
        )
