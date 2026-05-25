"""The uniform result schema every diagnostic returns.

Reporters and dashboards consume this — never raw diagnostic-specific data.
Diagnostic-specific data goes into `raw` and is opaque to general tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional


class Severity(str, Enum):
    PASS = "pass"          # behaviour as expected
    INFO = "info"          # noteworthy but not failing
    MODERATE = "moderate"  # concerning, worth investigating
    CRITICAL = "critical"  # failure mode confirmed
    UNKNOWN = "unknown"    # diagnostic could not run conclusively


@dataclass
class DiagnosticResult:
    """The contract every diagnostic returns.

    Reporters look at the top-level fields. `raw` is for diagnostic-specific
    visualizations (per-frame heatmaps, attention tensors, per-variant
    arrows, etc.) — diagnostic-specific visualizers know its shape.
    """

    diagnostic_name: str            # e.g., "counterfactual_noun_swap"
    axis: str                       # e.g., "language.noun_swap"
    model_id: str                   # which VLA was tested
    scene_id: str                   # which Scene
    scalar_score: float             # headline number
    severity: Severity
    direction: Literal["lower_is_worse", "higher_is_worse"]
    explanation: str                # one-paragraph verdict in plain English
    recommendation: Optional[str] = None
    per_variant: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> dict:
        """Lightweight JSON-safe view, skipping `raw`."""
        return {
            "diagnostic_name": self.diagnostic_name,
            "axis": self.axis,
            "model_id": self.model_id,
            "scene_id": self.scene_id,
            "scalar_score": self.scalar_score,
            "severity": self.severity.value,
            "direction": self.direction,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "per_variant": self.per_variant,
            "metadata": self.metadata,
        }
