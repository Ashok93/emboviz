"""The uniform result schema every diagnostic returns.

A ``DiagnosticResult`` is the contract every diagnostic produces. It
carries two things the user actually reads:

  • ``finding``  — the structured plain-English verdict (see ``Finding``
                   below). This is what the user sees in CLI output,
                   markdown reports, HTML dashboards.
  • ``raw``      — diagnostic-specific payload (per-frame heatmaps,
                   attention tensors, per-variant numbers). Opaque to
                   general tooling; visualizers that know the diagnostic
                   reach into it.

It ALSO carries an internal sort key:

  • ``severity`` — an enum used by the framework to sort, filter, and
                   route results (e.g. "show me the diagnostics that
                   need attention first"). **Never rendered to user-
                   facing text.** The user does not need to read the
                   word "CRITICAL"; the Finding explains what happened
                   in plain English.

The legacy ``explanation`` string is kept for backward compatibility
during the migration to ``Finding``-everywhere. New diagnostics should
populate ``finding`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional


class Severity(str, Enum):
    """Internal sort key for diagnostic results.

    Never rendered to user-facing text — use ``Finding`` for that.
    The framework uses this to:

      • Sort results worst-first when displaying multiple diagnostics
      • Decide which frames are "failure moments" for find-the-moment
      • Filter aggregate reports to "diagnostics that need attention"
    """

    PASS = "pass"          # behaviour as expected
    INFO = "info"          # noteworthy but not failing
    MODERATE = "moderate"  # concerning, worth investigating
    CRITICAL = "critical"  # failure mode confirmed
    UNKNOWN = "unknown"    # diagnostic could not run conclusively

    @property
    def sort_key(self) -> int:
        """Numeric priority for worst-first ordering (higher = worse)."""
        return {
            Severity.PASS:      0,
            Severity.INFO:      1,
            Severity.UNKNOWN:   2,
            Severity.MODERATE:  3,
            Severity.CRITICAL:  4,
        }[self]


@dataclass(frozen=True)
class Finding:
    """A plain-English verdict for ONE diagnostic on ONE scene/trajectory.

    Three short sentences the user reads, plus the raw numbers for the
    power user who wants to drill in:

      • ``observed``  — what we measured, in present-tense English.
                        "On 7 of 10 frames, the model's predicted action
                         barely changed when we masked the target object."
      • ``meaning``   — what that observation likely implies for the
                        user's policy. "Suggests the policy is replaying
                         memorized motion from training rather than
                         visually grounding the task."
      • ``next_step`` — concrete action the user can take to confirm or
                        rule out the implication. "Compare to an unseen
                         episode — if Δaction grows, the model IS
                         grounded when forced to."
      • ``raw_numbers`` — the actual measurements behind the sentences,
                          for the user who wants the data.

    Hard rules:
      • No severity words ("CRITICAL", "PASS", "MODERATE") in any field.
        The user does not need to read those.
      • No bare numbers without units in the sentences. Either name the
        unit ("normalized Δaction = 0.05") or describe the magnitude
        ("barely changed", "shifted by ~5% of typical action magnitude").
      • ``next_step`` is required — every finding tells the user what
        to do next. Even if it's "this looks healthy; move on."
    """

    observed:    str
    meaning:     str
    next_step:   str
    raw_numbers: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "observed":    self.observed,
            "meaning":     self.meaning,
            "next_step":   self.next_step,
            "raw_numbers": self.raw_numbers,
        }

    def to_markdown(self) -> str:
        """Render as a 3-bullet markdown card."""
        return (
            f"- **Observed**: {self.observed}\n"
            f"- **Meaning**: {self.meaning}\n"
            f"- **Next step**: {self.next_step}\n"
        )


@dataclass
class DiagnosticResult:
    """The contract every diagnostic returns.

    Reporters look at ``finding`` (user-facing) and ``raw`` (diagnostic-
    specific payload for visualizers). ``severity`` is internal sorting
    only; it must never appear in user-rendered text.
    """

    diagnostic_name: str            # e.g., "memorization_test"
    axis: str                       # e.g., "vision.memorization"
    model_id: str                   # which VLA was tested
    scene_id: str                   # which Scene
    scalar_score: float             # headline number (used by aggregators)
    severity: Severity              # internal sort key — DO NOT render to user
    direction: Literal["lower_is_worse", "higher_is_worse"]
    explanation: str = ""           # legacy; new diagnostics populate `finding`
    finding: Optional[Finding] = None  # plain-English verdict (preferred)
    recommendation: Optional[str] = None
    per_variant: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> dict:
        """Lightweight JSON-safe view, skipping ``raw``.

        The Severity enum is included as a stable string (for downstream
        sorting code), but reports SHOULD render Finding, not severity.
        """
        return {
            "diagnostic_name": self.diagnostic_name,
            "axis": self.axis,
            "model_id": self.model_id,
            "scene_id": self.scene_id,
            "scalar_score": self.scalar_score,
            "severity": self.severity.value,   # for internal sort/filter
            "direction": self.direction,
            "finding": self.finding.to_dict() if self.finding is not None else None,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "per_variant": self.per_variant,
            "metadata": self.metadata,
        }
