"""Failure-prediction diagnostic — the commercial wedge.

Loads a pre-trained failure-prediction probe and scores frames by
P(failure). When wrapped with `TrajectoryDiagnostic`, you get a per-frame
P(failure) curve over a rollout — the 'Emboviz warned you at frame 47'
output that doesn't require interpretability knowledge to use.

Reference: SAFE / FIPER family of works on internal-feature failure
detection for VLAs.
"""

from __future__ import annotations

from pathlib import Path

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Scene, TokenSelector
from emboviz.diagnostics.base import Diagnostic
from emboviz.models.protocol import Capability, NotSupported, VLAModel
from emboviz.probes.base import LinearProbe
from emboviz.probes.store import load_probe


class FailurePredictionDiagnostic(Diagnostic):
    """Score each scene with a trained 'is this frame failing' probe."""

    required_capabilities = Capability.INFERENCE | Capability.HIDDEN_STATES

    def __init__(
        self,
        probe: LinearProbe,
        warn_threshold: float = 0.5,
        critical_threshold: float = 0.8,
    ):
        if "failure" not in probe.spec.classes:
            raise ValueError(
                f"Probe must include 'failure' in its classes; got {probe.spec.classes}"
            )
        self.probe = probe
        self.failure_class_idx = probe.spec.classes.index("failure")
        self.warn_threshold = warn_threshold
        self.critical_threshold = critical_threshold
        self.name = "failure_prediction"
        self.axis = "internal.failure_prediction"

    @classmethod
    def from_path(cls, path: Path, **kw) -> "FailurePredictionDiagnostic":
        return cls(load_probe(path), **kw)

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks HIDDEN_STATES")
        if self.probe.spec.model_id != model.model_id:
            return self._not_applicable(
                model, scene,
                f"probe trained for {self.probe.spec.model_id}, got {model.model_id}",
            )
        try:
            hs = model.extract_hidden_states(
                scene,
                self.probe.spec.layer_indices,
                TokenSelector(relative="before_action"),
            )
        except NotSupported as e:
            return self._not_applicable(model, scene, str(e))

        feats = self.probe.features_from_hidden_states(hs.states)
        probs = self.probe.predict_proba(feats)
        p_fail = float(probs[self.failure_class_idx])

        if p_fail >= self.critical_threshold:
            sev = Severity.CRITICAL
            verdict = (
                f"P(failure) = {p_fail:.2f} — above critical threshold {self.critical_threshold}. "
                f"Emboviz flags this frame as high-risk for failure."
            )
        elif p_fail >= self.warn_threshold:
            sev = Severity.MODERATE
            verdict = (
                f"P(failure) = {p_fail:.2f} — above warning threshold {self.warn_threshold}. "
                f"Investigate the rollout around this frame."
            )
        else:
            sev = Severity.PASS
            verdict = f"P(failure) = {p_fail:.2f} — within normal range."

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=p_fail,
            severity=sev,
            direction="higher_is_worse",
            explanation=verdict,
            per_variant={
                cls: float(p) for cls, p in zip(self.probe.spec.classes, probs)
            },
            raw={
                "probe_name": self.probe.spec.name,
                "probe_val_accuracy": self.probe.spec.val_accuracy,
                "p_failure": p_fail,
                "per_class_probs": [float(p) for p in probs],
            },
        )
