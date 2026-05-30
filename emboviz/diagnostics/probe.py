"""Probe-based diagnostics — 'information present but unused.'

Two flavours:

  • `ProbeDiagnostic` — run a single LinearProbe on the scene's hidden states
    and report its prediction + confidence. By itself this is just an
    inspection tool.

  • `ProbeVsActionDiagnostic` — the *interesting* one. Pair a trained probe
    with a counterfactual perturbation (e.g., color swap in instruction).
    If the probe ALREADY decodes the target attribute with high confidence
    from hidden states (model 'sees' the color) BUT the action is invariant
    to perturbing that attribute → **information present but unused.**
    The most damning mechanistic claim available.
"""

from __future__ import annotations


import numpy as np

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Scene, TokenSelector
from emboviz.diagnostics.base import Diagnostic
from emboviz.models.protocol import Capability, NotSupported, VLAModel
from emboviz.perturb.base import Perturber
from emboviz.probes.base import LinearProbe


class ProbeDiagnostic(Diagnostic):
    """Run one probe on one scene; report decoded label + confidence."""

    required_capabilities = Capability.INFERENCE | Capability.HIDDEN_STATES

    def __init__(self, probe: LinearProbe, confidence_threshold: float = 0.7):
        self.probe = probe
        self.confidence_threshold = confidence_threshold
        self.name = f"probe.{probe.spec.name}"
        self.axis = f"internal.probe.{probe.spec.name}"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks HIDDEN_STATES capability")
        if self.probe.spec.model_id != model.model_id:
            return self._not_applicable(
                model, scene,
                f"probe was trained for {self.probe.spec.model_id}, got {model.model_id}",
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
        label, conf = self.probe.predict(feats)
        probs = self.probe.predict_proba(feats)

        sev = (
            Severity.INFO if conf >= self.confidence_threshold else
            Severity.UNKNOWN
        )
        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=float(conf),
            severity=sev,
            direction="higher_is_worse",        # high probe confidence = model has the info
            explanation=(
                f"Probe '{self.probe.spec.name}' predicts {label!r} with "
                f"confidence {conf:.2f}. (Decoder validation accuracy = "
                f"{self.probe.spec.val_accuracy:.2f}.)"
            ),
            per_variant={
                cls: float(p) for cls, p in zip(self.probe.spec.classes, probs)
            },
            raw={
                "label": label,
                "confidence": conf,
                "per_class": [float(p) for p in probs],
                "probe_spec": self.probe.spec.name,
                "probe_val_acc": self.probe.spec.val_accuracy,
            },
        )


class ProbeVsActionDiagnostic(Diagnostic):
    """The 'information present but unused' diagnostic.

    Combines a Probe + a Perturber. Asks:
      • Does the probe's *answer* change when we perturb? (it should, if
        the perturbation actually changed the target attribute the probe
        decodes)
      • Does the *action* change when we perturb? (proportionally — if not,
        the action head is ignoring the info the probe sees)

    Scalar score = decoder_certainty − action_sensitivity. Positive ⇒ the
    model 'knows X but doesn't use X' — exactly the failure we want to
    surface.
    """

    required_capabilities = (
        Capability.INFERENCE | Capability.HIDDEN_STATES
    )

    def __init__(
        self,
        probe: LinearProbe,
        perturber: Perturber,
        action_sensitivity_floor: float = 0.5,
    ):
        self.probe = probe
        self.perturber = perturber
        self.floor = action_sensitivity_floor
        self.name = f"probe_vs_action.{probe.spec.name}.{perturber.name}"
        self.axis = f"internal.probe_vs_action.{probe.spec.name}"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "missing capabilities")
        if self.probe.spec.model_id != model.model_id:
            return self._not_applicable(
                model, scene,
                f"probe trained for {self.probe.spec.model_id}, got {model.model_id}",
            )

        # 1) Probe certainty at baseline.
        try:
            hs_base = model.extract_hidden_states(
                scene,
                self.probe.spec.layer_indices,
                TokenSelector(relative="before_action"),
            )
        except NotSupported as e:
            return self._not_applicable(model, scene, str(e))
        probs_base = self.probe.predict_proba(self.probe.features_from_hidden_states(hs_base.states))
        base_label = self.probe.spec.classes[int(np.argmax(probs_base))]
        base_certainty = float(np.max(probs_base))

        # 2) Per perturbation variant: action change AND probe-label change.
        action_changes: list[float] = []
        label_changes: list[int] = []
        baseline_action = model.predict(scene)
        for variant in self.perturber.variants(scene):
            v_scene = variant.scene
            pert_action = model.predict(v_scene)
            action_changes.append(float(model.compare_actions(baseline_action, pert_action)))
            try:
                hs_v = model.extract_hidden_states(
                    v_scene,
                    self.probe.spec.layer_indices,
                    TokenSelector(relative="before_action"),
                )
                probs_v = self.probe.predict_proba(self.probe.features_from_hidden_states(hs_v.states))
                pert_label = self.probe.spec.classes[int(np.argmax(probs_v))]
                label_changes.append(1 if pert_label != base_label else 0)
            except NotSupported:
                label_changes.append(0)

        if not action_changes:
            return self._not_applicable(model, scene, "perturber produced no variants")

        mean_action_change = float(np.mean(action_changes))
        label_change_rate = float(np.mean(label_changes))

        # Heuristic: model 'knows but doesn't act' if probe has high
        # baseline certainty AND its label changes under perturbation, yet
        # the action does not.
        knows = base_certainty >= 0.7
        information_changes = label_change_rate >= 0.5
        action_ignores = mean_action_change < self.floor

        if knows and information_changes and action_ignores:
            sev = Severity.CRITICAL
            verdict = (
                f"INFORMATION PRESENT BUT UNUSED. Probe '{self.probe.spec.name}' "
                f"decodes '{base_label}' with {base_certainty:.2f} confidence and "
                f"changes its prediction under perturbation ({label_change_rate:.0%} of variants), "
                f"yet the action changes by only {mean_action_change:.3f} "
                f"(< floor {self.floor}). The model sees this attribute but its "
                f"action head ignores it."
            )
        elif knows and not action_ignores:
            sev = Severity.PASS
            verdict = (
                f"Probe decodes '{base_label}' with {base_certainty:.2f} confidence "
                f"and action responds proportionally to perturbation "
                f"(Δaction={mean_action_change:.3f}). Healthy."
            )
        else:
            sev = Severity.INFO
            verdict = (
                f"Probe baseline certainty {base_certainty:.2f}, label-change rate "
                f"{label_change_rate:.0%}, mean action change {mean_action_change:.3f}. "
                f"Information state is inconclusive."
            )

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=base_certainty - mean_action_change,
            severity=sev,
            direction="higher_is_worse",        # positive = knows-but-doesn't-act
            explanation=verdict,
            per_variant={
                "base_certainty": base_certainty,
                "label_change_rate": label_change_rate,
                "mean_action_change": mean_action_change,
            },
            raw={
                "base_label": base_label,
                "base_probs": [float(p) for p in probs_base],
                "action_changes": action_changes,
                "label_changes": label_changes,
            },
        )
