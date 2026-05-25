"""Modality-dropout diagnostic.

Drop one input modality at a time and measure how much the predicted
action changes. If the action is unchanged when modality X is dropped →
the model doesn't actually use X, regardless of whether it's declared in
``required_inputs``.

Multi-camera contract:
  • Each declared camera is dropped INDIVIDUALLY (``drop_image:<camera>``)
    so you can see "model uses the wrist camera but ignores the head
    camera" as a separate signal per camera.
  • Additionally an ``drop_image:ALL`` variant blanks every declared
    camera together — the "no visual input at all" baseline.
  • Other modalities (state, gripper, action_history, instruction) are
    dropped once each as before.

Severity:
  - PASS: every declared modality (and every declared camera) moves the
          action above ``grounded_threshold`` when dropped.
  - MODERATE: every modality is above noise floor, but at least one is
              only weakly used.
  - CRITICAL: at least one modality (including any specific camera) is
              declared but ignored (Δaction < noise_floor).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from emboviz.calibration import ModelCalibration
from emboviz.core.observations import (
    ActionHistory,
    GripperState,
    Proprioception,
    RGBImage,
)
from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Observations, Scene, resolve_cameras
from emboviz.diagnostics.base import Diagnostic
from emboviz.models.protocol import Capability, VLAModel
from emboviz.perturb.image._image_utils import to_array, to_pil


class ModalityDropoutDiagnostic(Diagnostic):
    """Drop each declared input modality (per camera for images) and measure drift.

    Calibration (recommended):
        When ``calibration`` is passed, every Δaction is normalized into a
        0-1 anchored score so thresholds mean the same thing across models.

    Args:
        noise_floor: normalized score below which the model is treated as
            "ignoring" that modality (memorization-like signature on that input).
        grounded_threshold: normalized score above which the model genuinely
            uses the modality.
        cameras: which cameras to test individually. None = every required
                 camera declared by the model that is also present in the scene.
        calibration: per-model anchors from ``emboviz.calibration.calibrate_model``.
            Without it scores are raw L2 and thresholds become model-specific
            magic numbers.
    """

    required_capabilities = Capability.INFERENCE

    def __init__(
        self,
        noise_floor: float = 0.05,
        grounded_threshold: float = 0.30,
        cameras: Optional[list[str]] = None,
        calibration: Optional[ModelCalibration] = None,
    ):
        self.noise_floor = noise_floor
        self.grounded_threshold = grounded_threshold
        self.cameras = cameras
        self.calibration = calibration
        self.name = "modality_dropout"
        self.axis = "input.modality_dropout"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks INFERENCE capability")

        reqs = model.required_inputs
        baseline = model.predict(scene)

        def _norm(raw_delta: float) -> float:
            return self.calibration.normalize(raw_delta) if self.calibration else raw_delta

        per_modality: dict[str, float] = {}
        per_modality_raw: dict[str, float] = {}

        # 1. Image dropout — one variant per camera the model requires AND
        # the scene actually provides, plus an ALL-cameras combined drop.
        requested_cams = (
            sorted(self.cameras) if self.cameras is not None else sorted(reqs.cameras)
        )
        droppable_cams = [c for c in requested_cams if c in scene.observations.images]
        skipped_cams = [c for c in requested_cams if c not in scene.observations.images]
        def _record(key: str, raw: float):
            per_modality_raw[key] = raw
            per_modality[key] = _norm(raw)

        for cam in droppable_cams:
            arr = to_array(scene.observations.images[cam].data)
            blank = to_pil(np.full_like(arr, fill_value=int(arr.mean())))
            blank_scene = scene.with_image(blank, camera=cam)
            ar = model.predict(blank_scene)
            _record(f"drop_image:{cam}",
                    float(np.linalg.norm(ar.action - baseline.action)))
        if len(droppable_cams) > 1:
            all_blank = {}
            for cam in droppable_cams:
                arr = to_array(scene.observations.images[cam].data)
                all_blank[cam] = to_pil(np.full_like(arr, fill_value=int(arr.mean())))
            ar = model.predict(scene.with_images(all_blank))
            _record("drop_image:ALL",
                    float(np.linalg.norm(ar.action - baseline.action)))

        # 2. State dropout.
        if reqs.state and scene.observations.state is not None:
            state = scene.observations.state
            zeroed_state = Proprioception(
                values=np.zeros_like(state.values), convention=state.convention,
            )
            new_obs = replace(scene.observations, state=zeroed_state)
            ar = model.predict(replace(scene, observations=new_obs))
            _record("state", float(np.linalg.norm(ar.action - baseline.action)))

        # 3. Gripper dropout.
        if reqs.gripper and scene.observations.gripper is not None:
            gripper = scene.observations.gripper
            mid = (
                (scene.profile.gripper.range[0] + scene.profile.gripper.range[1]) / 2
                if scene.profile is not None and scene.profile.gripper is not None
                else 0.5
            )
            new_gripper = replace(gripper, value=float(mid))
            new_obs = replace(scene.observations, gripper=new_gripper)
            ar = model.predict(replace(scene, observations=new_obs))
            _record("gripper", float(np.linalg.norm(ar.action - baseline.action)))

        # 4. Action-history dropout.
        if reqs.action_history and scene.observations.action_history is not None:
            hist = scene.observations.action_history
            zeroed_hist = replace(hist, actions=np.zeros_like(hist.actions))
            new_obs = replace(scene.observations, action_history=zeroed_hist)
            ar = model.predict(replace(scene, observations=new_obs))
            _record("action_history",
                    float(np.linalg.norm(ar.action - baseline.action)))

        # 5. Instruction dropout — substitute a single space (non-empty so
        # strict instruction validation passes; semantically empty so the
        # model sees no task content).
        if reqs.instruction and scene.instruction is not None:
            ar = model.predict(scene.with_instruction(" "))
            _record("instruction",
                    float(np.linalg.norm(ar.action - baseline.action)))

        if not per_modality:
            return self._not_applicable(
                model, scene,
                "model declares no input modalities we can drop "
                "(check required_inputs and that the scene has those fields populated)",
            )

        # Per-modality severity verdict.
        # SNR-aware per-modality verdict:
        #   below_noise → "unknown" (effect indistinguishable from model noise)
        #   below noise_floor_score → "ignored" (significant but tiny response — memorization-like)
        #   below grounded_threshold → "partial"
        #   above grounded_threshold → "used"
        # We use the RAW per-modality delta (not the normalized score) to
        # decide below_noise so the SNR comparison is honest.
        per_modality_severity: dict[str, str] = {}
        unknown: list[str] = []
        ignored: list[str] = []
        partial: list[str] = []
        used: list[str] = []
        noise_raw = self.calibration.noise_floor if self.calibration else 0.0
        for modality in per_modality:
            score_norm = per_modality[modality]
            raw_d = per_modality_raw.get(modality, score_norm)
            if self.calibration is not None and raw_d < noise_raw:
                per_modality_severity[modality] = "unknown"
                unknown.append(modality)
            elif score_norm < self.noise_floor:
                per_modality_severity[modality] = "ignored"
                ignored.append(modality)
            elif score_norm < self.grounded_threshold:
                per_modality_severity[modality] = "partial"
                partial.append(modality)
            else:
                per_modality_severity[modality] = "used"
                used.append(modality)

        # Scalar score = number of declared-but-ignored modalities (0 = healthy).
        # "unknown" modalities are NOT counted as ignored — we don't know.
        scalar = float(len(ignored))

        if ignored:
            sev = Severity.CRITICAL
            verdict = (
                f"Declared-but-ignored modalities (above noise floor, but "
                f"normalized response < {self.noise_floor}): {', '.join(ignored)}."
            )
        elif partial:
            sev = Severity.MODERATE
            verdict = (
                f"All testable modalities respond above noise floor; "
                f"{', '.join(partial)} only weakly (normalized < grounded "
                f"threshold {self.grounded_threshold})."
            )
        elif used:
            sev = Severity.PASS
            verdict = (
                f"All testable declared modalities are genuinely used "
                f"(normalized Δaction > {self.grounded_threshold} when each is dropped)."
            )
        else:
            # Everything came back unknown — can't say anything about usage.
            sev = Severity.UNKNOWN
            verdict = (
                f"No modality produced a Δaction above the model's noise "
                f"floor ({noise_raw:.4f}). Cannot distinguish 'ignored' from "
                f"'masked by decoding stochasticity'. To get a usable signal "
                f"either (a) use a less-stochastic model, (b) use stronger "
                f"perturbations, or (c) average many seeds."
            )
        if unknown:
            verdict += (
                f" Below-noise-floor (UNKNOWN — cannot judge): {', '.join(unknown)}."
            )
        if used:
            verdict += f" Genuinely used: {', '.join(used)}."
        if skipped_cams:
            verdict += (
                f" Skipped {skipped_cams} (not present in scene; the "
                "dataset adapter did not load them — investigate)."
            )

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=scalar,
            severity=sev,
            direction="higher_is_worse",   # more ignored modalities = worse
            explanation=verdict,
            per_variant=per_modality,
            raw={
                "per_modality_score":    per_modality,          # normalized (or raw if no calib)
                "per_modality_raw_delta": per_modality_raw,     # always raw L2
                "per_modality_verdict":  per_modality_severity,
                "calibration_used":      self.calibration.to_summary() if self.calibration else None,
                "noise_floor":           self.noise_floor,
                "grounded_threshold":    self.grounded_threshold,
                "ignored":               ignored,
                "partial":               partial,
                "unknown_below_noise":   unknown,
                "used":                  used,
                "tested_cameras": droppable_cams,
                "skipped_cameras_absent_from_scene": skipped_cams,
                "baseline_action": baseline.action.tolist(),
            },
        )
