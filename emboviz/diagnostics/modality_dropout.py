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

    Args:
        noise_floor: Δaction below this counts as "model didn't notice."
        grounded_threshold: Δaction above this counts as "model genuinely uses it."
        cameras: which cameras to test individually. None = every required
                 camera declared by the model that is also present in the scene.
    """

    required_capabilities = Capability.INFERENCE

    def __init__(
        self,
        noise_floor: float = 0.05,
        grounded_threshold: float = 0.30,
        cameras: Optional[list[str]] = None,
    ):
        self.noise_floor = noise_floor
        self.grounded_threshold = grounded_threshold
        self.cameras = cameras
        self.name = "modality_dropout"
        self.axis = "input.modality_dropout"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks INFERENCE capability")

        reqs = model.required_inputs
        baseline = model.predict(scene)

        per_modality: dict[str, float] = {}

        # 1. Image dropout — one variant per camera the model requires AND
        # the scene actually provides, plus an ALL-cameras combined drop.
        requested_cams = (
            sorted(self.cameras) if self.cameras is not None else sorted(reqs.cameras)
        )
        droppable_cams = [c for c in requested_cams if c in scene.observations.images]
        skipped_cams = [c for c in requested_cams if c not in scene.observations.images]
        for cam in droppable_cams:
            arr = to_array(scene.observations.images[cam].data)
            blank = to_pil(np.full_like(arr, fill_value=int(arr.mean())))
            blank_scene = scene.with_image(blank, camera=cam)
            ar = model.predict(blank_scene)
            per_modality[f"drop_image:{cam}"] = float(
                np.linalg.norm(ar.action - baseline.action)
            )
        if len(droppable_cams) > 1:
            all_blank = {}
            for cam in droppable_cams:
                arr = to_array(scene.observations.images[cam].data)
                all_blank[cam] = to_pil(np.full_like(arr, fill_value=int(arr.mean())))
            ar = model.predict(scene.with_images(all_blank))
            per_modality["drop_image:ALL"] = float(
                np.linalg.norm(ar.action - baseline.action)
            )

        # 2. State dropout.
        if reqs.state and scene.observations.state is not None:
            state = scene.observations.state
            zeroed_state = Proprioception(
                values=np.zeros_like(state.values), convention=state.convention,
            )
            new_obs = replace(scene.observations, state=zeroed_state)
            ar = model.predict(replace(scene, observations=new_obs))
            per_modality["state"] = float(np.linalg.norm(ar.action - baseline.action))

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
            per_modality["gripper"] = float(np.linalg.norm(ar.action - baseline.action))

        # 4. Action-history dropout.
        if reqs.action_history and scene.observations.action_history is not None:
            hist = scene.observations.action_history
            zeroed_hist = replace(hist, actions=np.zeros_like(hist.actions))
            new_obs = replace(scene.observations, action_history=zeroed_hist)
            ar = model.predict(replace(scene, observations=new_obs))
            per_modality["action_history"] = float(
                np.linalg.norm(ar.action - baseline.action)
            )

        # 5. Instruction dropout — we substitute a single space rather than
        # an empty string. Models with strict required_inputs validation
        # reject empty instructions outright (correct contract), and we
        # want to MEASURE the model's response, not trip the validator.
        # A space is non-empty (passes validation), tokenizes to whitespace
        # only, and conveys no task content — a clean "instruction absent"
        # signal.
        if reqs.instruction and scene.instruction is not None:
            ar = model.predict(scene.with_instruction(" "))
            per_modality["instruction"] = float(np.linalg.norm(ar.action - baseline.action))

        if not per_modality:
            return self._not_applicable(
                model, scene,
                "model declares no input modalities we can drop "
                "(check required_inputs and that the scene has those fields populated)",
            )

        # Per-modality severity verdict.
        per_modality_severity: dict[str, str] = {}
        ignored: list[str] = []
        partial: list[str] = []
        used: list[str] = []
        for modality, d in per_modality.items():
            if d < self.noise_floor:
                per_modality_severity[modality] = "ignored"
                ignored.append(modality)
            elif d < self.grounded_threshold:
                per_modality_severity[modality] = "partial"
                partial.append(modality)
            else:
                per_modality_severity[modality] = "used"
                used.append(modality)

        # Scalar score = number of declared-but-ignored modalities (0 = healthy).
        scalar = float(len(ignored))

        if ignored:
            sev = Severity.CRITICAL
            verdict = (
                f"Declared-but-ignored modalities: {', '.join(ignored)}. "
                f"Dropping these inputs barely moves the action "
                f"(< noise floor {self.noise_floor})."
            )
            if partial:
                verdict += f" Partial use of: {', '.join(partial)}."
            if used:
                verdict += f" Genuinely used: {', '.join(used)}."
        elif partial:
            sev = Severity.MODERATE
            verdict = (
                f"All declared modalities respond above noise floor, but "
                f"{', '.join(partial)} only weakly (< grounded threshold "
                f"{self.grounded_threshold}). Genuinely used: {', '.join(used)}."
            )
        else:
            sev = Severity.PASS
            verdict = (
                f"All declared modalities are genuinely used (Δaction > "
                f"{self.grounded_threshold} when each is dropped)."
            )
        if skipped_cams:
            verdict += (
                f" Skipped {skipped_cams} (not present in scene; "
                "the dataset adapter did not load them — investigate)."
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
                "per_modality_delta": per_modality,
                "per_modality_verdict": per_modality_severity,
                "noise_floor": self.noise_floor,
                "grounded_threshold": self.grounded_threshold,
                "ignored": ignored,
                "partial": partial,
                "used": used,
                "tested_cameras": droppable_cams,
                "skipped_cameras_absent_from_scene": skipped_cams,
                "baseline_action": baseline.action.tolist(),
            },
        )
