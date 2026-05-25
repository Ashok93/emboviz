"""Memorization-sniff diagnostic.

If we mask out the target and the model still produces a sizeable, coherent
action — it's running on memorized trajectories, not visual feedback.

Output:
  • scalar = ‖action_with_target_masked − null_action‖
    (where null_action is what the model does on an entirely blanked scene)
  • severity: HIGH if model still acts vigorously without target visible.
"""

from __future__ import annotations

import numpy as np

from policylens.core.results import DiagnosticResult, Severity
from policylens.core.types import Scene
from policylens.diagnostics.base import Diagnostic
from policylens.models.protocol import Capability, VLAModel
from policylens.perturb.image._image_utils import to_array, to_pil
from policylens.perturb.image.target_remove import TargetRemovalPerturber


class MemorizationDiagnostic(Diagnostic):
    """Mask the target; check whether the model still executes a coherent action."""

    required_capabilities = Capability.INFERENCE

    def __init__(
        self,
        bbox: tuple[int, int, int, int] | None = None,
        coherent_threshold: float = 0.20,
    ):
        self.bbox = bbox
        self.coherent_threshold = coherent_threshold
        self.name = "memorization_test"
        self.axis = "vision.memorization"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(model, scene, "model lacks INFERENCE capability")

        # Baseline with full scene
        baseline = model.predict(scene.image, scene.instruction)

        # Target removed
        target_remover = TargetRemovalPerturber(bbox=self.bbox)
        masked_scene = next(iter(target_remover.variants(scene))).scene
        action_no_target = model.predict(masked_scene.image, masked_scene.instruction)

        # Reference: fully blanked image
        arr = to_array(scene.image)
        blank = np.full_like(arr, fill_value=int(arr.mean()))
        action_blank = model.predict(to_pil(blank), scene.instruction)

        # How vigorous is the action when the target is masked vs blanked?
        diff_vs_blank = float(np.linalg.norm(action_no_target.action - action_blank.action))
        diff_vs_baseline = float(np.linalg.norm(action_no_target.action - baseline.action))
        action_magnitude = float(np.linalg.norm(action_no_target.action))

        if diff_vs_baseline < self.coherent_threshold and action_magnitude > self.coherent_threshold:
            sev = Severity.CRITICAL
            verdict = (
                f"Even with the target masked, the model produces an action that is nearly "
                f"identical to the original (Δ={diff_vs_baseline:.3f}) and has substantial "
                f"magnitude ({action_magnitude:.3f}). It's memorizing the trajectory rather "
                f"than reading the scene."
            )
        elif diff_vs_baseline < 2 * self.coherent_threshold:
            sev = Severity.MODERATE
            verdict = (
                f"With target masked, action stays similar (Δ={diff_vs_baseline:.3f}). "
                f"Partial memorization."
            )
        else:
            sev = Severity.PASS
            verdict = (
                f"With target masked, the model's action changes substantially "
                f"(Δ={diff_vs_baseline:.3f}) — it's reading visual feedback, not memorizing."
            )

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=scene.scene_id,
            scalar_score=diff_vs_baseline,
            severity=sev,
            direction="lower_is_worse",
            explanation=verdict,
            per_variant={
                "diff_vs_baseline": diff_vs_baseline,
                "diff_vs_blank": diff_vs_blank,
                "action_magnitude": action_magnitude,
            },
            raw={
                "baseline_action": baseline.action.tolist(),
                "action_target_masked": action_no_target.action.tolist(),
                "action_blank_scene": action_blank.action.tolist(),
            },
        )
