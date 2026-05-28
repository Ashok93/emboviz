"""Gripper-flip perturber — the dishwasher-dropped-item diagnostic.

If we flip the gripper input from full to empty (or vice versa) and the
model's action doesn't change, the model isn't grounding on gripper
state. This is the failure mode underlying many "policy keeps trying to
move an object it already dropped" rollouts.

Yields up to two variants:
  - "empty"  — gripper value at the low end of the range
  - "full"   — gripper value at the high end of the range
Variants whose new value is within `epsilon` of the current value are
skipped (no perturbation = no signal).

Multi-finger / suction grippers are out of scope for v1 — the perturber
yields 0 variants in that case, which the diagnostic surfaces as N/A.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from emboviz.core.observations import GripperState
from emboviz.core.types import PerturbedScene, Scene
from emboviz.perturb.base import Perturber
from emboviz.perturb.state._state_utils import make_perturbed_gripper_scene


_SUPPORTED_KINDS = frozenset({"parallel_jaw", "binary", "magnetic", "suction"})


class GripperFlipPerturber(Perturber):
    """Flip the gripper input between fully-open and fully-closed."""

    name = "gripper_flip"
    axis = "state.gripper_flip"
    affects = frozenset({"gripper"})

    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def variants(self, scene: Scene) -> Iterable[PerturbedScene]:
        gripper = scene.observations.gripper
        if gripper is None:
            return  # nothing to flip
        if gripper.kind not in _SUPPORTED_KINDS:
            return  # multi_finger etc. — diagnostic-level N/A

        # Pull the open/closed range from the RobotProfile if available;
        # otherwise fall back to convention by units.
        if scene.profile is not None and scene.profile.gripper is not None:
            lo, hi = scene.profile.gripper.range
        elif gripper.units == "unit":
            lo, hi = 0.0, 1.0
        elif gripper.units == "binary":
            lo, hi = 0.0, 1.0
        elif gripper.units == "mm":
            lo, hi = 0.0, 85.0   # Robotiq 2F-85 default span
        elif gripper.units == "rad":
            lo, hi = 0.0, 1.57   # ~pi/2
        else:
            lo, hi = 0.0, 1.0

        current = gripper.value
        for variant_id, new_value in (("empty", lo), ("full", hi)):
            if abs(new_value - current) <= self.epsilon:
                continue
            new_gripper = replace(gripper, value=float(new_value))
            yield make_perturbed_gripper_scene(
                scene=scene,
                perturber_name=self.name,
                axis=self.axis,
                variant_id=variant_id,
                new_gripper=new_gripper,
                description=f"gripper {gripper.value:.2f} → {new_value:.2f} ({variant_id})",
                parameters={
                    "original_value": float(current),
                    "new_value": float(new_value),
                    "kind": gripper.kind,
                    "units": gripper.units,
                },
            )
