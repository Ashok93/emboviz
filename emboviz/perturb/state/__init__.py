"""State-side perturbers — mutate the robot's proprioception, gripper,
or action history input to the policy, leaving image and text untouched.

These are the diagnostics that surface the "model declares it reads
state but actually ignores it" failure mode — the canonical case being
the dishwasher-dropped-item: if the gripper input flips from full to
empty and the action doesn't change, the model isn't grounding on
gripper state.

Every perturber here declares `affects` against the relevant typed
field so the Counterfactual diagnostic auto-skips against models that
don't consume that input.
"""

from emboviz.perturb.state.gripper_flip import GripperFlipPerturber
from emboviz.perturb.state.history_ablate import HistoryAblatePerturber
from emboviz.perturb.state.history_scramble import HistoryScramblePerturber
from emboviz.perturb.state.state_jitter import StateJitterPerturber

__all__ = [
    "GripperFlipPerturber",
    "HistoryAblatePerturber",
    "HistoryScramblePerturber",
    "StateJitterPerturber",
]
