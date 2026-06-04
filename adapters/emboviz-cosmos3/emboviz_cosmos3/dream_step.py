"""The per-step bridge that lets a user's policy fly inside the Cosmos dream.

One turn of the closed loop: the world model hands us the current (dreamed)
``concat_view`` frame; we split it into the policy's cameras, hand the policy that
plus the *tracked* end-effector state, take its action chunk, convert it to Cosmos
conditioning, and advance our tracked state by integrating those actions. The next
world-model step conditions on the dream this produced.

The state is tracked, not observed: Cosmos dreams pixels, not proprioception, so we
maintain the end-effector pose ourselves by integrating the policy's own actions
from the real seed pose (the same bridge math used to encode them). This object is
stateful by design — one instance per clip, called once per loop step.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from emboviz_wire.observations import RGBImage
from emboviz_wire.observations.gripper import GripperState
from emboviz_wire.observations.state import Proprioception
from emboviz_wire.types import Observations, Scene

from emboviz_cosmos3.bridge import ActionConvention, integrate_policy_chunk
from emboviz_cosmos3.concat_view import ConcatRegion, split_concat_view
from emboviz_cosmos3.domains import encode_droid_states

_VALID_REGIONS = {"wrist", "exterior_left", "exterior_right"}


class PolicyDreamStepper:
    """Stateful ``concat_image -> cosmos_actions`` step for the closed-loop sim.

    Parameters
    ----------
    predict_fn
        Maps a :class:`Scene` to an ``ActionResult`` (a connected policy's
        ``predict``).
    action_convention
        How to interpret the policy's action chunk (see
        :mod:`emboviz_cosmos3.bridge`).
    camera_map
        ``{policy_camera_role: concat_region}`` — which split region feeds each
        camera the policy expects (e.g. ``{"primary": "exterior_left", "wrist":
        "wrist"}``). Every region must be one of ``wrist``, ``exterior_left``,
        ``exterior_right``.
    seed_state
        The real end-effector pose ``[x, y, z, roll, pitch, yaw]`` at the seed
        frame — the integration anchor.
    seed_gripper
        The real gripper value in ``[0, 1]`` at the seed frame.
    n_actions
        Steps to take from the policy's chunk per loop turn (the world model then
        renders this many frames). Must not exceed the policy's chunk length.
    state_convention
        Proprioception convention label handed to the policy (default
        ``"ee_pose"`` — DROID cartesian).
    """

    def __init__(
        self,
        predict_fn: Callable[[Scene], "object"],
        *,
        action_convention: ActionConvention,
        camera_map: dict[str, ConcatRegion],
        seed_state: np.ndarray,
        seed_gripper: float,
        n_actions: int = 16,
        state_convention: str = "ee_pose",
    ):
        if not camera_map:
            raise ValueError("PolicyDreamStepper: camera_map must map at least one policy camera.")
        bad = {r for r in camera_map.values() if r not in _VALID_REGIONS}
        if bad:
            raise ValueError(
                f"PolicyDreamStepper: invalid concat regions {sorted(bad)}; "
                f"valid regions are {sorted(_VALID_REGIONS)}."
            )
        if int(n_actions) < 1:
            raise ValueError(f"PolicyDreamStepper: n_actions must be >= 1, got {n_actions}.")

        state = np.asarray(seed_state, dtype=np.float32).reshape(-1)
        if state.shape[0] < 6:
            raise ValueError(f"PolicyDreamStepper: seed_state must be >=6-D [xyz, euler], got {state.shape}.")

        self._predict_fn = predict_fn
        self._action_convention = action_convention
        self._camera_map = dict(camera_map)
        self._n_actions = int(n_actions)
        self._state_convention = state_convention
        self._state = state[:6].copy()
        self._gripper = float(seed_gripper)
        self.steps_taken = 0

    def __call__(self, concat_image: np.ndarray) -> np.ndarray:
        """One loop turn: dreamed frame in, Cosmos conditioning actions out."""
        regions = split_concat_view(concat_image)
        images = {
            role: RGBImage(data=regions[region], camera_id=role)
            for role, region in self._camera_map.items()
        }
        scene = Scene(
            observations=Observations(
                images=images,
                state=Proprioception(values=self._state.copy(), convention=self._state_convention),
                gripper=GripperState(value=self._gripper),
            )
        )

        result = self._predict_fn(scene)
        chunk = getattr(result, "action_chunk", None)
        if chunk is None:
            raise ValueError(
                "PolicyDreamStepper: the policy returned no action_chunk; the closed "
                "loop needs a multi-step chunk to render. Use a chunk-predicting policy."
            )
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] < self._n_actions:
            raise ValueError(
                f"PolicyDreamStepper: policy chunk {chunk.shape} too short for "
                f"n_actions={self._n_actions}."
            )

        states, grippers = integrate_policy_chunk(
            self._state, chunk[: self._n_actions], self._action_convention
        )
        cosmos_actions = encode_droid_states(states, grippers)
        # Advance the tracked pose/gripper to where the policy's actions took it.
        self._state = states[-1].astype(np.float32)
        self._gripper = float(grippers[-1])
        self.steps_taken += 1
        return cosmos_actions


__all__ = ["PolicyDreamStepper"]
