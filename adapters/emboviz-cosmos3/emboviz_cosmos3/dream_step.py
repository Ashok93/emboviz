"""The per-step bridge that lets a user's policy fly inside the Cosmos dream.

One turn of the closed loop: the world model hands us the current (dreamed)
``concat_view`` frame; we split it into the policy's cameras, hand the policy that
plus its *tracked* proprioceptive state and the task instruction, take its action
chunk, convert it to Cosmos conditioning, and advance the tracked state by
integrating those actions. The next world-model step conditions on the dream this
produced.

State is tracked, not observed: Cosmos dreams pixels, not proprioception. The
:class:`emboviz_wire.policy_bridge.StateTracker` maintains it — a Cartesian
tracker follows the end-effector pose, a joint tracker follows the joint vector
and forward-kinematics it. This object is stateful by design: one instance per
clip, called once per loop step.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from emboviz_wire.observations import RGBImage
from emboviz_wire.policy_bridge import StateTracker
from emboviz_wire.types import Observations, Scene

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
    tracker
        The :class:`StateTracker` that holds the policy's proprioceptive state and
        encodes its action chunk into Cosmos conditioning (Cartesian or joint).
    camera_map
        ``{policy_camera_role: concat_region}`` — which split region feeds each
        camera the policy expects (e.g. ``{"primary": "exterior_left",
        "wrist_left": "wrist"}``). Every region must be one of ``wrist``,
        ``exterior_left``, ``exterior_right``.
    instruction
        The task string handed to the policy each turn. Required by language-
        conditioned policies (π0 raises on an empty instruction); pass the seed
        episode's instruction.
    n_actions
        Prediction horizon: rows taken from the policy's chunk and encoded as
        conditioning, so the world model renders this many frames per turn. Must
        not exceed the policy's chunk length.
    execute_steps
        Execution horizon: how many of those dreamed frames the policy commits to
        before re-planning (receding horizon). Defaults to ``n_actions`` (commit
        the whole chunk). ``1`` is the most reactive — the policy re-plans on the
        next dreamed frame. Must satisfy ``1 <= execute_steps <= n_actions``. The
        loop must commit the same number of frames (see
        :func:`emboviz.world_models.simulate.closed_loop_rollout`) so the tracked
        state stays aligned with the conditioning frame.
    """

    def __init__(
        self,
        predict_fn: Callable[[Scene], "object"],
        *,
        tracker: StateTracker,
        camera_map: dict[str, ConcatRegion],
        instruction: Optional[str] = None,
        n_actions: int = 16,
        execute_steps: Optional[int] = None,
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
        if execute_steps is not None and not 1 <= int(execute_steps) <= int(n_actions):
            raise ValueError(
                f"PolicyDreamStepper: execute_steps must satisfy 1 <= execute_steps <= "
                f"n_actions ({int(n_actions)}); got {execute_steps}."
            )

        self._predict_fn = predict_fn
        self._tracker = tracker
        self._camera_map = dict(camera_map)
        self._instruction = instruction
        self._n_actions = int(n_actions)
        self._execute_steps = None if execute_steps is None else int(execute_steps)
        self.steps_taken = 0

    @property
    def execute_steps(self) -> int:
        """Resolved execution horizon (``n_actions`` when unset)."""
        return self._n_actions if self._execute_steps is None else self._execute_steps

    @property
    def tracker(self) -> StateTracker:
        return self._tracker

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
                state=self._tracker.proprioception(),
                gripper=self._tracker.gripper_state(),
            ),
            instruction=self._instruction,
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

        # Encode the FULL prediction horizon as conditioning (the dream renders
        # the whole chunk), then advance the tracked state only by the committed
        # steps (receding horizon) so next turn's proprioception matches the
        # committed frame.
        states, grippers = self._tracker.integrate(chunk, self._n_actions)
        cosmos_actions = encode_droid_states(states, grippers)
        self._tracker.advance(chunk, self.execute_steps)
        self.steps_taken += 1
        return cosmos_actions


__all__ = ["PolicyDreamStepper"]
