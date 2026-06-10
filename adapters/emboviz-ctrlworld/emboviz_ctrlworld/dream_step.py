"""The per-step bridge that lets a user's policy fly inside the Ctrl-World dream.

One turn of the closed loop: the world model hands us the current (dreamed)
view stack; we split it into the policy's cameras, hand the policy that plus
its *tracked* proprioceptive state and the task instruction, take its action
chunk, integrate it to absolute end-effector poses at the world model's native
rate, and advance the tracked state by the committed steps. The next
world-model step conditions on the dream this produced.

Rate bridging. The policy runs at its control rate (π0-DROID: 15 Hz) while the
world model conditions at its profile's native rate (DROID profile: 5 Hz), so
each dreamed frame spans ``control_hz / native_fps`` control steps. A turn of
``n_actions`` future frames therefore consumes that many chunk rows per frame;
a policy horizon shorter than the turn is extended by repeating its last row
(zero-order hold), exactly the reference rollout's index padding for the pi0
policy (``rollout_interact_pi.py`` line 254). The extension count is exposed
on ``last_extended_rows`` and only conditions the dream's tail; set
``execute_steps`` below ``n_actions`` to keep extrapolated state out of the
committed rollout entirely.

State is tracked, not observed: the world model dreams pixels, not
proprioception. The :class:`emboviz_wire.policy_bridge.StateTracker` maintains
it — one instance per clip, called once per loop turn.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from emboviz_wire.observations import RGBImage
from emboviz_wire.policy_bridge import StateTracker
from emboviz_wire.types import Observations, Scene

from emboviz_ctrlworld.profiles import CtrlWorldProfile
from emboviz_ctrlworld.stack_view import split_stack_view


class CtrlWorldDreamStepper:
    """Stateful ``stack_image -> pose conditioning`` step for the closed-loop sim.

    Parameters
    ----------
    predict_fn
        Maps a :class:`Scene` to an ``ActionResult`` (a connected policy's
        ``predict``).
    profile
        The checkpoint profile driving the loop — supplies the view layout,
        the native rate, and the chunk quantum.
    tracker
        The :class:`StateTracker` holding the policy's proprioceptive state.
    camera_map
        ``{policy_camera_role: view_name}`` — which stack view feeds each
        camera the policy expects (e.g. ``{"primary": "exterior_1",
        "wrist_left": "wrist"}``). Views must be in ``profile.views``.
    instruction
        The task string handed to the policy each turn. Required by language-
        conditioned policies; pass the seed episode's instruction.
    n_actions
        Prediction horizon in dreamed frames per turn. Must be a positive
        multiple of ``profile.frames_per_chunk``.
    execute_steps
        Execution horizon: dreamed frames the policy commits to before
        re-planning (receding horizon). Defaults to ``n_actions``. Must
        satisfy ``1 <= execute_steps <= n_actions``. The loop must commit the
        same number of frames so the tracked state stays aligned with the
        conditioning frame.
    control_hz
        The policy's control rate. ``control_hz / profile.native_fps`` must be
        a positive integer (control steps per dreamed frame).
    """

    def __init__(
        self,
        predict_fn: Callable[[Scene], "object"],
        *,
        profile: CtrlWorldProfile,
        tracker: StateTracker,
        camera_map: dict[str, str],
        instruction: Optional[str] = None,
        n_actions: Optional[int] = None,
        execute_steps: Optional[int] = None,
        control_hz: float = 15.0,
    ):
        if not camera_map:
            raise ValueError("CtrlWorldDreamStepper: camera_map must map at least one policy camera.")
        bad = {v for v in camera_map.values() if v not in profile.views}
        if bad:
            raise ValueError(
                f"CtrlWorldDreamStepper: invalid stack views {sorted(bad)}; "
                f"profile '{profile.name}' has views {sorted(profile.views)}."
            )
        n_actions = profile.frames_per_chunk if n_actions is None else int(n_actions)
        if n_actions < 1 or n_actions % profile.frames_per_chunk != 0:
            raise ValueError(
                f"CtrlWorldDreamStepper: n_actions must be a positive multiple of "
                f"{profile.frames_per_chunk} (profile '{profile.name}''s chunk "
                f"quantum); got {n_actions}."
            )
        if execute_steps is not None and not 1 <= int(execute_steps) <= n_actions:
            raise ValueError(
                f"CtrlWorldDreamStepper: execute_steps must satisfy 1 <= execute_steps "
                f"<= n_actions ({n_actions}); got {execute_steps}."
            )
        steps_per_frame = float(control_hz) / profile.native_fps
        if abs(steps_per_frame - round(steps_per_frame)) > 1e-9 or round(steps_per_frame) < 1:
            raise ValueError(
                f"CtrlWorldDreamStepper: control_hz ({control_hz:g}) must be a "
                f"positive integer multiple of profile '{profile.name}''s "
                f"{profile.native_fps:g} Hz native rate."
            )

        self._predict_fn = predict_fn
        self._profile = profile
        self._tracker = tracker
        self._camera_map = dict(camera_map)
        self._instruction = instruction
        self._n_actions = n_actions
        self._execute_steps = None if execute_steps is None else int(execute_steps)
        self._steps_per_frame = int(round(steps_per_frame))
        self.steps_taken = 0
        #: Chunk rows appended by zero-order hold on the LAST call (0 when the
        #: policy's horizon covered the turn). Recorded so a clip can disclose
        #: how much of its conditioning tail was extrapolated.
        self.last_extended_rows = 0

    @property
    def execute_steps(self) -> int:
        """Resolved execution horizon (``n_actions`` when unset)."""
        return self._n_actions if self._execute_steps is None else self._execute_steps

    @property
    def tracker(self) -> StateTracker:
        return self._tracker

    def __call__(self, stack_image: np.ndarray) -> np.ndarray:
        """One loop turn: dreamed stack in, ``(n_actions, 7)`` pose rows out."""
        views = split_stack_view(stack_image, views=self._profile.views)
        images = {
            role: RGBImage(data=views[view], camera_id=role)
            for role, view in self._camera_map.items()
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
                "CtrlWorldDreamStepper: the policy returned no action_chunk; the "
                "closed loop needs a multi-step chunk to render. Use a chunk-"
                "predicting policy."
            )
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] < 1:
            raise ValueError(
                f"CtrlWorldDreamStepper: policy chunk has shape {chunk.shape}; "
                "expected (T, row_dim) with T >= 1."
            )

        needed = self._n_actions * self._steps_per_frame
        if chunk.shape[0] < needed:
            self.last_extended_rows = needed - chunk.shape[0]
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1:], self.last_extended_rows, axis=0)]
            )
        else:
            self.last_extended_rows = 0

        states, grippers = self._tracker.integrate(chunk, needed)
        rows = [
            np.concatenate(
                [
                    states[k * self._steps_per_frame],
                    [grippers[k * self._steps_per_frame - 1]],
                ]
            )
            for k in range(1, self._n_actions + 1)
        ]
        self._tracker.advance(chunk, self.execute_steps * self._steps_per_frame)
        self.steps_taken += 1
        return np.stack(rows).astype(np.float32)


__all__ = ["CtrlWorldDreamStepper"]
