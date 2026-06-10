"""Tests for the Ctrl-World dream stepper — pure numpy, no GPU.

A fake policy returns a fixed joint-velocity chunk and records the Scene it was
handed; a fake kinematics gives an affine joints->xyz readout. The tests verify
the control-rate -> native-rate bridging, the zero-order-hold extension for a
policy horizon shorter than the turn, the tracked-state advance, and the
camera mapping — all against the droid profile.

Run::

    uv run python adapters/emboviz-ctrlworld/tests/test_dream_step.py
"""

from __future__ import annotations

import numpy as np

from emboviz_wire.policy_bridge import JointStateTracker
from emboviz_wire.types import ActionResult, Scene

from emboviz_ctrlworld.dream_step import CtrlWorldDreamStepper
from emboviz_ctrlworld.profiles import get_profile
from emboviz_ctrlworld.stack_view import build_stack_view

_DROID = get_profile("droid")


class _FakeKinematics:
    """Duck-typed Kinematics — EE position is an affine readout of the joints,
    so a delta in joints maps to a delta in xyz; the real FK is validated in
    emboviz-robot's tests."""

    n_joints = 7

    class _Pose:
        def __init__(self, t):
            self.translation = t
            self.rotation = np.eye(3, dtype=np.float64)

    def fk(self, q):
        q = np.asarray(q, dtype=np.float64)
        return self._Pose(np.array([0.3 + q[0] * 0.1, q[1] * 0.1, 0.4 + q[2] * 0.1]))


class _FakePolicy:
    """Returns a chunk of constant joint velocities; records the scenes seen."""

    def __init__(self, horizon: int, vel: float = 0.15, gripper: float = 0.7):
        rows = np.zeros((horizon, 8), np.float32)
        rows[:, 0] = vel                       # joint 0 moves at `vel` rad/s
        rows[:, 7] = gripper
        self.chunk = rows
        self.scenes: list[Scene] = []

    def __call__(self, scene: Scene) -> ActionResult:
        self.scenes.append(scene)
        return ActionResult(action=self.chunk[0], action_chunk=self.chunk)


def _stack() -> np.ndarray:
    return build_stack_view(
        {
            "exterior_1": np.full((192, 320, 3), 10, np.uint8),
            "exterior_2": np.full((192, 320, 3), 20, np.uint8),
            "wrist": np.full((192, 320, 3), 30, np.uint8),
        },
        views=_DROID.views, view_hw=_DROID.view_hw,
    )


def _stepper(policy, **kwargs) -> CtrlWorldDreamStepper:
    tracker = JointStateTracker(np.zeros(7, np.float32), 0.0, _FakeKinematics(), control_hz=15.0)
    defaults = dict(
        profile=_DROID,
        tracker=tracker,
        camera_map={"primary": "exterior_1", "wrist_left": "wrist"},
        instruction="pick the marker",
        n_actions=4,
        control_hz=15.0,
    )
    defaults.update(kwargs)
    return CtrlWorldDreamStepper(policy, **defaults)


def test_rate_bridging_and_pose_rows() -> None:
    policy = _FakePolicy(horizon=12)           # covers 4 frames x 3 steps exactly
    stepper = _stepper(policy)
    rows = stepper(_stack())

    assert rows.shape == (4, 7)
    assert stepper.last_extended_rows == 0
    # Joint 0 integrates at vel * dt = 0.15 / 15 = 0.01 per control step; xyz
    # readout scales by 0.1, so frame k (3 control steps each) adds 0.003 in x.
    np.testing.assert_allclose(rows[:, 0], 0.3 + 0.003 * np.arange(1, 5), atol=1e-6)
    np.testing.assert_allclose(rows[:, 6], 0.7)   # absolute gripper rides along

    # The policy saw the mapped cameras + the tracked state.
    scene = policy.scenes[0]
    assert set(scene.observations.images) == {"primary", "wrist_left"}
    assert int(np.asarray(scene.observations.images["primary"].data)[0, 0, 0]) == 10
    assert int(np.asarray(scene.observations.images["wrist_left"].data)[0, 0, 0]) == 30
    assert scene.observations.state.convention == "joint_angles"
    assert scene.instruction == "pick the marker"


def test_n_actions_defaults_to_profile_chunk() -> None:
    policy = _FakePolicy(horizon=12)
    stepper = _stepper(policy, n_actions=None)
    assert stepper(_stack()).shape == (_DROID.frames_per_chunk, 7)


def test_zero_order_hold_extension_for_short_horizon() -> None:
    policy = _FakePolicy(horizon=10)           # π0-DROID: 2 rows short of 12
    stepper = _stepper(policy)
    rows = stepper(_stack())
    assert stepper.last_extended_rows == 2
    # Constant velocity means ZOH extension continues the same motion.
    np.testing.assert_allclose(rows[:, 0], 0.3 + 0.003 * np.arange(1, 5), atol=1e-6)


def test_tracker_advances_by_execute_steps_control_rows() -> None:
    policy = _FakePolicy(horizon=12)
    stepper = _stepper(policy, execute_steps=3)
    stepper(_stack())
    # 3 committed frames x 3 control steps x 0.01 rad = 0.09 on joint 0.
    np.testing.assert_allclose(stepper.tracker.joints[0], 0.09, atol=1e-6)
    # Re-planning next turn starts from the committed state.
    rows2 = stepper(_stack())
    np.testing.assert_allclose(rows2[0, 0], 0.3 + 0.1 * (0.09 + 0.03), atol=1e-6)
    assert stepper.steps_taken == 2


def test_rejects_bad_configuration() -> None:
    policy = _FakePolicy(horizon=12)
    for kwargs, fragment in (
        ({"camera_map": {"primary": "exterior_left"}}, "invalid stack views"),
        ({"n_actions": 5}, "multiple"),
        ({"execute_steps": 9}, "execute_steps"),
        ({"control_hz": 7.0}, "integer multiple"),
    ):
        try:
            _stepper(policy, **kwargs)
        except ValueError as e:
            assert fragment in str(e), str(e)
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_rejects_policy_without_chunk() -> None:
    def no_chunk(scene):
        return ActionResult(action=np.zeros(8, np.float32), action_chunk=None)

    stepper = _stepper(no_chunk)
    try:
        stepper(_stack())
    except ValueError as e:
        assert "action_chunk" in str(e)
    else:
        raise AssertionError("expected ValueError for chunk-less policy")


def _run_all() -> None:
    test_rate_bridging_and_pose_rows()
    test_n_actions_defaults_to_profile_chunk()
    test_zero_order_hold_extension_for_short_horizon()
    test_tracker_advances_by_execute_steps_control_rows()
    test_rejects_bad_configuration()
    test_rejects_policy_without_chunk()
    print("OK: all ctrl-world dream-stepper checks passed")


if __name__ == "__main__":
    _run_all()
