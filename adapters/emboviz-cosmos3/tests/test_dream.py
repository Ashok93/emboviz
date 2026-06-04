"""Tests for the closed-loop glue: concat_view split/build + PolicyDreamStepper.

Pure numpy, no GPU/server. Verifies the stitch/split geometry round-trips, and
that the stepper feeds the policy the right cameras + tracked state, converts its
chunk to (n,10) Cosmos conditioning, and advances the tracked pose each turn.

Run::

    uv run --with pillow python adapters/emboviz-cosmos3/tests/test_dream.py
"""

from __future__ import annotations

import numpy as np

from emboviz_wire.types import ActionResult, Scene

from emboviz_cosmos3.bridge import CartesianStateTracker, JointStateTracker
from emboviz_cosmos3.concat_view import build_concat_view, split_concat_view
from emboviz_cosmos3.dream_step import PolicyDreamStepper


class _FakeKinematics:
    """Duck-typed Kinematics for the joint path — no pinocchio needed.

    EE position is an affine readout of the joints (so a delta in joints maps to a
    delta in xyz, exercising integration + FK + encode without a real robot); the
    real FK is validated in emboviz-robot's tests.
    """

    n_joints = 7

    class _Pose:
        def __init__(self, t):
            self.translation = t
            self.rotation = np.eye(3, dtype=np.float64)

    def fk(self, q):
        q = np.asarray(q, dtype=np.float64)
        return self._Pose(np.array([0.3 + q[:3].sum() * 0.01, q[3] * 0.01, 0.4 + q[4] * 0.01]))


def test_build_then_split_round_trips_geometry() -> None:
    wrist = np.random.RandomState(1).randint(0, 256, (12, 16, 3), np.uint8)
    left = np.full((8, 10, 3), 70, np.uint8)
    right = np.full((8, 10, 3), 200, np.uint8)
    concat = build_concat_view(wrist, left, right)
    assert concat.shape == (18, 16, 3)            # 12 + 12//2 rows

    regions = split_concat_view(concat)
    assert regions["wrist"].shape == (12, 16, 3)
    assert np.array_equal(regions["wrist"], wrist)   # wrist preserved exactly
    assert regions["exterior_left"].shape == (6, 8, 3)
    assert regions["exterior_right"].shape == (6, 8, 3)


def test_concat_wrist_size_sets_training_resolution() -> None:
    from emboviz_cosmos3.concat_view import DROID_TRAIN_WRIST_HW

    wrist = np.full((180, 320, 3), 100, np.uint8)     # our native DROID camera size
    left = np.full((180, 320, 3), 50, np.uint8)
    right = np.full((180, 320, 3), 200, np.uint8)

    # wrist_size upscales to the Cosmos DROID training scale -> 540x640 concat.
    concat = build_concat_view(wrist, left, right, wrist_size=DROID_TRAIN_WRIST_HW)
    assert concat.shape == (540, 640, 3)              # 360 wrist + 180 bottom, width 640
    regions = split_concat_view(concat)
    assert regions["wrist"].shape == (360, 640, 3)
    assert regions["exterior_left"].shape == (180, 320, 3)
    assert regions["exterior_right"].shape == (180, 320, 3)

    # Without wrist_size the concat stays at the cameras' native scale.
    assert build_concat_view(wrist, left, right).shape == (270, 320, 3)


def test_split_rejects_bad_shape() -> None:
    try:
        split_concat_view(np.zeros((10, 10), np.uint8))
    except ValueError as e:
        assert "(H, W, 3)" in str(e)
    else:
        raise AssertionError("expected ValueError for non-RGB input")


def test_stepper_feeds_policy_and_advances_state() -> None:
    seen: dict = {}

    def predict_fn(scene: Scene) -> ActionResult:
        seen["cameras"] = sorted(scene.observations.images)
        seen["state"] = np.asarray(scene.observations.state.values, dtype=np.float32).copy()
        seen["gripper"] = float(scene.observations.gripper.value)
        seen["instruction"] = scene.instruction
        seen["wrist_shape"] = scene.observations.images["wrist"].data.shape
        # A small, mostly-translation chunk in base-frame deltas.
        chunk = np.zeros((16, 7), dtype=np.float32)
        chunk[:, 0] = 0.001          # +x each step
        chunk[:, 6] = 0.5            # gripper command
        return ActionResult(action=chunk[0], action_chunk=chunk)

    seed_state = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    stepper = PolicyDreamStepper(
        predict_fn,
        tracker=CartesianStateTracker(seed_state, 0.9, "delta_xyz_euler_base"),
        camera_map={"primary": "exterior_left", "wrist": "wrist"},
        instruction="pick the marker from the cup",
        n_actions=16,
    )

    concat = build_concat_view(
        np.zeros((12, 16, 3), np.uint8),
        np.full((8, 10, 3), 50, np.uint8),
        np.full((8, 10, 3), 150, np.uint8),
    )
    actions = stepper(concat)

    assert actions.shape == (16, 10)                       # Cosmos droid_lerobot conditioning
    assert seen["cameras"] == ["primary", "wrist"]         # policy got its mapped cameras
    assert np.allclose(seen["state"], seed_state)          # first turn sees the seed pose
    assert seen["gripper"] == 0.9
    assert seen["instruction"] == "pick the marker from the cup"   # language threaded through
    assert seen["wrist_shape"] == (12, 16, 3)              # wrist region, full width
    # The tracked pose advanced by +x over 16 steps (~0.016 m).
    assert stepper.tracker.state[0] > seed_state[0] + 0.01
    assert stepper.steps_taken == 1


def test_joint_stepper_tracks_joints_and_feeds_joint_state() -> None:
    seen: dict = {}

    def predict_fn(scene: Scene) -> ActionResult:
        seen["state"] = np.asarray(scene.observations.state.values, dtype=np.float32).copy()
        seen["convention"] = scene.observations.state.convention
        # 8-D droid_joint_delta chunk: [joint_delta(7), gripper(1)].
        chunk = np.zeros((16, 8), dtype=np.float32)
        chunk[:, 0] = 0.01           # +joint0 each step
        chunk[:, 7] = 0.8            # absolute gripper
        return ActionResult(action=chunk[0], action_chunk=chunk)

    seed_joints = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.6, 0.0], dtype=np.float32)
    stepper = PolicyDreamStepper(
        predict_fn,
        tracker=JointStateTracker(seed_joints, 0.2, _FakeKinematics()),
        camera_map={"primary": "exterior_left", "wrist_left": "wrist"},
        instruction="unfold the cloth",
        n_actions=16,
    )
    concat = build_concat_view(
        np.zeros((12, 16, 3), np.uint8),
        np.full((8, 10, 3), 50, np.uint8),
        np.full((8, 10, 3), 150, np.uint8),
    )
    actions = stepper(concat)

    assert actions.shape == (16, 10)                       # joint path also yields (n, 10)
    assert seen["convention"] == "joint_angles"            # policy gets joints, not a pose
    assert np.allclose(seen["state"], seed_joints)         # first turn sees the seed joints
    # joint0 advanced by +0.01 * 16 = 0.16 rad.
    assert np.isclose(stepper.tracker.joints[0], seed_joints[0] + 0.16, atol=1e-4)
    assert stepper.steps_taken == 1


def test_stepper_rejects_bad_region_and_missing_chunk() -> None:
    tracker = CartesianStateTracker(np.zeros(6, np.float32), 0.0, "absolute_xyz_euler")
    try:
        PolicyDreamStepper(
            lambda s: None, tracker=tracker, camera_map={"primary": "front"},
        )
    except ValueError as e:
        assert "invalid concat regions" in str(e)
    else:
        raise AssertionError("expected ValueError for invalid region")

    def no_chunk(scene):
        return ActionResult(action=np.zeros(7, np.float32))  # action_chunk is None

    stepper = PolicyDreamStepper(
        no_chunk,
        tracker=CartesianStateTracker(np.zeros(6, np.float32), 0.0, "absolute_xyz_euler"),
        camera_map={"primary": "wrist"}, n_actions=4,
    )
    concat = build_concat_view(
        np.zeros((12, 16, 3), np.uint8), np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8)
    )
    try:
        stepper(concat)
    except ValueError as e:
        assert "no action_chunk" in str(e)
    else:
        raise AssertionError("expected ValueError for missing chunk")


def _run_all() -> None:
    test_build_then_split_round_trips_geometry()
    test_concat_wrist_size_sets_training_resolution()
    test_split_rejects_bad_shape()
    test_stepper_feeds_policy_and_advances_state()
    test_joint_stepper_tracks_joints_and_feeds_joint_state()
    test_stepper_rejects_bad_region_and_missing_chunk()
    print("OK: all closed-loop glue (concat + stepper) checks passed")


if __name__ == "__main__":
    _run_all()
