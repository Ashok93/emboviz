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

from emboviz_cosmos3.concat_view import build_concat_view, split_concat_view
from emboviz_cosmos3.dream_step import PolicyDreamStepper


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
        seen["wrist_shape"] = scene.observations.images["wrist"].data.shape
        # A small, mostly-translation chunk in base-frame deltas.
        chunk = np.zeros((16, 7), dtype=np.float32)
        chunk[:, 0] = 0.001          # +x each step
        chunk[:, 6] = 0.5            # gripper command
        return ActionResult(action=chunk[0], action_chunk=chunk)

    seed_state = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    stepper = PolicyDreamStepper(
        predict_fn,
        action_convention="delta_xyz_euler_base",
        camera_map={"primary": "exterior_left", "wrist": "wrist"},
        seed_state=seed_state, seed_gripper=0.9, n_actions=16,
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
    assert seen["wrist_shape"] == (12, 16, 3)              # wrist region, full width
    # The tracked pose advanced by +x over 16 steps (~0.016 m).
    assert stepper._state[0] > seed_state[0] + 0.01
    assert stepper.steps_taken == 1


def test_stepper_rejects_bad_region_and_missing_chunk() -> None:
    try:
        PolicyDreamStepper(
            lambda s: None, action_convention="absolute_xyz_euler",
            camera_map={"primary": "front"}, seed_state=np.zeros(6, np.float32), seed_gripper=0.0,
        )
    except ValueError as e:
        assert "invalid concat regions" in str(e)
    else:
        raise AssertionError("expected ValueError for invalid region")

    def no_chunk(scene):
        return ActionResult(action=np.zeros(7, np.float32))  # action_chunk is None

    stepper = PolicyDreamStepper(
        no_chunk, action_convention="absolute_xyz_euler",
        camera_map={"primary": "wrist"}, seed_state=np.zeros(6, np.float32), seed_gripper=0.0, n_actions=4,
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
    test_split_rejects_bad_shape()
    test_stepper_feeds_policy_and_advances_state()
    test_stepper_rejects_bad_region_and_missing_chunk()
    print("OK: all closed-loop glue (concat + stepper) checks passed")


if __name__ == "__main__":
    _run_all()
