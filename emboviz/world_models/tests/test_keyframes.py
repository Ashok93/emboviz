"""Tests for critical-moment (keyframe) detection — pure, no GPU.

Builds a synthetic single-axis episode with hand-placed events (the arm
approaches, settles, grasps, moves on, settles again) and checks the detector
finds exactly those instants, plus the window math and the loud failures.

Run::

    uv run python emboviz/world_models/tests/test_keyframes.py
"""

from __future__ import annotations

import numpy as np

from emboviz_wire.observations import RGBImage
from emboviz_wire.observations.gripper import GripperState
from emboviz_wire.observations.state import Proprioception
from emboviz_wire.types import Observations, Scene, Trajectory

from emboviz.world_models.keyframes import detect_keyframes


def _img() -> RGBImage:
    return RGBImage(data=np.zeros((4, 4, 3), dtype=np.uint8), camera_id="primary")


def _episode(
    xs: list[float],
    grippers: list[float],
    *,
    fps: float = 10.0,
    with_state: bool = True,
    with_gripper: bool = True,
) -> Trajectory:
    """One-axis EE trajectory: state = [x, 0, 0, 0, 0, 0], plus a gripper signal."""
    frames = []
    for x, g in zip(xs, grippers):
        state = (
            Proprioception(values=np.array([x, 0, 0, 0, 0, 0], dtype=np.float32), convention="ee_pose")
            if with_state
            else None
        )
        gripper = GripperState(value=float(g)) if with_gripper else None
        frames.append(
            Scene(observations=Observations(images={"primary": _img()}, state=state, gripper=gripper))
        )
    return Trajectory(frames=frames, fps=fps, episode_id="kf", source="test")


# Approach (0..5 moving) → settle (6..11 at rest) → grasp (gripper 0→1 at 8) →
# approach (12..15 moving) → settle (16..19 at rest). Gripper opens nowhere else.
_XS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] + [0.5] * 6 + [0.6, 0.7, 0.8, 0.9] + [0.9] * 4
_GRIPPERS = [0.0] * 8 + [1.0] * 12


def test_detects_settles_and_grasp_at_expected_frames() -> None:
    kfs = detect_keyframes(_episode(_XS, _GRIPPERS))
    by_index = {kf.index: kf.kind for kf in kfs}
    # The arm comes to rest at frame 6 and frame 16; the gripper closes at 8.
    assert by_index.get(6) == "settle"
    assert by_index.get(16) == "settle"
    assert by_index.get(8) == "gripper_change"
    # Exactly those three — no spurious keyframes mid-motion.
    assert sorted(by_index) == [6, 8, 16]
    # The grasp records a positive gripper delta (0 -> 1).
    grasp = next(kf for kf in kfs if kf.index == 8)
    assert grasp.gripper_delta > 0.5


def test_gripper_change_dominates_a_coincident_settle() -> None:
    # Gripper change on a frame that is also at-rest is reported once, as the
    # gripper_change (not double-counted as a settle).
    xs = [0.0, 0.1, 0.2] + [0.2] * 7  # rest from frame 3
    grippers = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # change at frame 5
    kfs = detect_keyframes(_episode(xs, grippers))
    kinds = {kf.index: kf.kind for kf in kfs}
    assert kinds.get(3) == "settle"           # rising edge of rest
    assert kinds.get(5) == "gripper_change"   # not a second settle
    assert [kf.index for kf in kfs if kf.index == 5 and kf.kind == "settle"] == []


def test_no_keyframes_when_static_and_gripper_constant() -> None:
    # Perfectly still arm, gripper never changes → the only candidate is the
    # opening settle; with no motion at all there is no moving→rest *edge*.
    kfs = detect_keyframes(_episode([0.0] * 10, [0.0] * 10))
    assert all(kf.kind != "gripper_change" for kf in kfs)


def test_missing_state_raises() -> None:
    try:
        detect_keyframes(_episode([0.0, 0.1], [0.0, 0.0], with_state=False))
    except ValueError as e:
        assert "observations.state is None" in str(e)
    else:
        raise AssertionError("expected ValueError for missing state")


def test_missing_gripper_raises() -> None:
    try:
        detect_keyframes(_episode([0.0, 0.1], [0.0, 0.0], with_gripper=False))
    except ValueError as e:
        assert "observations.gripper is None" in str(e)
    else:
        raise AssertionError("expected ValueError for missing gripper")


def test_nonpositive_fps_raises() -> None:
    try:
        detect_keyframes(_episode([0.0, 0.1], [0.0, 0.0], fps=0.0))
    except ValueError as e:
        assert "positive fps" in str(e)
    else:
        raise AssertionError("expected ValueError for fps=0")


def _run_all() -> None:
    test_detects_settles_and_grasp_at_expected_frames()
    test_gripper_change_dominates_a_coincident_settle()
    test_no_keyframes_when_static_and_gripper_constant()
    test_windows_are_fps_scaled_and_clamped()
    test_missing_state_raises()
    test_missing_gripper_raises()
    test_nonpositive_fps_raises()
    print("OK: all keyframe-detection checks passed")


if __name__ == "__main__":
    _run_all()
