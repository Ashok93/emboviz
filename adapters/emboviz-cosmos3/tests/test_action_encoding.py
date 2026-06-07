"""Tests for the vendored Cosmos action math and the DROID action builder.

Validates (a) the vendored numpy/scipy pose math against mathematical identities
and scipy directly, and (b) that the DROID domain builder encodes an episode into
the 10-D ``[pos_delta(3), rot6d_delta(6), gripper(1)]`` quantile-normalized form,
reproducing ``DROIDLeRobotDataset._build_raw_action``.

A static episode (the robot does not move) must encode to the identity pose delta:
zero translation and rot6d ``[1,0,0,0,1,0]`` — a clean, diagnostic check that the
whole pipeline (abs pose → DROID→OpenCV → relative delta → rot6d → normalize) is
wired correctly.

Run::

    uv run --with scipy --with numpy python \
        adapters/emboviz-cosmos3/tests/test_action_encoding.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from emboviz_wire.observations import GripperState, Proprioception, RGBImage
from emboviz_wire.types import Observations, Scene, Trajectory

from emboviz_cosmos3._cosmos_action import (
    build_abs_pose_from_components,
    convert_rotation,
    pose_abs_to_rel,
)
from emboviz_cosmos3 import domains


# ── vendored pose math ───────────────────────────────────────────────────────


def test_euler_identity_to_rot6d() -> None:
    # Zero euler → identity rotation → rot6d [1,0,0,0,1,0].
    rot6d = convert_rotation(np.zeros((1, 3), np.float32), "euler_xyz", "rot6d")[0]
    assert np.allclose(rot6d, [1, 0, 0, 0, 1, 0], atol=1e-5)


def test_rot6d_matrix_roundtrip_matches_scipy() -> None:
    rng = np.random.RandomState(0)
    eulers = rng.uniform(-1.0, 1.0, size=(5, 3)).astype(np.float32)
    mats_ref = R.from_euler("xyz", eulers).as_matrix().astype(np.float32)
    mats = convert_rotation(eulers, "euler_xyz", "matrix")
    assert np.allclose(mats, mats_ref, atol=1e-5)
    # matrix → rot6d → matrix is a faithful round trip.
    rot6d = convert_rotation(mats, "matrix", "rot6d")
    back = convert_rotation(rot6d, "rot6d", "matrix")
    assert np.allclose(back, mats, atol=1e-5)


def test_static_trajectory_has_identity_delta() -> None:
    # Two identical poses → relative delta is identity: zero translation, rot6d id.
    xyz = np.zeros((2, 3), np.float32)
    euler = np.zeros((2, 3), np.float32)
    poses = build_abs_pose_from_components(xyz, euler, "euler_xyz")
    rel = pose_abs_to_rel(poses, rotation_format="rot6d", pose_convention="backward_framewise")
    assert rel.shape == (1, 9)
    assert np.allclose(rel[0], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-5)


# ── DROID domain builder ─────────────────────────────────────────────────────


def _droid_episode(n_frames: int, *, gripper: float = 0.5, moving: bool = False) -> Trajectory:
    """An episode with cartesian state [xyz, euler] + gripper per frame."""
    scenes = []
    img = np.zeros((8, 8, 3), np.uint8)
    for k in range(n_frames):
        x = 0.008 * k if moving else 0.0  # +x drift when moving (within the q99 range)
        state = np.array([x, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        scenes.append(Scene(
            observations=Observations(
                images={"primary": RGBImage(data=img, camera_id="primary")},
                state=Proprioception(values=state, convention="ee_pose"),
                gripper=GripperState(value=gripper, kind="parallel_jaw", units="normalized"),
            ),
        ))
    return Trajectory(frames=scenes, frame_indices=list(range(n_frames)), fps=15.0,
                      episode_id="droid", source="test", metadata={})


def test_droid_static_encodes_identity_delta() -> None:
    ep = _droid_episode(6, gripper=0.5)  # not moving
    actions = domains.prepare_actions("droid_lerobot", ep, frame_start=0, n_actions=5)
    assert actions.shape == (5, 10)
    assert actions.dtype == np.float32
    assert np.all(actions >= -1.0001) and np.all(actions <= 1.0001)  # quantile range
    # Identity delta after normalization: rot6d-identity dims (idx 3, 7) → ~1;
    # translation + off-diagonal rot6d dims → ~0; gripper (idx 9): 1-2*0.5 = 0.
    a = actions[0]
    assert abs(a[3] - 1.0) < 1e-3 and abs(a[7] - 1.0) < 1e-3
    # Off-block dims map near 0, but the asymmetric q01/q99 shift "zero" slightly.
    for i in (0, 1, 2, 4, 5, 6, 8):
        assert abs(a[i]) < 0.12, (i, a[i])
    assert abs(a[9]) < 1e-3  # gripper 0.5 → 0


def test_droid_gripper_inversion() -> None:
    # gripper=0.0 → 1-g=1 → normalized (q01=0,q99=1): 2*1-1 = +1.
    a = domains.prepare_actions("droid_lerobot", _droid_episode(4, gripper=0.0), frame_start=0, n_actions=3)
    assert abs(a[0, 9] - 1.0) < 1e-3
    # gripper=1.0 → 1-g=0 → 2*0-1 = -1.
    b = domains.prepare_actions("droid_lerobot", _droid_episode(4, gripper=1.0), frame_start=0, n_actions=3)
    assert abs(b[0, 9] + 1.0) < 1e-3


def test_droid_motion_changes_translation() -> None:
    moving = domains.prepare_actions("droid_lerobot", _droid_episode(6, moving=True), frame_start=0, n_actions=5)
    static = domains.prepare_actions("droid_lerobot", _droid_episode(6, moving=False), frame_start=0, n_actions=5)
    # Motion shows up in the (normalized) translation block; static stays ~0 there.
    assert np.linalg.norm(moving[:, 0:3]) > np.linalg.norm(static[:, 0:3]) + 0.05


def test_droid_needs_enough_frames() -> None:
    try:
        domains.prepare_actions("droid_lerobot", _droid_episode(3), frame_start=0, n_actions=5)
    except IndexError as e:
        assert "frames" in str(e)
    else:
        raise AssertionError("expected IndexError for too few frames")


def test_missing_state_raises() -> None:
    cam = "primary"
    ep = Trajectory(
        frames=[Scene(observations=Observations(images={cam: RGBImage(data=np.zeros((8, 8, 3), np.uint8), camera_id=cam)}))] * 3,
        frame_indices=[0, 1, 2], fps=15.0, episode_id="x", source="t", metadata={},
    )
    try:
        domains.prepare_actions("droid_lerobot", ep, frame_start=0, n_actions=2)
    except ValueError as e:
        assert "state" in str(e)
    else:
        raise AssertionError("expected ValueError for missing state")


def test_unimplemented_domain_raises() -> None:
    try:
        domains.prepare_actions("agibotworld", _droid_episode(4), frame_start=0, n_actions=3)
    except NotImplementedError as e:
        assert "agibotworld" in str(e)
    else:
        raise AssertionError("expected NotImplementedError for unimplemented domain")


# ── adapter override (Cosmos3WorldModel.prepare_actions) ─────────────────────


def test_adapter_prepare_actions_uses_domain() -> None:
    from emboviz_cosmos3.model import Cosmos3WorldModel

    wm = Cosmos3WorldModel(
        server_url="http://localhost:8000", domain_name="droid_lerobot", action_dim=10,
    )
    actions = wm.prepare_actions(_droid_episode(6), frame_start=0, n_actions=5)
    assert actions.shape == (5, 10)
    # n_actions=None → as many as the episode supports (n_frames - frame_start - 1).
    assert wm.prepare_actions(_droid_episode(6)).shape == (5, 10)


def test_adapter_prepare_actions_action_dim_mismatch_raises() -> None:
    from emboviz_cosmos3.model import Cosmos3WorldModel

    wm = Cosmos3WorldModel(
        server_url="http://localhost:8000", domain_name="droid_lerobot", action_dim=7,
    )
    try:
        wm.prepare_actions(_droid_episode(6), n_actions=5)
    except ValueError as e:
        assert "action_dim" in str(e)
    else:
        raise AssertionError("expected ValueError for action_dim/domain mismatch")


def _run_all() -> None:
    test_euler_identity_to_rot6d()
    test_rot6d_matrix_roundtrip_matches_scipy()
    test_static_trajectory_has_identity_delta()
    test_droid_static_encodes_identity_delta()
    test_droid_gripper_inversion()
    test_droid_motion_changes_translation()
    test_droid_needs_enough_frames()
    test_missing_state_raises()
    test_unimplemented_domain_raises()
    test_adapter_prepare_actions_uses_domain()
    test_adapter_prepare_actions_action_dim_mismatch_raises()
    print("OK: all cosmos action-encoding checks passed")


if __name__ == "__main__":
    _run_all()
