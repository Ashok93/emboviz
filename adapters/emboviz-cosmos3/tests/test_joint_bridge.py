"""End-to-end joint-bridge test with REAL forward kinematics (needs pinocchio).

Validates the ``droid_joint_delta`` path against physical DROID data: feeding the
bridge the joint *deltas* between consecutive episode-312 frames must (a) recover
the recorded joint configurations and (b) forward-kinematics them to the recorded
``cartesian_position`` — the exact poses the Cosmos DROID encoder conditions on.
This ties ``integrate_joint_chunk`` + ``JointStateTracker`` to ground truth, not
just to internal consistency.

Run::

    PYTHONPATH=adapters/emboviz-robot uv run --with pin --with robot_descriptions \
        --with scipy python adapters/emboviz-cosmos3/tests/test_joint_bridge.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from emboviz_robot import load_kinematics

from emboviz_cosmos3.bridge import (
    JointStateTracker,
    integrate_joint_chunk,
    make_state_tracker,
)
from emboviz_cosmos3.domains import encode_droid_states

# Six consecutive frames of DAVIAN-Robotics/droid_v3 episode 312.
_JOINTS = [
    [0.09163462, -0.28839064, 0.24611428, -2.0873816, -0.16040711, 1.57743382, -0.01140388],
    [0.09163336, -0.28838104, 0.24611428, -2.08738327, -0.16040893, 1.57744324, -0.01141115],
    [0.0916292, -0.28837618, 0.24610519, -2.08738327, -0.16040663, 1.57744741, -0.0114025],
    [0.09162848, -0.28838098, 0.24611066, -2.08737969, -0.16040337, 1.57745183, -0.01140548],
    [0.09163015, -0.28836593, 0.24611063, -2.08738542, -0.16040343, 1.57745337, -0.01141166],
    [0.09163015, -0.28836557, 0.24611066, -2.08737803, -0.1604054, 1.57745779, -0.01141085],
]
_CART = [
    [0.42353743, 0.14038895, 0.52755874, 2.91282821, 0.23199207, 0.3242442],
    [0.42354041, 0.14038883, 0.52755439, 2.91283059, 0.23199555, 0.32425281],
    [0.42354357, 0.14038342, 0.52755272, 2.91283488, 0.23199363, 0.32423311],
    [0.42354277, 0.14038639, 0.52755648, 2.91283631, 0.2319821, 0.32424125],
    [0.42354497, 0.14038701, 0.52754742, 2.91284132, 0.23200235, 0.32424986],
    [0.42354646, 0.14038762, 0.52755094, 2.91283941, 0.23199086, 0.32425064],
]


def _rot_err_deg(R_a, R_b):
    dR = R_a.T @ R_b
    return float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))


def test_integrate_joint_chunk_reproduces_recorded_states() -> None:
    kin = load_kinematics("franka_panda")
    joints = np.array(_JOINTS)
    seed = joints[0]
    deltas = np.diff(joints, axis=0)                    # 5 joint-delta rows
    gripper = np.full((len(deltas), 1), 0.0, dtype=np.float32)
    chunk = np.concatenate([deltas, gripper], axis=1)   # (5, 8)

    joint_states, cartesian_states, grippers = integrate_joint_chunk(seed, chunk, kin)

    # Integrated joint configs recover the recorded ones.
    assert np.allclose(joint_states, joints, atol=1e-6)
    # FK of each recovered config matches the recorded cartesian_position.
    for i in range(len(joints)):
        assert np.linalg.norm(cartesian_states[i, :3] - np.array(_CART[i][:3])) < 1e-3
        R_fk = R.from_euler("xyz", cartesian_states[i, 3:6]).as_matrix()
        R_data = R.from_euler("xyz", _CART[i][3:6]).as_matrix()
        assert _rot_err_deg(R_fk, R_data) < 0.1
    assert grippers.shape == (5,)


def test_joint_tracker_to_cosmos_and_factory() -> None:
    kin = load_kinematics("franka_panda")
    tracker = make_state_tracker(
        np.array(_JOINTS[0]), 0.0,
        action_convention="droid_joint_delta", kinematics=kin,
    )
    assert isinstance(tracker, JointStateTracker)
    assert tracker.proprioception().convention == "joint_angles"

    deltas = np.diff(np.array(_JOINTS), axis=0)
    chunk = np.concatenate([deltas, np.full((5, 1), 0.5, np.float32)], axis=1)
    cosmos = tracker.to_cosmos(chunk, n_actions=5)
    assert cosmos.shape == (5, 10)                       # Cosmos droid_lerobot conditioning
    assert np.allclose(tracker.joints, _JOINTS[-1], atol=1e-6)   # advanced to the last config


def test_joint_path_matches_recorded_cartesian_encoder() -> None:
    """The joint bridge yields the SAME Cosmos conditioning as encoding the
    recorded cartesian_position directly — the two paths agree on real data."""
    kin = load_kinematics("franka_panda")
    joints = np.array(_JOINTS)
    deltas = np.diff(joints, axis=0)
    grip = np.full(len(deltas), 0.3, dtype=np.float32)
    chunk = np.concatenate([deltas, grip[:, None]], axis=1)

    _, cartesian_states, grippers = integrate_joint_chunk(joints[0], chunk, kin)
    via_joints = encode_droid_states(cartesian_states, grippers)

    # The cartesian path on the dataset's own recorded cartesian_position.
    via_cartesian = encode_droid_states(np.array(_CART, dtype=np.float32), grip)
    assert np.allclose(via_joints, via_cartesian, atol=1e-3), np.abs(via_joints - via_cartesian).max()


def test_factory_rejects_mismatched_robot_and_convention() -> None:
    for kwargs in (
        dict(action_convention="droid_joint_delta", kinematics=None),
        dict(action_convention="absolute_xyz_euler", kinematics=load_kinematics("franka_panda")),
    ):
        try:
            make_state_tracker(np.zeros(7, np.float32), 0.0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def _run_all() -> None:
    test_integrate_joint_chunk_reproduces_recorded_states()
    test_joint_tracker_to_cosmos_and_factory()
    test_joint_path_matches_recorded_cartesian_encoder()
    test_factory_rejects_mismatched_robot_and_convention()
    print("OK: all joint-bridge (real FK) checks passed")


if __name__ == "__main__":
    _run_all()
