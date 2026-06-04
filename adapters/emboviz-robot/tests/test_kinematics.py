"""Forward-kinematics tests for emboviz-robot.

The load-bearing test is :func:`test_fk_matches_droid_cartesian`: the Franka
Panda FK is checked against real DROID ``cartesian_position`` values recorded
on a physical robot (episode 312, the marker task). DROID logs the
``panda_link8`` flange pose in the base frame, computed by its own forward
kinematics; reproducing it to sub-millimeter from ``joint_position`` proves the
model, frame, and Euler convention are all correct — not just internally
consistent. The reference rows are embedded so the test needs no dataset
download.

Run::

    uv run --with pin --with robot_descriptions --with scipy \
        python adapters/emboviz-robot/tests/test_kinematics.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from emboviz_robot import available_robots, load_kinematics
from emboviz_robot.kinematics import RobotKinematics


# (joint_position (7), cartesian_position (6) = [xyz, extrinsic-xyz euler]) —
# verbatim from DAVIAN-Robotics/droid_v3 episode 312 (first / middle / last
# frame). cartesian_position is the panda_link8 pose in panda_link0.
_DROID_EP312 = [
    ([0.09163462, -0.28839064, 0.24611428, -2.0873816, -0.16040711, 1.57743382, -0.01140388],
     [0.42353743, 0.14038895, 0.52755874, 2.91282821, 0.23199207, 0.3242442]),
    ([0.03826781, 0.16535784, 0.03189354, -2.50998282, 0.51357657, 2.70578718, -1.46648145],
     [0.46695399, 0.01616935, 0.21885765, -3.09261584, -0.22355442, 1.06631815]),
    ([-0.01355032, 0.33219197, 0.20537661, -1.95068038, -0.25896466, 2.21371841, -0.94551814],
     [0.59823823, 0.10253862, 0.30780461, 3.11983466, 0.14557543, 1.2886461]),
]


def _rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    dR = R_a.T @ R_b
    return float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))


def test_fk_matches_droid_cartesian() -> None:
    kin = load_kinematics("franka_panda")
    assert kin.ee_frame == "panda_link8"
    assert kin.n_joints == 7

    for joints, cart in _DROID_EP312:
        pose = kin.fk(np.array(joints))
        xyz, euler = pose.as_xyz_euler()           # extrinsic xyz (DROID convention)
        pos_err = np.linalg.norm(xyz - np.array(cart[:3]))
        rot_err = _rotation_error_deg(pose.rotation, R.from_euler("xyz", cart[3:6]).as_matrix())
        assert pos_err < 1e-3, f"position error {pos_err*1000:.3f} mm too large"
        assert rot_err < 0.1, f"rotation error {rot_err:.4f} deg too large"


def test_fk_is_deterministic_and_batch_consistent() -> None:
    kin = load_kinematics("panda")                 # alias resolves to franka_panda
    q = np.array([j for j, _ in _DROID_EP312][0])
    a, b = kin.fk(q), kin.fk(q)
    assert np.array_equal(a.translation, b.translation)
    assert np.array_equal(a.rotation, b.rotation)

    qs = np.array([j for j, _ in _DROID_EP312])
    batch = kin.fk_batch(qs)
    assert len(batch) == len(qs)
    for single, row in zip(batch, qs):
        ref = kin.fk(row)
        assert np.allclose(single.translation, ref.translation)
        assert np.allclose(single.rotation, ref.rotation)


def test_custom_urdf_path_matches_catalog() -> None:
    """The custom-URDF path produces identical FK to the catalog path."""
    from robot_descriptions import panda_description

    custom = RobotKinematics(
        panda_description.URDF_PATH,
        ee_frame="panda_link8",
        joint_names=[f"panda_joint{i}" for i in range(1, 8)],
    )
    catalog = load_kinematics("franka_panda")
    q = np.array([j for j, _ in _DROID_EP312][1])
    assert np.allclose(custom.fk(q).translation, catalog.fk(q).translation)
    assert np.allclose(custom.fk(q).rotation, catalog.fk(q).rotation)


def test_wrong_joint_count_raises() -> None:
    kin = load_kinematics("franka_panda")
    try:
        kin.fk(np.zeros(6))
    except ValueError as e:
        assert "expected 7 joint values" in str(e)
    else:
        raise AssertionError("expected ValueError for wrong joint count")


def test_unknown_robot_and_bad_args_raise() -> None:
    for kwargs in ({"robot": "nonesuch"}, {}, {"robot": "panda", "urdf": "/x.urdf"}):
        try:
            load_kinematics(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
    assert "franka_panda" in available_robots()


def test_unknown_frame_raises() -> None:
    from robot_descriptions import panda_description

    try:
        RobotKinematics(panda_description.URDF_PATH, "no_such_frame",
                        [f"panda_joint{i}" for i in range(1, 8)])
    except ValueError as e:
        assert "not in the reduced model" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown ee_frame")


def _run_all() -> None:
    test_fk_matches_droid_cartesian()
    test_fk_is_deterministic_and_batch_consistent()
    test_custom_urdf_path_matches_catalog()
    test_wrong_joint_count_raises()
    test_unknown_robot_and_bad_args_raise()
    test_unknown_frame_raises()
    print("OK: all emboviz-robot kinematics checks passed")


if __name__ == "__main__":
    _run_all()
