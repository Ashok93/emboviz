"""Tests for the shared policy bridge — rotation helpers + trackers.

The pure-numpy extrinsic-XYZ euler helpers must agree with scipy's
``Rotation.from_euler("xyz")`` / ``as_euler("xyz")`` (the DROID convention's
reference implementation); the trackers' integrate/advance split must keep the
prediction and execution horizons independent.

Run::

    uv run python adapters/emboviz-wire/tests/test_policy_bridge.py
"""

from __future__ import annotations

import numpy as np
import pytest

from emboviz_wire.policy_bridge import (
    CartesianStateTracker,
    euler_xyz_to_matrix,
    integrate_policy_chunk,
    matrix_to_euler_xyz,
)

scipy_rotation = pytest.importorskip(
    "scipy.spatial.transform", reason="scipy is the reference for the euler convention"
)
R = scipy_rotation.Rotation


def test_euler_matrix_roundtrip_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        euler = rng.uniform(-np.pi, np.pi, size=3)
        euler[1] *= 0.49  # keep pitch away from the gimbal singularity
        m_ours = euler_xyz_to_matrix(euler)
        m_scipy = R.from_euler("xyz", euler).as_matrix()
        np.testing.assert_allclose(m_ours, m_scipy, atol=1e-12)

        e_ours = matrix_to_euler_xyz(m_scipy)
        e_scipy = R.from_matrix(m_scipy).as_euler("xyz")
        np.testing.assert_allclose(e_ours, e_scipy, atol=1e-9)


def test_matrix_to_euler_rejects_bad_shape() -> None:
    try:
        matrix_to_euler_xyz(np.eye(4))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a non-3x3 matrix")


def test_cartesian_tracker_integrate_is_pure_and_advance_commits() -> None:
    seed = np.array([0.4, 0.1, 0.5, 0.0, 0.0, 0.0], np.float32)
    tracker = CartesianStateTracker(seed, 0.2, "delta_xyz_euler_base")
    chunk = np.zeros((4, 7), np.float32)
    chunk[:, 0] = 0.01      # +1 cm in x per row
    chunk[:, 6] = 0.8

    states, grippers = tracker.integrate(chunk, 4)
    assert states.shape == (5, 6) and grippers.shape == (4,)
    np.testing.assert_allclose(states[:, 0], 0.4 + 0.01 * np.arange(5), atol=1e-6)
    # integrate() is pure — the tracked state did not move.
    np.testing.assert_allclose(tracker.state, seed, atol=1e-7)

    tracker.advance(chunk, 2)   # commit only 2 of the 4 integrated rows
    np.testing.assert_allclose(tracker.state[0], 0.42, atol=1e-6)
    assert tracker.gripper == np.float32(0.8)


def test_delta_integration_composes_rotations_like_scipy() -> None:
    seed = np.array([0.0, 0.0, 0.0, 0.1, -0.2, 0.3], np.float32)
    delta = np.array([[0.0, 0.0, 0.0, 0.05, 0.04, -0.03, 0.5]], np.float32)
    states, _ = integrate_policy_chunk(seed, delta, "delta_xyz_euler_base")
    expected = (
        R.from_euler("xyz", delta[0, 3:6]).as_matrix() @ R.from_euler("xyz", seed[3:6]).as_matrix()
    )
    np.testing.assert_allclose(
        R.from_euler("xyz", states[1, 3:6]).as_matrix(), expected, atol=1e-6
    )


def _run_all() -> None:
    test_euler_matrix_roundtrip_matches_scipy()
    test_matrix_to_euler_rejects_bad_shape()
    test_cartesian_tracker_integrate_is_pure_and_advance_commits()
    test_delta_integration_composes_rotations_like_scipy()
    print("OK: all policy-bridge checks passed")


if __name__ == "__main__":
    _run_all()
