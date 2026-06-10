"""Tests for the policy→Cosmos action bridge (pure numpy, no GPU/server).

The strong check: when a "policy" emits the episode's *actual* next states, the
bridge's absolute path must reproduce the gold recorded encoder bit-for-bit — so
driving Cosmos with the policy uses the identical representation the model was
trained on. The delta path is checked by round-trip (deltas rebuilt from a state
sequence integrate back to it).

Run::

    uv run --extra cosmos3 python adapters/emboviz-cosmos3/tests/test_bridge.py
"""

from __future__ import annotations

import numpy as np

from emboviz_wire.observations import RGBImage
from emboviz_wire.observations.gripper import GripperState
from emboviz_wire.observations.state import Proprioception
from emboviz_wire.types import Observations, Scene, Trajectory

from emboviz_wire.policy_bridge import integrate_policy_chunk

from emboviz_cosmos3 import domains
from emboviz_cosmos3._cosmos_action import convert_rotation
from emboviz_cosmos3.bridge import policy_chunk_to_cosmos


def _episode(states: np.ndarray, grippers: np.ndarray) -> Trajectory:
    frames = []
    for s, g in zip(states, grippers):
        frames.append(
            Scene(
                observations=Observations(
                    images={"primary": RGBImage(data=np.zeros((4, 4, 3), np.uint8), camera_id="primary")},
                    state=Proprioception(values=s.astype(np.float32), convention="ee_pose"),
                    gripper=GripperState(value=float(g)),
                )
            )
        )
    return Trajectory(frames=frames, fps=15.0, episode_id="bridge", source="test")


def _random_states(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(0)
    xyz = np.cumsum(rng.uniform(-0.01, 0.01, size=(n, 3)), axis=0)
    euler = np.cumsum(rng.uniform(-0.02, 0.02, size=(n, 3)), axis=0)
    states = np.concatenate([xyz, euler], axis=1).astype(np.float32)  # (n, 6)
    grippers = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    return states, grippers


def test_absolute_path_reproduces_recorded_encoder() -> None:
    n_states = 8
    states, grippers = _random_states(n_states)
    episode = _episode(states, grippers)
    n_actions = n_states - 1

    recorded = domains.prepare_actions("droid_lerobot", episode, frame_start=0, n_actions=n_actions)

    # A policy that "predicts" the true next states. Row i = [state_{i+1}, g_i].
    chunk = np.concatenate([states[1:], grippers[:n_actions, None]], axis=1).astype(np.float32)
    bridged = policy_chunk_to_cosmos(states[0], chunk, "absolute_xyz_euler")

    assert bridged.shape == recorded.shape == (n_actions, 10)
    assert np.allclose(bridged, recorded, atol=1e-5), np.abs(bridged - recorded).max()


def test_delta_base_path_round_trips_to_same_states() -> None:
    n_states = 6
    states, grippers = _random_states(n_states)
    n_actions = n_states - 1

    # Build base-frame deltas from the true states: d_xyz = x_{i+1}-x_i,
    # dR = R_{i+1} @ R_i^T  ->  d_euler.
    deltas = []
    for i in range(n_actions):
        d_xyz = states[i + 1, :3] - states[i, :3]
        r_i = convert_rotation(states[i, 3:6].reshape(1, 3), "euler_xyz", "matrix")[0]
        r_n = convert_rotation(states[i + 1, 3:6].reshape(1, 3), "euler_xyz", "matrix")[0]
        d_euler = convert_rotation((r_n @ r_i.T).reshape(1, 3, 3), "matrix", "euler_xyz")[0]
        deltas.append(np.concatenate([d_xyz, d_euler, [grippers[i]]]))
    chunk = np.stack(deltas).astype(np.float32)

    recovered, _ = integrate_policy_chunk(states[0], chunk, "delta_xyz_euler_base")
    # Positions recover exactly; rotations recover up to Euler representation.
    assert np.allclose(recovered[:, :3], states[:, :3], atol=1e-5)
    r_true = convert_rotation(states[:, 3:6], "euler_xyz", "matrix")
    r_rec = convert_rotation(recovered[:, 3:6], "euler_xyz", "matrix")
    assert np.allclose(r_true, r_rec, atol=1e-4)

    # And encoding the recovered states equals encoding the true states.
    enc_true = domains.encode_droid_states(states, grippers[:n_actions])
    enc_rec = domains.encode_droid_states(recovered, grippers[:n_actions])
    assert np.allclose(enc_true, enc_rec, atol=1e-4)


def test_unsupported_convention_raises() -> None:
    states, grippers = _random_states(3)
    chunk = np.concatenate([states[1:], grippers[:2, None]], axis=1).astype(np.float32)
    try:
        policy_chunk_to_cosmos(states[0], chunk, "joint_velocity")  # type: ignore[arg-type]
    except ValueError as e:
        assert "unsupported action convention" in str(e)
    else:
        raise AssertionError("expected ValueError for unsupported convention")


def test_wrong_chunk_dim_raises() -> None:
    try:
        integrate_policy_chunk(np.zeros(6, np.float32), np.zeros((4, 10), np.float32), "absolute_xyz_euler")
    except ValueError as e:
        assert "must be (T, 7)" in str(e)
    else:
        raise AssertionError("expected ValueError for wrong chunk dim")


def _run_all() -> None:
    test_absolute_path_reproduces_recorded_encoder()
    test_delta_base_path_round_trips_to_same_states()
    test_unsupported_convention_raises()
    test_wrong_chunk_dim_raises()
    print("OK: all action-bridge checks passed")


if __name__ == "__main__":
    _run_all()
