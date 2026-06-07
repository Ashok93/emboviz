"""Per-embodiment action encoding for Cosmos forward dynamics.

Each Cosmos robot *domain* conditions on actions in its own representation, so a
recorded episode's state/actions must be encoded into that domain's format before
the world model can roll it forward. This module is the bridge between an emboviz
:class:`Trajectory` and a Cosmos domain's action vectors, using the vendored
NVIDIA encoding in :mod:`emboviz_cosmos3._cosmos_action`.

Only the domains implemented here are supported; an unimplemented domain raises
rather than silently passing raw actions (which the model was not trained on).

DROID (``droid_lerobot``)
-------------------------
10-D ``[pos_delta(3), rot6d_delta(6), gripper(1)]``, reproducing
``DROIDLeRobotDataset._build_raw_action`` exactly:

  1. Build absolute end-effector poses from the cartesian state
     (``observations.state.values[:6]`` = ``[xyz, euler_xyz]``).
  2. Apply the DROID→OpenCV frame rotation.
  3. Encode frame-to-frame relative deltas as ``[pos(3), rot6d(6)]``
     (``backward_framewise``).
  4. Append ``1 − gripper`` from ``observations.gripper.value``.
  5. Quantile-normalize with the shipped DROID stats.

Input contract (the dataset config must satisfy it; nothing is inferred):
  * ``observations.state`` present, ≥6-D, laid out as ``[x, y, z, roll, pitch,
    yaw]`` (the DROID ``observation.state.cartesian_position`` convention).
  * ``observations.gripper`` present, value in ``[0, 1]`` (the DROID
    ``action.gripper_position`` convention).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from emboviz_wire.types import Trajectory

from emboviz_cosmos3._cosmos_action import (
    build_abs_pose_from_components,
    load_action_stats,
    normalize_action,
    pose_abs_to_rel,
)

# 90° clockwise rotation about the local Z axis — the production DROID wrapper's
# Franka panda_link8 → OpenCV conversion (cosmos-framework droid_lerobot_dataset.py).
_DROID_TO_OPENCV = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
)
_DROID_STATS_PATH = Path(__file__).parent / "_cosmos_action" / "droid_lerobot_normalization.json"

#: Action dimensionality per implemented domain — validated against the adapter's
#: configured action_dim so a mismatch fails loudly at startup.
ACTION_DIMS: dict[str, int] = {
    "droid_lerobot": 10,
}


def prepare_actions(
    domain_name: str, episode: Trajectory, *, frame_start: int, n_actions: int
) -> np.ndarray:
    """Encode ``n_actions`` actions for ``domain_name`` from ``episode``.

    Returns ``(n_actions, action_dim)`` float32 actions in the domain's normalized
    representation, conditioning the rollout that starts at ``frame_start``.
    """
    builder = _BUILDERS.get(domain_name)
    if builder is None:
        raise NotImplementedError(
            f"cosmos3: action encoding for domain '{domain_name}' is not implemented. "
            f"Implemented: {sorted(_BUILDERS)}. Each Cosmos embodiment encodes actions "
            "differently; add its builder rather than passing raw actions the model "
            "was not trained on."
        )
    if n_actions < 1:
        raise ValueError(f"n_actions must be >= 1, got {n_actions}")
    return builder(episode, frame_start, n_actions)


def _state_xyz_euler(episode: Trajectory, idx: int) -> np.ndarray:
    state = episode.frames[idx].observations.state
    if state is None:
        raise ValueError(
            f"droid_lerobot action encoding needs proprioceptive state at frame {idx}, "
            "but observations.state is None. Map the dataset's cartesian state "
            "(observation.state.cartesian_position) in the config."
        )
    values = np.asarray(state.values, dtype=np.float32)
    if values.ndim != 1 or values.shape[0] < 6:
        raise ValueError(
            f"droid_lerobot needs a >=6-D cartesian state [xyz, euler_xyz] at frame "
            f"{idx}; got shape {values.shape}."
        )
    return values[:6]


def _gripper(episode: Trajectory, idx: int) -> float:
    gripper = episode.frames[idx].observations.gripper
    if gripper is None:
        raise ValueError(
            f"droid_lerobot action encoding needs the gripper at frame {idx}, but "
            "observations.gripper is None. Map the dataset's gripper in the config."
        )
    return float(gripper.value)


def encode_droid_states(state_xyz_euler: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Encode a cartesian-state sequence into Cosmos ``droid_lerobot`` actions.

    ``state_xyz_euler`` is ``(T+1, 6)`` end-effector poses ``[x, y, z, roll, pitch,
    yaw]`` (the DROID ``cartesian_position`` convention); ``gripper`` is ``(T,)``
    in ``[0, 1]`` aligned with the first ``T`` states. Returns ``(T, 10)`` actions
    ``[pos_delta(3), rot6d_delta(6), gripper]`` quantile-normalized — the exact
    representation ``DROIDLeRobotDataset._build_raw_action`` produces, so it is the
    single encoder shared by the recorded-episode path and the policy bridge.
    """
    state = np.asarray(state_xyz_euler, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] < 6:
        raise ValueError(f"state must be (T+1, >=6) [xyz, euler]; got {state.shape}.")
    grip = np.asarray(gripper, dtype=np.float32).reshape(-1)
    if grip.shape[0] != state.shape[0] - 1:
        raise ValueError(
            f"gripper length {grip.shape[0]} must be states-1 ({state.shape[0] - 1})."
        )

    poses_abs = build_abs_pose_from_components(state[:, 0:3], state[:, 3:6], "euler_xyz")
    poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _DROID_TO_OPENCV
    poses_rel = pose_abs_to_rel(
        poses_abs, rotation_format="rot6d", pose_convention="backward_framewise"
    )  # (T, 9)

    action = np.concatenate([poses_rel, (1.0 - grip).reshape(-1, 1)], axis=-1)  # (T, 10)
    stats = load_action_stats(str(_DROID_STATS_PATH))
    return normalize_action(action, "quantile", stats).astype(np.float32)


def _prepare_droid_lerobot(
    episode: Trajectory, frame_start: int, n_actions: int
) -> np.ndarray:
    # n_actions relative deltas require n_actions + 1 consecutive state frames.
    lo, hi = frame_start, frame_start + n_actions + 1
    if frame_start < 0 or hi > len(episode.frames):
        raise IndexError(
            f"droid_lerobot needs frames [{lo}, {hi}) (n_actions+1 states) but the "
            f"episode has {len(episode.frames)} frames."
        )

    state = np.stack([_state_xyz_euler(episode, i) for i in range(lo, hi)])  # (n+1, 6)
    gripper = np.array([_gripper(episode, i) for i in range(lo, lo + n_actions)], dtype=np.float32)
    return encode_droid_states(state, gripper)


_BUILDERS: dict[str, Callable[[Trajectory, int, int], np.ndarray]] = {
    "droid_lerobot": _prepare_droid_lerobot,
}
