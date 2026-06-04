"""Action bridge — drive Cosmos forward dynamics from a user's policy.

The stress test runs the *user's* policy at a critical moment and renders the
consequence. Cosmos forward dynamics conditions on its own ``droid_lerobot``
representation (10-D normalized ``[pos_delta, rot6d_delta, gripper]`` derived from
end-effector *state*), but a policy emits an *action*. This module converts a
policy's predicted action chunk into Cosmos conditioning by:

  1. integrating the chunk into a sequence of absolute end-effector poses, from
     the seed frame's real state, under an **explicitly declared** action
     convention (never inferred — a guessed convention silently corrupts every
     rendered frame), then
  2. encoding that pose sequence with the shared DROID encoder
     (:func:`emboviz_cosmos3.domains.encode_droid_states`).

Supported conventions (the policy's action-chunk row layout, all 7-D
``[..., gripper]`` with gripper in ``[0, 1]``):

  * ``"absolute_xyz_euler"`` — ``[x, y, z, roll, pitch, yaw, gripper]``: each row
    is the absolute next end-effector pose in the DROID cartesian convention.
  * ``"delta_xyz_euler_base"`` — ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``:
    each row is a base-frame pose delta; position adds, rotation pre-multiplies
    (``R_next = dR @ R_cur``).

Other conventions raise — add them explicitly rather than approximating.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from emboviz_wire.types import Scene, Trajectory

from emboviz_cosmos3._cosmos_action import convert_rotation
from emboviz_cosmos3.domains import encode_droid_states

ActionConvention = Literal["absolute_xyz_euler", "delta_xyz_euler_base"]
_POLICY_ROW_DIM = 7  # [pose/delta (6), gripper (1)]


def integrate_policy_chunk(
    seed_state_xyz_euler: np.ndarray,
    chunk: np.ndarray,
    convention: ActionConvention,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a policy action chunk into a cartesian-state sequence.

    Returns ``(states (T+1, 6), grippers (T,))`` where ``states[0]`` is the seed
    pose and each subsequent row is the end-effector pose after applying one
    action row, under ``convention``. ``grippers`` are the chunk's gripper column.
    """
    seed = np.asarray(seed_state_xyz_euler, dtype=np.float32).reshape(-1)[:6]
    if seed.shape[0] != 6:
        raise ValueError(f"seed state must be >=6-D [xyz, euler]; got {seed.shape}.")
    rows = np.asarray(chunk, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != _POLICY_ROW_DIM:
        raise ValueError(
            f"policy action chunk must be (T, {_POLICY_ROW_DIM}) "
            f"[pose-or-delta(6), gripper]; got {rows.shape}."
        )

    grippers = rows[:, 6].astype(np.float32)
    states = [seed.copy()]

    if convention == "absolute_xyz_euler":
        for row in rows:
            states.append(row[:6].astype(np.float32))
    elif convention == "delta_xyz_euler_base":
        cur_xyz = seed[:3].copy()
        cur_R = _euler_to_matrix(seed[3:6])
        for row in rows:
            cur_xyz = cur_xyz + row[:3]
            cur_R = _euler_to_matrix(row[3:6]) @ cur_R
            states.append(np.concatenate([cur_xyz, _matrix_to_euler(cur_R)]).astype(np.float32))
    else:
        raise ValueError(
            f"unsupported action convention {convention!r}; supported: "
            "'absolute_xyz_euler', 'delta_xyz_euler_base'. Declare the policy's "
            "action convention explicitly — it is never inferred."
        )

    return np.stack(states).astype(np.float32), grippers


def policy_chunk_to_cosmos(
    seed_state_xyz_euler: np.ndarray,
    chunk: np.ndarray,
    convention: ActionConvention,
) -> np.ndarray:
    """Convert a policy action chunk into Cosmos ``droid_lerobot`` conditioning.

    Returns ``(T, 10)`` normalized actions ready for forward dynamics.
    """
    states, grippers = integrate_policy_chunk(seed_state_xyz_euler, chunk, convention)
    return encode_droid_states(states, grippers)


def policy_action_source(
    predict_fn: Callable[[Scene], "object"],
    *,
    convention: ActionConvention,
) -> Callable[[Trajectory, int, int], np.ndarray]:
    """Build a stress-test action source driven by the user's policy.

    ``predict_fn`` maps a :class:`Scene` to an ``ActionResult`` (e.g. a connected
    ``VLAModel`` client's ``predict``). The source runs the policy on the seed
    frame, takes the first ``n_actions`` rows of its action chunk, and bridges
    them into Cosmos conditioning via ``convention``. The seed frame must carry
    end-effector state (the integration anchor); a missing chunk or state raises.
    """

    def source(traj: Trajectory, seed_index: int, n_actions: int) -> np.ndarray:
        seed = traj.frames[seed_index]
        if seed.observations.state is None:
            raise ValueError(
                f"policy bridge needs end-effector state at seed frame {seed_index} "
                "to integrate the policy's actions, but observations.state is None."
            )
        result = predict_fn(seed)
        chunk = getattr(result, "action_chunk", None)
        if chunk is None:
            raise ValueError(
                "policy bridge needs a multi-step action_chunk to roll a critical "
                "moment, but the policy returned a single action (action_chunk is "
                "None). Use a chunk-predicting policy, or render a single step."
            )
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape[0] < n_actions:
            raise ValueError(
                f"policy predicted a chunk of {chunk.shape[0]} steps but the stress "
                f"rollout needs {n_actions}; lower --n-actions or use a longer chunk."
            )
        seed_state = np.asarray(seed.observations.state.values, dtype=np.float32)
        return policy_chunk_to_cosmos(seed_state, chunk[:n_actions], convention)

    return source


def _euler_to_matrix(euler_xyz: np.ndarray) -> np.ndarray:
    return np.asarray(
        convert_rotation(np.asarray(euler_xyz, dtype=np.float32).reshape(1, 3), "euler_xyz", "matrix"),
        dtype=np.float32,
    )[0]


def _matrix_to_euler(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(
        convert_rotation(np.asarray(matrix, dtype=np.float32).reshape(1, 3, 3), "matrix", "euler_xyz"),
        dtype=np.float32,
    )[0]


__all__ = [
    "ActionConvention",
    "integrate_policy_chunk",
    "policy_action_source",
    "policy_chunk_to_cosmos",
]
