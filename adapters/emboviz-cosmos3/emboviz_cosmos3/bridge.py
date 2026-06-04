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

Supported conventions (the policy's action-chunk row layout):

Cartesian (7-D rows ``[..., gripper]``, gripper in ``[0, 1]``):

  * ``"absolute_xyz_euler"`` — ``[x, y, z, roll, pitch, yaw, gripper]``: each row
    is the absolute next end-effector pose in the DROID cartesian convention.
  * ``"delta_xyz_euler_base"`` — ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``:
    each row is a base-frame pose delta; position adds, rotation pre-multiplies
    (``R_next = dR @ R_cur``).

Joint (``[joint_delta(n), gripper(1)]`` rows; gripper absolute in ``[0, 1]``):

  * ``"droid_joint_delta"`` — the π0-DROID action space: 7 joint-position deltas
    (radians, added to the current joint configuration) plus an absolute gripper.
    The tracked state is the joint vector itself (what the policy reads as
    ``observation/joint_position``); forward kinematics maps each joint
    configuration to the ``panda_link8`` pose the DROID encoder expects. Requires
    an injected ``kinematics`` object (a :class:`emboviz_robot.RobotKinematics`),
    so this module never imports the kinematics engine — the driver wires it.

Other conventions raise — add them explicitly rather than approximating.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional, Protocol

import numpy as np

from emboviz_wire.observations.gripper import GripperState
from emboviz_wire.observations.state import Proprioception
from emboviz_wire.types import Scene, Trajectory

from emboviz_cosmos3._cosmos_action import convert_rotation
from emboviz_cosmos3.domains import encode_droid_states

ActionConvention = Literal["absolute_xyz_euler", "delta_xyz_euler_base", "droid_joint_delta"]
_CARTESIAN_CONVENTIONS = frozenset({"absolute_xyz_euler", "delta_xyz_euler_base"})
_JOINT_CONVENTIONS = frozenset({"droid_joint_delta"})
_POLICY_ROW_DIM = 7  # cartesian rows: [pose/delta (6), gripper (1)]


class Kinematics(Protocol):
    """The forward-kinematics interface the joint bridge depends on.

    Structural type satisfied by :class:`emboviz_robot.RobotKinematics`; declared
    here so this module needs no import of (and no dependency on) the kinematics
    engine. The driver constructs the concrete object and injects it.
    """

    n_joints: int

    def fk(self, q: np.ndarray) -> "object":  # returns an object with .translation/.rotation
        ...


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


def integrate_joint_chunk(
    seed_joints: np.ndarray,
    chunk: np.ndarray,
    kinematics: Kinematics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a joint-delta action chunk into joint and Cartesian sequences.

    ``chunk`` is ``(T, n_joints + 1)`` rows ``[joint_delta(n), gripper(1)]`` — the
    ``droid_joint_delta`` convention. Each joint configuration is the running sum
    of deltas from ``seed_joints`` (π0-DROID's ``use_delta_joint_actions``: the 7
    joint outputs are deltas added to the current ``joint_position``); the gripper
    column is absolute. Forward kinematics maps each configuration to the
    end-effector pose, returned as ``[xyz, euler_xyz]`` in the DROID
    ``cartesian_position`` convention (extrinsic-XYZ euler, matching
    ``encode_droid_states``).

    Returns ``(joint_states (T+1, n), cartesian_states (T+1, 6), grippers (T,))``
    where row 0 of each state array is the seed.
    """
    n = int(kinematics.n_joints)
    seed = np.asarray(seed_joints, dtype=np.float64).reshape(-1)
    if seed.shape[0] != n:
        raise ValueError(
            f"droid_joint_delta seed must be the {n}-joint configuration, got "
            f"{seed.shape[0]}-D."
        )
    rows = np.asarray(chunk, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != n + 1:
        raise ValueError(
            f"droid_joint_delta action chunk must be (T, {n + 1}) "
            f"[joint_delta({n}), gripper(1)]; got {rows.shape}."
        )

    grippers = rows[:, n].astype(np.float32)
    joint_states = [seed.copy()]
    cur = seed.copy()
    for row in rows:
        cur = cur + row[:n]
        joint_states.append(cur.copy())

    cartesian = []
    for q in joint_states:
        pose = kinematics.fk(q)
        euler = convert_rotation(
            np.asarray(pose.rotation, dtype=np.float32).reshape(1, 3, 3), "matrix", "euler_xyz"
        )[0]
        cartesian.append(np.concatenate([np.asarray(pose.translation, dtype=np.float32), euler]))

    return (
        np.stack(joint_states).astype(np.float32),
        np.stack(cartesian).astype(np.float32),
        grippers,
    )


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


# ---------------------------------------------------------------------------
# Stateful policy-state trackers for the closed-loop simulator.
#
# Cosmos dreams pixels, not proprioception, so the policy's input state is
# tracked across loop turns by integrating the policy's own actions from the
# real seed. A cartesian policy tracks a 6-D end-effector pose; a joint policy
# tracks its joint vector (and forward-kinematics it to the pose Cosmos needs).
# Each turn the tracker reports the proprioception to feed the policy and, given
# the policy's chunk, returns the Cosmos conditioning while advancing its state.
# ---------------------------------------------------------------------------


class StateTracker(ABC):
    """Tracks the policy's proprioceptive state across closed-loop turns."""

    @property
    @abstractmethod
    def gripper(self) -> float:
        """Current gripper value in ``[0, 1]`` (DROID convention, 0=open)."""

    @abstractmethod
    def proprioception(self) -> Proprioception:
        """The proprioception to feed the policy this turn."""

    @abstractmethod
    def to_cosmos(self, chunk: np.ndarray, n_actions: int) -> np.ndarray:
        """Encode the first ``n_actions`` rows as ``(n_actions, 10)`` Cosmos
        conditioning and advance the tracked state to where they led."""

    def gripper_state(self) -> GripperState:
        return GripperState(value=self.gripper)


class CartesianStateTracker(StateTracker):
    """Tracks a 6-D end-effector pose ``[xyz, euler_xyz]``."""

    def __init__(
        self,
        seed_state: np.ndarray,
        seed_gripper: float,
        action_convention: ActionConvention,
        state_convention: str = "ee_pose",
    ):
        if action_convention not in _CARTESIAN_CONVENTIONS:
            raise ValueError(
                f"CartesianStateTracker: action_convention {action_convention!r} is "
                f"not cartesian (expected one of {sorted(_CARTESIAN_CONVENTIONS)})."
            )
        state = np.asarray(seed_state, dtype=np.float32).reshape(-1)
        if state.shape[0] < 6:
            raise ValueError(
                f"CartesianStateTracker: seed_state must be >=6-D [xyz, euler], got {state.shape}."
            )
        self._state = state[:6].copy()
        self._gripper = float(seed_gripper)
        self._convention = action_convention
        self._state_convention = state_convention

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def gripper(self) -> float:
        return self._gripper

    def proprioception(self) -> Proprioception:
        return Proprioception(values=self._state.copy(), convention=self._state_convention)

    def to_cosmos(self, chunk: np.ndarray, n_actions: int) -> np.ndarray:
        states, grippers = integrate_policy_chunk(self._state, chunk[:n_actions], self._convention)
        cosmos = encode_droid_states(states, grippers)
        self._state = states[-1].astype(np.float32)
        self._gripper = float(grippers[-1])
        return cosmos


class JointStateTracker(StateTracker):
    """Tracks a joint configuration and forward-kinematics it for Cosmos.

    The policy reads the joint vector as ``observation/joint_position``; the
    injected ``kinematics`` maps each configuration to the ``panda_link8`` pose
    the DROID encoder conditions on.
    """

    def __init__(self, seed_joints: np.ndarray, seed_gripper: float, kinematics: Kinematics):
        joints = np.asarray(seed_joints, dtype=np.float32).reshape(-1)
        if joints.shape[0] != int(kinematics.n_joints):
            raise ValueError(
                f"JointStateTracker: seed_joints must be the {kinematics.n_joints}-joint "
                f"configuration, got {joints.shape[0]}-D."
            )
        self._joints = joints.copy()
        self._gripper = float(seed_gripper)
        self._kinematics = kinematics

    @property
    def joints(self) -> np.ndarray:
        return self._joints.copy()

    @property
    def gripper(self) -> float:
        return self._gripper

    def proprioception(self) -> Proprioception:
        return Proprioception(values=self._joints.copy(), convention="joint_angles")

    def to_cosmos(self, chunk: np.ndarray, n_actions: int) -> np.ndarray:
        joint_states, cartesian_states, grippers = integrate_joint_chunk(
            self._joints, chunk[:n_actions], self._kinematics
        )
        cosmos = encode_droid_states(cartesian_states, grippers)
        self._joints = joint_states[-1].astype(np.float32)
        self._gripper = float(grippers[-1])
        return cosmos


def make_state_tracker(
    seed_state: np.ndarray,
    seed_gripper: float,
    *,
    action_convention: ActionConvention,
    state_convention: str = "ee_pose",
    kinematics: Optional[Kinematics] = None,
) -> StateTracker:
    """Build the tracker matching ``action_convention``.

    Joint conventions require an injected ``kinematics``; cartesian conventions
    must not receive one. The mismatch is rejected, never silently resolved.
    """
    if action_convention in _JOINT_CONVENTIONS:
        if kinematics is None:
            raise ValueError(
                f"action_convention {action_convention!r} is joint-space and needs a "
                "robot: set cosmos_stress.robot (a preconfigured name) or a custom "
                "urdf/ee_frame/joint_names so forward kinematics can map joints to the "
                "end-effector pose Cosmos conditions on."
            )
        return JointStateTracker(seed_state, seed_gripper, kinematics)
    if kinematics is not None:
        raise ValueError(
            f"action_convention {action_convention!r} is cartesian but a robot/"
            "kinematics was provided; cartesian policies track the end-effector pose "
            "directly and must not declare a robot."
        )
    return CartesianStateTracker(seed_state, seed_gripper, action_convention, state_convention)


__all__ = [
    "ActionConvention",
    "CartesianStateTracker",
    "JointStateTracker",
    "Kinematics",
    "StateTracker",
    "integrate_joint_chunk",
    "integrate_policy_chunk",
    "make_state_tracker",
    "policy_action_source",
    "policy_chunk_to_cosmos",
]
