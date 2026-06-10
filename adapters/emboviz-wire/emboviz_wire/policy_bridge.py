"""Policy-to-world-model action bridge — shared state integration.

A world-model stress test runs the *user's* policy and renders the consequence.
World models condition on end-effector *state* sequences (Cosmos ``droid_lerobot``
encodes normalized pose deltas; Ctrl-World conditions on absolute poses), but a
policy emits an *action* chunk. This module integrates a policy's predicted chunk
into a Cartesian state sequence under an **explicitly declared** action
convention (never inferred — a guessed convention silently corrupts every
rendered frame). The per-model action *encoding* stays in each world-model
adapter; only the model-agnostic integration lives here.

Supported conventions (the policy's action-chunk row layout):

Cartesian (7-D rows ``[..., gripper]``, gripper in ``[0, 1]``):

  * ``"absolute_xyz_euler"`` — ``[x, y, z, roll, pitch, yaw, gripper]``: each row
    is the absolute next end-effector pose in the DROID cartesian convention.
  * ``"delta_xyz_euler_base"`` — ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``:
    each row is a base-frame pose delta; position adds, rotation pre-multiplies
    (``R_next = dR @ R_cur``).

Joint (``[joint_velocity(n), gripper(1)]`` rows; gripper absolute in ``[0, 1]``):

  * ``"droid_joint_velocity"`` — the π0-DROID action space: 7 joint *velocities*
    (rad/s) integrated at the control rate (``dt = 1/control_hz``, 15 Hz for
    DROID) plus an absolute gripper. π0-DROID is trained and deployed on joint
    velocities, NOT position deltas (openpi: ``RobotEnv(action_space=
    "joint_velocity")``, ``DROID_CONTROL_FREQUENCY = 15``); integrating them
    without the ``dt`` factor inflates every motion ~15x. The tracked state is
    the joint vector itself (what the policy reads as
    ``observation/joint_position``); forward kinematics maps each joint
    configuration to the end-effector pose (``panda_link8`` for DROID). Requires
    an injected ``kinematics`` object (a :class:`emboviz_robot.RobotKinematics`),
    so this module never imports the kinematics engine — the driver wires it.

Other conventions raise — add them explicitly rather than approximating.

Euler convention: extrinsic XYZ (rotations about the fixed base axes, applied
x-then-y-then-z), the DROID ``cartesian_position`` convention (``droid/misc/
transformations.py`` uses scipy ``as_euler("xyz")``). The pure-numpy helpers
below reproduce scipy's ``from_euler("xyz")`` / ``as_euler("xyz")`` so this
module adds no scipy dependency to the wire package; equivalence is covered by
the adapter test suites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional, Protocol

import numpy as np

from emboviz_wire.observations.gripper import GripperState
from emboviz_wire.observations.state import Proprioception

ActionConvention = Literal[
    "absolute_xyz_euler", "delta_xyz_euler_base", "droid_joint_velocity"
]
CARTESIAN_ACTION_CONVENTIONS = frozenset({"absolute_xyz_euler", "delta_xyz_euler_base"})
JOINT_ACTION_CONVENTIONS = frozenset({"droid_joint_velocity"})
ACTION_CONVENTIONS = CARTESIAN_ACTION_CONVENTIONS | JOINT_ACTION_CONVENTIONS

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


# ---------------------------------------------------------------------------
# Extrinsic-XYZ euler <-> rotation matrix, pure numpy.
#
# R = Rz(yaw) @ Ry(pitch) @ Rx(roll) — extrinsic rotations about the fixed
# x, y, z axes in that order, matching scipy's lowercase "xyz" sequence.
# ---------------------------------------------------------------------------


def euler_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    """``[roll, pitch, yaw]`` (radians, extrinsic XYZ) -> (3, 3) rotation matrix."""
    e = np.asarray(euler, dtype=np.float64).reshape(-1)
    if e.shape[0] != 3:
        raise ValueError(f"euler_xyz_to_matrix expects 3 angles, got shape {e.shape}.")
    cx, cy, cz = np.cos(e)
    sx, sy, sz = np.sin(e)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


def matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    """(3, 3) rotation matrix -> ``[roll, pitch, yaw]`` (radians, extrinsic XYZ).

    At the gimbal singularity (``|pitch| = π/2``) roll and yaw are not
    independent; roll is set to 0 and the combined angle is reported as yaw —
    the same resolution scipy applies.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"matrix_to_euler_xyz expects a (3, 3) matrix, got {m.shape}.")
    sy = -m[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = float(np.arcsin(sy))
    if abs(sy) < 1.0 - 1e-9:
        roll = float(np.arctan2(m[2, 1], m[2, 2]))
        yaw = float(np.arctan2(m[1, 0], m[0, 0]))
    else:
        # cos(pitch) == 0: only roll±yaw is observable. Convention: roll = 0.
        roll = 0.0
        yaw = float(np.arctan2(-m[0, 1], m[1, 1]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


# ---------------------------------------------------------------------------
# Chunk integration: action rows -> state sequences.
# ---------------------------------------------------------------------------


def integrate_policy_chunk(
    seed_state_xyz_euler: np.ndarray,
    chunk: np.ndarray,
    convention: ActionConvention,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a cartesian policy action chunk into a state sequence.

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
        cur_xyz = seed[:3].astype(np.float64)
        cur_R = euler_xyz_to_matrix(seed[3:6])
        for row in rows:
            cur_xyz = cur_xyz + row[:3]
            cur_R = euler_xyz_to_matrix(row[3:6]) @ cur_R
            states.append(
                np.concatenate([cur_xyz, matrix_to_euler_xyz(cur_R)]).astype(np.float32)
            )
    else:
        raise ValueError(
            f"unsupported action convention {convention!r}; supported: "
            "'absolute_xyz_euler', 'delta_xyz_euler_base'. Declare the policy's "
            "action convention explicitly — it is never inferred."
        )

    return np.stack(states).astype(np.float32), grippers


def integrate_joint_chunk(
    seed_joints: np.ndarray,
    chunk: np.ndarray,
    kinematics: Kinematics,
    *,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a joint-velocity action chunk into joint and Cartesian sequences.

    ``chunk`` is ``(T, n_joints + 1)`` rows ``[joint_velocity(n), gripper(1)]`` —
    the ``droid_joint_velocity`` convention. Each joint configuration advances by
    ``velocity * dt`` (zero-order hold over one control step), with
    ``dt = 1 / control_hz``. The gripper column is absolute position (unscaled).
    Forward kinematics maps each configuration to the end-effector pose, returned
    as ``[xyz, euler_xyz]`` in the DROID ``cartesian_position`` convention
    (extrinsic-XYZ euler).

    ``dt = 1`` integrates raw position deltas instead — used by recorded-state
    validation paths, which feed logged joint-position differences (already
    per-step).

    Returns ``(joint_states (T+1, n), cartesian_states (T+1, 6), grippers (T,))``
    where row 0 of each state array is the seed.
    """
    if dt <= 0:
        raise ValueError(f"integrate_joint_chunk: dt must be > 0, got {dt}.")
    n = int(kinematics.n_joints)
    seed = np.asarray(seed_joints, dtype=np.float64).reshape(-1)
    if seed.shape[0] != n:
        raise ValueError(
            f"droid_joint_velocity seed must be the {n}-joint configuration, got "
            f"{seed.shape[0]}-D."
        )
    rows = np.asarray(chunk, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != n + 1:
        raise ValueError(
            f"droid_joint_velocity action chunk must be (T, {n + 1}) "
            f"[joint_velocity({n}), gripper(1)]; got {rows.shape}."
        )

    grippers = rows[:, n].astype(np.float32)
    joint_states = [seed.copy()]
    cur = seed.copy()
    for row in rows:
        cur = cur + dt * row[:n]
        joint_states.append(cur.copy())

    cartesian = []
    for q in joint_states:
        pose = kinematics.fk(q)
        euler = matrix_to_euler_xyz(np.asarray(pose.rotation, dtype=np.float64))
        cartesian.append(
            np.concatenate([np.asarray(pose.translation, dtype=np.float64), euler])
        )

    return (
        np.stack(joint_states).astype(np.float32),
        np.stack(cartesian).astype(np.float32),
        grippers,
    )


# ---------------------------------------------------------------------------
# Stateful policy-state trackers for the closed-loop simulator.
#
# A world model dreams pixels, not proprioception, so the policy's input state
# is tracked across loop turns by integrating the policy's own actions from the
# real seed. A cartesian policy tracks a 6-D end-effector pose; a joint policy
# tracks its joint vector (and forward-kinematics it to the pose the world
# model conditions on). Each turn the tracker reports the proprioception to
# feed the policy, integrates the policy's chunk into the Cartesian state
# sequence the adapter encodes, and advances by the committed steps.
# ---------------------------------------------------------------------------


class StateTracker(ABC):
    """Tracks the policy's proprioceptive state across closed-loop turns.

    Split API — integration is pure, advancing is explicit — so the prediction
    horizon (frames dreamed this turn) and the execution horizon (frames
    committed before the policy re-plans, receding horizon) stay independent:

    * :meth:`integrate` maps the chunk's first ``n_steps`` rows to the Cartesian
      state sequence the world-model adapter encodes; the tracked state does
      not move.
    * :meth:`advance` commits the chunk's first ``n_steps`` rows into the
      tracked state, so the proprioception the policy reads next turn matches
      the dreamed frame the loop commits to. Advancing past the committed
      frame would desync proprioception from pixels.
    """

    @property
    @abstractmethod
    def gripper(self) -> float:
        """Current gripper value in ``[0, 1]`` (DROID convention, 0=open)."""

    @abstractmethod
    def proprioception(self) -> Proprioception:
        """The proprioception to feed the policy this turn."""

    @abstractmethod
    def integrate(self, chunk: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Integrate the first ``n_steps`` chunk rows from the current state.

        Returns ``(cartesian_states (n_steps+1, 6), grippers (n_steps,))`` with
        row 0 the current state. Pure — the tracked state does not change."""

    @abstractmethod
    def advance(self, chunk: np.ndarray, n_steps: int) -> None:
        """Advance the tracked state by the first ``n_steps`` chunk rows."""

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
        if action_convention not in CARTESIAN_ACTION_CONVENTIONS:
            raise ValueError(
                f"CartesianStateTracker: action_convention {action_convention!r} is "
                f"not cartesian (expected one of {sorted(CARTESIAN_ACTION_CONVENTIONS)})."
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

    def integrate(self, chunk: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        _check_n_steps(chunk, n_steps)
        return integrate_policy_chunk(self._state, chunk[:n_steps], self._convention)

    def advance(self, chunk: np.ndarray, n_steps: int) -> None:
        states, grippers = self.integrate(chunk, n_steps)
        self._state = states[n_steps].astype(np.float32)
        self._gripper = float(grippers[n_steps - 1])


class JointStateTracker(StateTracker):
    """Tracks a joint configuration and forward-kinematics it to a pose.

    The policy reads the joint vector as ``observation/joint_position``; the
    injected ``kinematics`` maps each configuration to the end-effector pose
    world models condition on (``panda_link8`` for DROID). The policy emits
    joint *velocities* (rad/s), integrated at the control timestep
    ``dt = 1 / control_hz`` — π0-DROID runs at ``control_hz = 15``.
    """

    def __init__(
        self, seed_joints: np.ndarray, seed_gripper: float, kinematics: Kinematics,
        *, control_hz: float = 15.0,
    ):
        joints = np.asarray(seed_joints, dtype=np.float32).reshape(-1)
        if joints.shape[0] != int(kinematics.n_joints):
            raise ValueError(
                f"JointStateTracker: seed_joints must be the {kinematics.n_joints}-joint "
                f"configuration, got {joints.shape[0]}-D."
            )
        if control_hz <= 0:
            raise ValueError(f"JointStateTracker: control_hz must be > 0, got {control_hz}.")
        self._joints = joints.copy()
        self._gripper = float(seed_gripper)
        self._kinematics = kinematics
        self._dt = 1.0 / float(control_hz)

    @property
    def joints(self) -> np.ndarray:
        return self._joints.copy()

    @property
    def gripper(self) -> float:
        return self._gripper

    def proprioception(self) -> Proprioception:
        return Proprioception(values=self._joints.copy(), convention="joint_angles")

    def cartesian_state(self) -> np.ndarray:
        """The current end-effector pose ``[xyz, euler_xyz]`` via forward kinematics."""
        pose = self._kinematics.fk(self._joints.astype(np.float64))
        euler = matrix_to_euler_xyz(np.asarray(pose.rotation, dtype=np.float64))
        return np.concatenate(
            [np.asarray(pose.translation, dtype=np.float64), euler]
        ).astype(np.float32)

    def integrate(self, chunk: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        _check_n_steps(chunk, n_steps)
        _, cartesian, grippers = integrate_joint_chunk(
            self._joints, chunk[:n_steps], self._kinematics, dt=self._dt
        )
        return cartesian, grippers

    def advance(self, chunk: np.ndarray, n_steps: int) -> None:
        _check_n_steps(chunk, n_steps)
        rows = np.asarray(chunk, dtype=np.float64)[:n_steps]
        n = int(self._kinematics.n_joints)
        if rows.shape[1] != n + 1:
            raise ValueError(
                f"JointStateTracker.advance: chunk rows must be {n + 1}-D "
                f"[joint_velocity({n}), gripper(1)]; got {rows.shape}."
            )
        self._joints = (
            self._joints.astype(np.float64) + self._dt * rows[:, :n].sum(axis=0)
        ).astype(np.float32)
        self._gripper = float(rows[n_steps - 1, n])


def _check_n_steps(chunk: np.ndarray, n_steps: int) -> None:
    rows = np.asarray(chunk)
    if rows.ndim != 2:
        raise ValueError(f"action chunk must be 2-D (T, row_dim); got shape {rows.shape}.")
    if not 1 <= int(n_steps) <= rows.shape[0]:
        raise ValueError(
            f"n_steps must satisfy 1 <= n_steps <= chunk length ({rows.shape[0]}); "
            f"got {n_steps}."
        )


def make_state_tracker(
    seed_state: np.ndarray,
    seed_gripper: float,
    *,
    action_convention: ActionConvention,
    state_convention: str = "ee_pose",
    kinematics: Optional[Kinematics] = None,
    control_hz: float = 15.0,
) -> StateTracker:
    """Build the tracker matching ``action_convention``.

    Joint conventions require an injected ``kinematics`` and integrate joint
    velocities at ``control_hz`` (the policy's control rate, 15 Hz for π0-DROID);
    cartesian conventions must not receive a robot. The mismatch is rejected,
    never silently resolved.
    """
    if action_convention in JOINT_ACTION_CONVENTIONS:
        if kinematics is None:
            raise ValueError(
                f"action_convention {action_convention!r} is joint-space and needs a "
                "robot: set stress.robot (a preconfigured name) or a custom "
                "urdf/ee_frame/joint_names so forward kinematics can map joints to the "
                "end-effector pose the world model conditions on."
            )
        return JointStateTracker(seed_state, seed_gripper, kinematics, control_hz=control_hz)
    if kinematics is not None:
        raise ValueError(
            f"action_convention {action_convention!r} is cartesian but a robot/"
            "kinematics was provided; cartesian policies track the end-effector pose "
            "directly and must not declare a robot."
        )
    return CartesianStateTracker(seed_state, seed_gripper, action_convention, state_convention)


__all__ = [
    "ACTION_CONVENTIONS",
    "ActionConvention",
    "CARTESIAN_ACTION_CONVENTIONS",
    "CartesianStateTracker",
    "JOINT_ACTION_CONVENTIONS",
    "JointStateTracker",
    "Kinematics",
    "StateTracker",
    "euler_xyz_to_matrix",
    "integrate_joint_chunk",
    "integrate_policy_chunk",
    "make_state_tracker",
    "matrix_to_euler_xyz",
]
