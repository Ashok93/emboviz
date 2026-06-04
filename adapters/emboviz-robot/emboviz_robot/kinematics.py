"""Forward kinematics for a serial manipulator, backed by Pinocchio.

``RobotKinematics`` loads a URDF, reduces the model to exactly the controlled
joints, and maps a joint configuration to the SE(3) pose of a named
end-effector frame in the robot base frame. It is convention-neutral: it returns
a rotation matrix + translation (:class:`EEPose`); callers pick the orientation
parametrization they need.

Engine: Pinocchio (`pin`), the reference rigid-body-kinematics library
(Carpentier et al., *The Pinocchio C++ library*, SII 2019). API used:

  * ``pin.buildModelFromUrdf(path)`` -> ``pin.Model``  (geometry not needed for FK)
  * ``pin.buildReducedModel(model, joints_to_lock, q_ref)`` -> reduced ``pin.Model``
  * ``pin.forwardKinematics(model, data, q)`` then ``pin.updateFramePlacements``
  * ``model.getFrameId(name)`` / ``data.oMf[frame_id]`` -> ``pin.SE3``
    (``.translation`` (3,), ``.rotation`` (3, 3))

Reduction rationale: the shipped URDF for a robot often includes gripper /
``mimic`` DOFs (Pinocchio does not honor ``<mimic>``, so a Panda URDF parses to
``nq == 9``, not 7). Those joints are locked at the model's neutral
configuration. Locking a joint cannot change the placement of a frame that is
its kinematic ancestor — and the end-effector frame used for conditioning
(e.g. ``panda_link8``) is upstream of the fingers — so the reduction is exact,
not an approximation. The contract this imposes on the caller is explicit:
``joint_names`` must list every joint on the chain from the base to ``ee_frame``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EEPose:
    """An end-effector pose in the robot base frame.

    Convention-neutral: ``rotation`` is a proper 3x3 rotation matrix and
    ``translation`` is in meters. Helpers convert to the orientation
    parametrization a consumer needs.
    """

    translation: np.ndarray  # (3,) meters
    rotation: np.ndarray     # (3, 3) rotation matrix (SO(3))

    def as_xyz_euler(self, seq: str = "xyz") -> tuple[np.ndarray, np.ndarray]:
        """Return ``(xyz (3,), euler (3,))``.

        ``seq`` follows scipy's convention: lowercase (default ``"xyz"``) is
        **extrinsic** rotations about fixed axes — the DROID
        ``cartesian_position`` convention (``droid/misc/transformations.py``
        ``quat_to_euler`` uses scipy ``as_euler("xyz")``). Radians.
        """
        from scipy.spatial.transform import Rotation as R

        euler = R.from_matrix(self.rotation).as_euler(seq, degrees=False)
        return self.translation.astype(np.float64), euler.astype(np.float64)

    def as_matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous transform."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T


class RobotKinematics:
    """Forward kinematics for one manipulator and one end-effector frame.

    Parameters
    ----------
    urdf_path
        Path to the robot's URDF.
    ee_frame
        Name of the frame whose pose ``fk`` returns (e.g. ``"panda_link8"``).
    joint_names
        The controlled joints, in the order ``fk``'s input vector uses. Must be
        every joint on the chain from the base to ``ee_frame``; all other joints
        in the URDF are locked at the neutral configuration.
    """

    def __init__(self, urdf_path: str, ee_frame: str, joint_names: list[str]):
        import pinocchio as pin

        if not joint_names:
            raise ValueError("RobotKinematics: joint_names must be non-empty.")

        full = pin.buildModelFromUrdf(str(urdf_path))

        for name in joint_names:
            if not full.existJointName(name):
                raise ValueError(
                    f"RobotKinematics: joint '{name}' is not in URDF {urdf_path!r}. "
                    f"Available joints: {[n for n in full.names if n != 'universe']}."
                )
        controlled = set(joint_names)
        # Lock every real joint that is not controlled (index 0 is the
        # 'universe' joint; skip it). q_ref is the neutral configuration.
        lock_ids = [
            full.getJointId(name)
            for name in full.names
            if name != "universe" and name not in controlled
        ]
        q_ref = pin.neutral(full)
        model = pin.buildReducedModel(full, lock_ids, q_ref) if lock_ids else full
        data = model.createData()

        if not model.existFrame(ee_frame):
            frame_names = [f.name for f in model.frames]
            raise ValueError(
                f"RobotKinematics: end-effector frame '{ee_frame}' is not in the "
                f"reduced model. Available frames: {frame_names}."
            )

        # Map each controlled joint to its slot in the reduced configuration
        # vector. Order follows joint_names, not the model's tree order, so the
        # input vector's layout is exactly what the caller declared.
        q_index: list[int] = []
        for name in joint_names:
            jid = model.getJointId(name)
            joint = model.joints[jid]
            if joint.nq != 1:
                raise ValueError(
                    f"RobotKinematics: controlled joint '{name}' has nq={joint.nq}; "
                    "only single-DOF joints (revolute/prismatic) are supported."
                )
            q_index.append(int(joint.idx_q))

        if model.nq != len(joint_names):
            raise ValueError(
                f"RobotKinematics: reduced model has nq={model.nq} but "
                f"{len(joint_names)} joints were declared. Every controlled DOF "
                "must be a declared joint and vice versa."
            )

        self._pin = pin
        self._model = model
        self._data = data
        self._ee_frame_id = int(model.getFrameId(ee_frame))
        self._q_index = np.asarray(q_index, dtype=np.int64)
        self.ee_frame = ee_frame
        self.joint_names = list(joint_names)
        self.n_joints = len(joint_names)

    def fk(self, q: np.ndarray) -> EEPose:
        """Pose of ``ee_frame`` for joint configuration ``q`` (``(n_joints,)``)."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.n_joints:
            raise ValueError(
                f"RobotKinematics.fk: expected {self.n_joints} joint values "
                f"({self.joint_names}), got {q.shape[0]}."
            )
        q_model = np.zeros(self._model.nq, dtype=np.float64)
        q_model[self._q_index] = q

        self._pin.forwardKinematics(self._model, self._data, q_model)
        self._pin.updateFramePlacements(self._model, self._data)
        oMf = self._data.oMf[self._ee_frame_id]
        return EEPose(
            translation=np.array(oMf.translation, dtype=np.float64),
            rotation=np.array(oMf.rotation, dtype=np.float64),
        )

    def fk_batch(self, qs: np.ndarray) -> list[EEPose]:
        """Forward kinematics for a stack of configurations ``(T, n_joints)``."""
        qs = np.asarray(qs, dtype=np.float64)
        if qs.ndim != 2 or qs.shape[1] != self.n_joints:
            raise ValueError(
                f"RobotKinematics.fk_batch: expected (T, {self.n_joints}), got {qs.shape}."
            )
        return [self.fk(row) for row in qs]


__all__ = ["EEPose", "RobotKinematics"]
