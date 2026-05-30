"""Proprioceptive state — joint angles, EE pose, etc."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


StateConvention = Literal[
    "joint_angles",
    "joint_velocities",
    "joint_torques",
    "ee_pose",          # (x, y, z, qx, qy, qz, qw) or (x, y, z, r, p, y)
    "ee_delta",         # relative pose delta from previous frame
    "ee_velocity",      # 6D twist: linear + angular
]


@dataclass(frozen=True)
class Proprioception:
    """The robot's proprioceptive state at one timestep.

    `convention` is mandatory because a joint-angle vector is NOT
    interchangeable with an EE-pose vector. Perturbers and diagnostics
    read the convention to operate correctly; downstream code that
    forgets to check is fail-loud rather than fail-silent.

    Per-joint names, segment layouts, and dim semantics live on the
    optional `emboviz_wire.profile.RobotProfile.state` — that's where
    richer per-team metadata belongs.
    """

    values: np.ndarray
    convention: StateConvention
