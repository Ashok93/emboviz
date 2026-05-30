"""End-effector / gripper state, unified across kinds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


GripperKind = Literal[
    "parallel_jaw",     # 2-finger parallel-jaw (Robotiq 2F-85, Panda Hand, etc.)
    "suction",          # vacuum cup (on/off or pressure)
    "binary",           # any open/closed end-effector
    "magnetic",         # electromagnet (engagement strength)
    "multi_finger",     # anthropomorphic / underactuated multi-finger hands
]

GripperUnits = Literal[
    "unit",             # normalized [0, 1] (the default for OpenVLA-style)
    "m",                # meters (parallel-jaw finger width, SI — e.g. Panda ±0.04 m)
    "mm",               # millimeters (parallel-jaw width)
    "rad",              # radians (rotary grippers, finger joints)
    "binary",           # 0.0 or 1.0
]


@dataclass(frozen=True)
class GripperState:
    """The gripper's state at one timestep.

    Most parallel-jaw and suction grippers fit cleanly with `value` +
    `units`. Multi-finger hands populate `joint_angles` as well, and
    `value` is the aggregate "openness" estimate.

    The gripper *kind* (mechanical category) is stable per robot and
    duplicated here for convenience; the canonical source is
    `RobotProfile.gripper`.
    """

    value: float
    kind: GripperKind = "parallel_jaw"
    units: GripperUnits = "unit"
    joint_angles: Optional[np.ndarray] = None
