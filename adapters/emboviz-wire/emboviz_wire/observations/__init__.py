"""Typed observation classes — one per sensor modality.

Each modality is its own frozen dataclass carrying its raw data plus the
metadata that makes the data unambiguous (units, conventions, frame).
This kills a category of silent-wrong-answer bugs where a joint-angle
vector ends up fed to a model expecting end-effector pose, or a gripper
value in mm ends up interpreted as unit-normalized.

The runtime shape of each modality is intentionally small; richer
per-team metadata (joint names, gripper kind specifics, action-dim
layout) lives in `emboviz_wire.profile.RobotProfile`.
"""

from emboviz_wire.observations.action_history import ActionHistory
from emboviz_wire.observations.depth import DepthMap
from emboviz_wire.observations.force import ForceTorque
from emboviz_wire.observations.gripper import GripperState
from emboviz_wire.observations.image import RGBImage
from emboviz_wire.observations.state import Proprioception
from emboviz_wire.observations.tactile import TactileReading

__all__ = [
    "ActionHistory",
    "DepthMap",
    "ForceTorque",
    "GripperState",
    "RGBImage",
    "Proprioception",
    "TactileReading",
]
