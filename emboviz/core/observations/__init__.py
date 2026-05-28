"""Typed observation classes — one per sensor modality.

Each modality is its own frozen dataclass carrying its raw data plus the
metadata that makes the data unambiguous (units, conventions, frame).
This kills a category of silent-wrong-answer bugs where a joint-angle
vector ends up fed to a model expecting end-effector pose, or a gripper
value in mm ends up interpreted as unit-normalized.

The runtime shape of each modality is intentionally small; richer
per-team metadata (joint names, gripper kind specifics, action-dim
layout) lives in `emboviz.core.profile.RobotProfile`.
"""

from emboviz.core.observations.action_history import ActionHistory
from emboviz.core.observations.depth import DepthMap
from emboviz.core.observations.force import ForceTorque
from emboviz.core.observations.gripper import GripperState
from emboviz.core.observations.image import RGBImage
from emboviz.core.observations.state import Proprioception
from emboviz.core.observations.tactile import TactileReading

__all__ = [
    "ActionHistory",
    "DepthMap",
    "ForceTorque",
    "GripperState",
    "RGBImage",
    "Proprioception",
    "TactileReading",
]
