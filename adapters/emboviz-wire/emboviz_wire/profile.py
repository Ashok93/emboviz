"""RobotProfile — declarative per-robot metadata.

A profile is optional. When present, it provides perturbers and
diagnostics with the team-specific information they need to operate
intelligently on the specific hardware setup: joint names for
segmenting state, action-dim labels for meaningful divergence
metrics, gripper kind specifics for the right perturbation, etc.

Profiles are pure data — no inference, no I/O. They're typically
shipped as small Python files under `emboviz/profiles/` (preset
configs like franka_robotiq, ur5_robotiq, trossen) or constructed
ad-hoc by a team's episode source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from emboviz_wire.observations.gripper import GripperKind, GripperUnits
from emboviz_wire.observations.state import StateConvention


@dataclass(frozen=True)
class CameraSpec:
    """One camera in the robot's perception setup."""

    name: str
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass(frozen=True)
class StateSpec:
    """Declarative info about the robot's proprioceptive state vector."""

    dim: int
    convention: StateConvention
    joint_names: Optional[list[str]] = None
    # Named segments mapping into the state vector — e.g.,
    # {"left_arm": slice(0, 7), "right_arm": slice(7, 14)} for bimanual.
    # Perturbers use this to isolate per-segment perturbation.
    segment_layout: Optional[dict[str, slice]] = None


@dataclass(frozen=True)
class GripperSpec:
    """Declarative info about the gripper / end-effector."""

    kind: GripperKind
    units: GripperUnits
    # Open/closed range in `units`. For binary, conventionally (0.0, 1.0).
    range: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class ActionSpec:
    """Declarative info about the policy's action space."""

    dim: int
    # Optional human-readable name per action dim — e.g.,
    # ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"] for OpenVLA.
    # Reports show dim names instead of bare indices.
    dim_names: Optional[list[str]] = None
    # Per-dim normalization scale (e.g., Bridge q99 - q01). Used by
    # `normalized_l2` so action distances are comparable across dims.
    dim_scale: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RobotProfile:
    """The full per-robot configuration.

    All fields except `name` are optional — generic defaults handle the
    common single-arm tabletop case without any profile at all. Profiles
    earn their keep when something non-generic is true: multi-cam,
    bimanual, non-standard gripper, exotic state convention.
    """

    name: str
    cameras: list[CameraSpec] = field(default_factory=list)
    state: Optional[StateSpec] = None
    gripper: Optional[GripperSpec] = None
    action: Optional[ActionSpec] = None
    # Free-form per-robot extras (e.g., tactile sensor layouts, custom
    # joint torque limits) that perturbers/diagnostics in the user's
    # custom code may reference.
    extras: dict[str, Any] = field(default_factory=dict)
