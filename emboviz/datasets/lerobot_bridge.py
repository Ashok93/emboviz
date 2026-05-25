"""BridgeV2 (IPEC-COMMUNITY/bridge_orig_lerobot) episode source.

One specific configuration of the generic `LeRobotEpisodeSource`.
Bridge's raw state vector is `[x, y, z, roll, pitch, yaw, gripper]` —
6-DOF end-effector pose plus a normalized [0, 1] gripper value. The
`gripper_extractor` here unpacks that into the typed Proprioception +
GripperState that downstream perturbers and diagnostics consume.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.core.profile import (
    ActionSpec,
    CameraSpec,
    GripperSpec,
    RobotProfile,
    StateSpec,
)
from emboviz.datasets.lerobot import LeRobotEpisodeSource


DATASET_REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"


BRIDGE_PROFILE = RobotProfile(
    name="bridge_orig",
    cameras=[CameraSpec(name="primary")],
    state=StateSpec(
        dim=6,
        convention="ee_pose",
        joint_names=["x", "y", "z", "roll", "pitch", "yaw"],
    ),
    gripper=GripperSpec(
        kind="parallel_jaw",
        units="unit",
        range=(0.0, 1.0),
    ),
    action=ActionSpec(
        dim=7,
        dim_names=["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
    ),
)


def _bridge_gripper_extractor(state: np.ndarray) -> tuple[np.ndarray, Optional[float]]:
    """Bridge state layout: [x, y, z, roll, pitch, yaw, gripper]."""
    if state.size < 7:
        return state, None
    return state[:6].copy(), float(state[6])


class BridgeEpisodeSource(LeRobotEpisodeSource):
    """BridgeV2 episode source. Thin instance of the generic LeRobot adapter."""

    def __init__(self):
        super().__init__(
            repo_id=DATASET_REPO,
            profile=BRIDGE_PROFILE,
            image_keys={"primary": "observation.images.image_0"},
            state_key="observation.state",
            action_key="action",
            gripper_extractor=_bridge_gripper_extractor,
            n_episodes=53192,
        )
        self.name = "bridge_v2"
