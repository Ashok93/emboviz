"""LIBERO episode sources via the lerobot-format conversions.

The community-maintained LIBERO splits live under HuggingFace as
``aopolin-lv/libero_x_no_noops_lerobot_v21{spatial,object,goal,10}_no_noops``. Each split
has two cameras (``image`` exterior + ``wrist_image``) and an 8-dim state
vector (3-DOF position + 4-DOF orientation + gripper).

These match what ``Pi0Adapter(config_name="pi0_libero")`` and
``OpenVLAOFTAdapter(unnorm_key="libero_*")`` consume.
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


LIBERO_PROFILE = RobotProfile(
    name="libero",
    cameras=[
        CameraSpec(name="primary"),    # exterior agent-view
        CameraSpec(name="wrist"),      # in-hand wrist cam
    ],
    state=StateSpec(
        dim=8,
        convention="ee_pose",
        joint_names=["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
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


def _libero_gripper_extractor(state: np.ndarray) -> tuple:
    """LIBERO state layout: [x, y, z, qx, qy, qz, qw, gripper]."""
    if state.size != 8:
        raise ValueError(
            f"LIBERO state vector has size {state.size}; expected 8."
        )
    # Strict: emit the full 8-dim state (model expects 8) plus the gripper scalar.
    # We return the full state as 'proprio' so the LeRobotEpisodeSource doesn't
    # truncate; the adapter splits it back out for π0/OFT.
    return state.copy(), float(state[7])


class LiberoSpatialSource(LeRobotEpisodeSource):
    """LIBERO-spatial split. 432 episodes, single-arm 7-DOF."""
    def __init__(self):
        super().__init__(
            repo_id="aopolin-lv/libero_spatial_no_noops_lerobot_v21",
            profile=LIBERO_PROFILE,
            image_keys={
                "primary": "observation.images.image",
                "wrist":   "observation.images.wrist_image",
            },
            state_key="observation.state",
            action_key="action",
            gripper_extractor=_libero_gripper_extractor,
            n_episodes=432,
        )
        self.name = "libero_spatial"


class LiberoObjectSource(LeRobotEpisodeSource):
    def __init__(self):
        super().__init__(
            repo_id="aopolin-lv/libero_object_no_noops_lerobot_v21",
            profile=LIBERO_PROFILE,
            image_keys={
                "primary": "observation.images.image",
                "wrist":   "observation.images.wrist_image",
            },
            state_key="observation.state",
            action_key="action",
            gripper_extractor=_libero_gripper_extractor,
            n_episodes=480,
        )
        self.name = "libero_object"


class LiberoGoalSource(LeRobotEpisodeSource):
    def __init__(self):
        super().__init__(
            repo_id="aopolin-lv/libero_goal_no_noops_lerobot_v21",
            profile=LIBERO_PROFILE,
            image_keys={
                "primary": "observation.images.image",
                "wrist":   "observation.images.wrist_image",
            },
            state_key="observation.state",
            action_key="action",
            gripper_extractor=_libero_gripper_extractor,
            n_episodes=432,
        )
        self.name = "libero_goal"


class Libero10Source(LeRobotEpisodeSource):
    def __init__(self):
        super().__init__(
            repo_id="aopolin-lv/libero_10_no_noops_lerobot_v21",
            profile=LIBERO_PROFILE,
            image_keys={
                "primary": "observation.images.image",
                "wrist":   "observation.images.wrist_image",
            },
            state_key="observation.state",
            action_key="action",
            gripper_extractor=_libero_gripper_extractor,
            n_episodes=379,
        )
        self.name = "libero_10"
