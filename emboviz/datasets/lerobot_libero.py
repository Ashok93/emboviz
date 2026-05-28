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


# ---------------------------------------------------------------------------
# OFFICIAL openpi training dataset
# ---------------------------------------------------------------------------
# ``physical-intelligence/libero`` is the EXACT HuggingFace dataset openpi
# trains its ``pi0_libero`` checkpoint on. Different schema than the
# community variants above:
#   • images: ``image`` and ``wrist_image`` (no ``observation.`` prefix; CHW
#     tensors are auto-converted to PIL HWC by LeRobotEpisodeSource)
#   • state: ``state`` — 8 dims = [x, y, z, roll, pitch, yaw, gripper_l, gripper_r]
#   • action: ``actions`` (plural) — 7 dims, gripper binarized to {-1, +1}
#     (≡ π0's output convention, so expert_delta gripper |Δ| ≈ 0 when paired
#     with pi0_libero — no convention transform needed)

PI_LIBERO_PROFILE = RobotProfile(
    name="pi_libero",
    cameras=[
        CameraSpec(name="primary"),
        CameraSpec(name="wrist"),
    ],
    state=StateSpec(
        dim=8,
        convention="ee_pose",
        joint_names=["x", "y", "z", "roll", "pitch", "yaw", "gripper_l", "gripper_r"],
    ),
    gripper=GripperSpec(
        kind="parallel_jaw",
        units="m",
        range=(-0.04, 0.04),
    ),
    action=ActionSpec(
        dim=7,
        dim_names=["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
    ),
)


def _pi_libero_gripper_extractor(state: np.ndarray) -> tuple:
    """physical-intelligence/libero state: 8 dims with two symmetric finger positions."""
    if state.size != 8:
        raise ValueError(
            f"physical-intelligence/libero state size {state.size}; expected 8."
        )
    return state.copy(), float(state[6])


class PhysicalIntelligenceLiberoSource(LeRobotEpisodeSource):
    """``physical-intelligence/libero`` — openpi's official pi0_libero training set.

    This is the dataset openpi trains the ``pi0_libero`` checkpoint on.
    Conventions match the model end-to-end — expert_delta on a pi0_libero
    rollout against this dataset is meaningful (gripper sign matches).
    """

    def __init__(self):
        super().__init__(
            repo_id="physical-intelligence/libero",
            profile=PI_LIBERO_PROFILE,
            image_keys={
                "primary": "image",
                "wrist":   "wrist_image",
            },
            state_key="state",
            action_key="actions",   # PLURAL — openpi's naming
            gripper_extractor=_pi_libero_gripper_extractor,
            n_episodes=1693,
        )
        self.name = "pi_libero"
