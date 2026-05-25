"""DROID episode source via the lerobot-format conversion.

DROID (the 76-task Stanford / Berkeley / TRI manipulator dataset) is the
canonical π0 / GR00T benchmark dataset. The lerobot conversion is at
``IPEC-COMMUNITY/droid_100`` (a 100-episode subset) or the full
``IPEC-COMMUNITY/droid`` (76k episodes; multi-TB). Each frame has:

  • two exterior cameras (``exterior_image_1_left`` / ``exterior_image_2_left``)
  • one wrist camera (``wrist_image_left``)
  • 7-DOF joint position (``joint_position``)
  • 1-DOF gripper (``gripper_position``)
  • a language instruction

The wrist-camera and bimanual-exterior layout matches what
``Pi0Adapter(config_name="pi0_fast_droid")`` consumes.
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


DROID_PROFILE = RobotProfile(
    name="droid",
    cameras=[
        CameraSpec(name="primary"),       # exterior_image_1_left
        CameraSpec(name="exterior_2"),    # exterior_image_2_left
        CameraSpec(name="wrist_left"),    # wrist_image_left
    ],
    state=StateSpec(
        dim=7,
        convention="joint_angles",
        joint_names=[f"q{i}" for i in range(7)],
    ),
    gripper=GripperSpec(
        kind="parallel_jaw",
        units="unit",
        range=(0.0, 1.0),
    ),
    action=ActionSpec(dim=7),
)


def _droid_gripper_extractor(state: np.ndarray) -> tuple:
    """DROID state layout: [q0..q6, gripper] when concatenated.

    Some lerobot DROID conversions store the gripper as a separate column;
    in that case the dataset's state vector is already 7-dim and the
    gripper has its own key — handle both.
    """
    if state.size == 7:
        return state.copy(), None   # gripper read from separate key
    if state.size == 8:
        return state[:7].copy(), float(state[7])
    raise ValueError(
        f"DROID state vector has size {state.size}; expected 7 or 8."
    )


class Droid100Source(LeRobotEpisodeSource):
    """100-episode DROID subset — light download (~5 GB) for quick experiments."""

    def __init__(self, repo_id: str = "lerobot/droid_100"):
        super().__init__(
            repo_id=repo_id,
            profile=DROID_PROFILE,
            image_keys={
                "primary":    "observation.images.exterior_image_1_left",
                "exterior_2": "observation.images.exterior_image_2_left",
                "wrist_left": "observation.images.wrist_image_left",
            },
            state_key="observation.state",
            action_key="action",
            gripper_extractor=_droid_gripper_extractor,
            n_episodes=100,
        )
        self.name = "droid_100"


class DroidFullSource(LeRobotEpisodeSource):
    """Full DROID dataset (76k episodes; many TB). Use only if you've
    pre-downloaded with ``lerobot.download_dataset``."""

    def __init__(self, repo_id: str = "lerobot/droid_1.0.1"):
        super().__init__(
            repo_id=repo_id,
            profile=DROID_PROFILE,
            image_keys={
                "primary":    "observation.images.exterior_image_1_left",
                "exterior_2": "observation.images.exterior_image_2_left",
                "wrist_left": "observation.images.wrist_image_left",
            },
            state_key="observation.state",
            action_key="action",
            gripper_extractor=_droid_gripper_extractor,
            n_episodes=76000,
        )
        self.name = "droid_full"
