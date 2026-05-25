"""ALOHA (bimanual Trossen Vipers) episode sources.

The canonical lerobot ALOHA datasets ship in two shapes:

  • Single-cam, demonstration-only:
      ``lerobot/aloha_sim_transfer_cube_human``
      ``lerobot/aloha_sim_insertion_human``
    State is (14,) — 7 joints per arm. Only one camera (`top`) is present.

  • Multi-cam, full hardware capture:
      ``lerobot/aloha_static_*``  (4 cameras: cam_high / cam_low /
      cam_left_wrist / cam_right_wrist), state (14,).

We declare distinct sources for each shape; do not silently downgrade a
multi-cam adapter to a single-cam dataset.
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


ALOHA_BIMANUAL_PROFILE_4CAM = RobotProfile(
    name="aloha_bimanual_4cam",
    cameras=[
        CameraSpec(name="cam_high"),
        CameraSpec(name="cam_low"),
        CameraSpec(name="cam_left_wrist"),
        CameraSpec(name="cam_right_wrist"),
    ],
    state=StateSpec(
        dim=14,
        convention="joint_angles",
        joint_names=[f"left_{i}" for i in range(7)] + [f"right_{i}" for i in range(7)],
        segment_layout={"left_arm": slice(0, 7), "right_arm": slice(7, 14)},
    ),
    # ALOHA grippers are commanded jointly with the joint vector — no
    # separate scalar gripper to break out.
    action=ActionSpec(dim=14),
)

ALOHA_BIMANUAL_PROFILE_1CAM = RobotProfile(
    name="aloha_bimanual_top_only",
    cameras=[CameraSpec(name="top")],
    state=StateSpec(
        dim=14,
        convention="joint_angles",
        joint_names=[f"left_{i}" for i in range(7)] + [f"right_{i}" for i in range(7)],
        segment_layout={"left_arm": slice(0, 7), "right_arm": slice(7, 14)},
    ),
    action=ActionSpec(dim=14),
)


class AlohaSimTransferCubeSource(LeRobotEpisodeSource):
    """``lerobot/aloha_sim_transfer_cube_human`` — single top camera, 400 frames/ep."""

    def __init__(self):
        super().__init__(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            profile=ALOHA_BIMANUAL_PROFILE_1CAM,
            image_keys={"top": "observation.images.top"},
            state_key="observation.state",
            action_key="action",
            n_episodes=50,
        )
        self.name = "aloha_sim_transfer_cube_human"


class AlohaSimInsertionSource(LeRobotEpisodeSource):
    """``lerobot/aloha_sim_insertion_human`` — single top camera, insertion task."""

    def __init__(self):
        super().__init__(
            repo_id="lerobot/aloha_sim_insertion_human",
            profile=ALOHA_BIMANUAL_PROFILE_1CAM,
            image_keys={"top": "observation.images.top"},
            state_key="observation.state",
            action_key="action",
            n_episodes=50,
        )
        self.name = "aloha_sim_insertion_human"


class AlohaStatic4CamSource(LeRobotEpisodeSource):
    """Full 4-camera ALOHA capture (e.g. ``lerobot/aloha_static_coffee``)."""

    def __init__(self, repo_id: str = "lerobot/aloha_static_coffee", n_episodes: int = 50):
        super().__init__(
            repo_id=repo_id,
            profile=ALOHA_BIMANUAL_PROFILE_4CAM,
            image_keys={
                "cam_high":        "observation.images.cam_high",
                "cam_low":         "observation.images.cam_low",
                "cam_left_wrist":  "observation.images.cam_left_wrist",
                "cam_right_wrist": "observation.images.cam_right_wrist",
            },
            state_key="observation.state",
            action_key="action",
            n_episodes=n_episodes,
        )
        self.name = repo_id.rsplit("/", 1)[-1]
