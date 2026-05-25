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


# BridgeV2 ships four camera streams per frame:
#   image_0  →  the over-the-shoulder exterior camera ("primary")
#   image_1  →  an alternate exterior view (table-side / shoulder-2)
#   image_2  →  a third exterior view (some episodes only)
#   image_3  →  a fourth exterior view (some episodes only)
# We declare ALL of them so diagnostics that iterate every camera see the
# full visual stream. If an episode has only image_0/image_1 populated, the
# loader skips absent keys (rather than fabricating black images).
BRIDGE_PROFILE = RobotProfile(
    name="bridge_orig",
    cameras=[
        CameraSpec(name="primary"),       # image_0
        CameraSpec(name="exterior_2"),    # image_1
        CameraSpec(name="exterior_3"),    # image_2
        CameraSpec(name="exterior_4"),    # image_3
    ],
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
    """Bridge state layout: [x, y, z, roll, pitch, yaw, gripper, <unused>].

    The dataset's raw state is 8-dim; index 6 is the normalised gripper
    value, index 7 is an unused trailing slot we discard.
    """
    if state.size < 7:
        raise ValueError(
            f"Bridge state vector has only {state.size} dims; expected ≥7 "
            "(layout: [x, y, z, roll, pitch, yaw, gripper, ...])."
        )
    return state[:6].copy(), float(state[6])


class BridgeEpisodeSource(LeRobotEpisodeSource):
    """BridgeV2 episode source. Thin instance of the generic LeRobot adapter.

    All four exterior camera streams (image_0..image_3) are declared. Any
    diagnostic that iterates ``scene.observations.images`` will see every
    camera that the dataset populates for the episode.
    """

    def __init__(self):
        super().__init__(
            repo_id=DATASET_REPO,
            profile=BRIDGE_PROFILE,
            image_keys={
                "primary":    "observation.images.image_0",
                "exterior_2": "observation.images.image_1",
                "exterior_3": "observation.images.image_2",
                "exterior_4": "observation.images.image_3",
            },
            state_key="observation.state",
            action_key="action",
            gripper_extractor=_bridge_gripper_extractor,
            n_episodes=53192,
        )
        self.name = "bridge_v2"
