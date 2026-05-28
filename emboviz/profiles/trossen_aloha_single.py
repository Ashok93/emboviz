"""Trossen ViperX-300 single arm — the cheap 'ALOHA single-arm' research rig."""

from emboviz.core.profile import (
    ActionSpec,
    CameraSpec,
    GripperSpec,
    RobotProfile,
    StateSpec,
)


TROSSEN_ALOHA_SINGLE = RobotProfile(
    name="trossen_aloha_single",
    cameras=[
        CameraSpec(name="primary"),         # workspace / top-down
        CameraSpec(name="wrist"),           # wrist-mounted
    ],
    state=StateSpec(
        dim=6,
        convention="joint_angles",
        joint_names=[
            "waist", "shoulder", "elbow",
            "wrist_angle", "wrist_rotate", "gripper",
        ],
    ),
    gripper=GripperSpec(
        # Trossen uses a 2-finger underactuated gripper driven by one motor;
        # treat as parallel_jaw with normalized units for portability.
        kind="parallel_jaw",
        units="unit",
        range=(0.0, 1.0),
    ),
    action=ActionSpec(
        dim=7,
        dim_names=[
            "waist_dq", "shoulder_dq", "elbow_dq",
            "wrist_angle_dq", "wrist_rotate_dq", "gripper",
            "torque_enable",
        ],
    ),
)
