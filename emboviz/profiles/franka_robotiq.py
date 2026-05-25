"""Franka Panda + Robotiq 2F-85 — the canonical single-arm tabletop setup."""

from emboviz.core.profile import (
    ActionSpec,
    CameraSpec,
    GripperSpec,
    RobotProfile,
    StateSpec,
)


FRANKA_ROBOTIQ = RobotProfile(
    name="franka_robotiq",
    cameras=[
        CameraSpec(name="primary"),       # typically a top-down or side workspace cam
        CameraSpec(name="wrist"),         # optional wrist-mounted camera
    ],
    state=StateSpec(
        dim=7,
        convention="joint_angles",
        joint_names=[f"joint{i+1}" for i in range(7)],  # Panda has 7 DOF
    ),
    gripper=GripperSpec(
        kind="parallel_jaw",
        units="mm",
        range=(0.0, 85.0),                # Robotiq 2F-85 max opening
    ),
    action=ActionSpec(
        dim=8,
        dim_names=[
            "joint1_dq", "joint2_dq", "joint3_dq",
            "joint4_dq", "joint5_dq", "joint6_dq", "joint7_dq",
            "gripper",
        ],
    ),
)
