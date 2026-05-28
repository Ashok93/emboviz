"""UR5 / UR10 + Robotiq 2F-85 — industrial-arm setup popular in warehouse research."""

from emboviz.core.profile import (
    ActionSpec,
    CameraSpec,
    GripperSpec,
    RobotProfile,
    StateSpec,
)


UR5_ROBOTIQ = RobotProfile(
    name="ur5_robotiq",
    cameras=[
        CameraSpec(name="primary"),
        CameraSpec(name="wrist"),
    ],
    state=StateSpec(
        dim=6,
        convention="joint_angles",
        joint_names=[
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ],
    ),
    gripper=GripperSpec(
        kind="parallel_jaw",
        units="mm",
        range=(0.0, 85.0),
    ),
    action=ActionSpec(
        dim=7,
        dim_names=[
            "shoulder_pan_dq",
            "shoulder_lift_dq",
            "elbow_dq",
            "wrist_1_dq",
            "wrist_2_dq",
            "wrist_3_dq",
            "gripper",
        ],
    ),
)
