# emboviz-robot

Robot-agnostic **forward kinematics** for emboviz: turn a manipulator's joint
configuration into its end-effector pose. This is the bridge that lets a
joint-space policy (e.g. π0-DROID, which reads `joint_position` and emits joint
deltas) drive a world model whose action conditioning is expressed in
**Cartesian** end-effector space (e.g. NVIDIA Cosmos `droid_lerobot`).

It is deliberately independent of any world model or policy: forward kinematics
is a property of the *robot*, so it lives in its own package. The world-model
adapter depends on this; this depends on neither.

## What it does

```python
from emboviz_robot import load_kinematics

# Preconfigured robot — the user names it, nothing else.
kin = load_kinematics("franka_panda")
pose = kin.fk(joint_position)          # (7,) radians -> EEPose
xyz, euler = pose.as_xyz_euler()       # extrinsic-XYZ euler, the DROID convention

# A robot not in the catalog — supply its URDF, same call shape, same flow.
kin = load_kinematics(urdf="/path/to/arm.urdf",
                      ee_frame="tool0",
                      joint_names=["j1", "j2", "j3", "j4", "j5", "j6"])
pose = kin.fk(q)
```

Common robots are **preconfigured** (the user passes only a name; the official
URDF is resolved through [`robot_descriptions`](https://github.com/robot-descriptions/robot_descriptions.py)
and cached). Uncommon robots take the same path via an explicit URDF. Both
produce the same `RobotKinematics`, so downstream code is identical.

## Engine

Forward kinematics is computed with [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
(`pin`), the standard rigid-body-kinematics library. The robot model is reduced
to exactly its controlled joints (gripper / `mimic` DOFs are locked at their
neutral configuration, which cannot affect a frame that is kinematically
upstream of them), so `fk` takes a joint vector whose length equals the number
of controlled joints, in the declared order.

See `LITERATURE.md` (repo root) for the kinematics and frame-convention
citations.
