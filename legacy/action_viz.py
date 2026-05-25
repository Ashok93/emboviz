"""Project 7-DOF Bridge actions to 2D screen-space arrows for visualization.

We don't know the camera intrinsics for arbitrary Bridge scenes, so this is
deliberately approximate — the goal is *comparing* arrows across instruction
variants on the same image, not absolute calibration. The viewer can see
"these two arrows are nearly identical / wildly different."

Bridge action layout (per `bridge_orig` unnorm stats):
    [0] dx  end-effector translation X (m)   — robot forward/back
    [1] dy  end-effector translation Y (m)   — robot left/right
    [2] dz  end-effector translation Z (m)   — robot up/down
    [3,4,5]  rotation (roll/pitch/yaw)
    [6]  gripper (0 = open, 1 = close)

For top-down-ish Bridge camera we map:
    image dx = +action[1] (lateral)
    image dy = -action[0] (forward translation → upward in image)
    arrow color = blue if gripper opening, red if closing
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Empirical scale: Bridge actions are unnormalized deltas; typical
# magnitudes are 0.005–0.05 m per step. We multiply to get pixel-ish arrows.
ARROW_SCALE_PX = 1200.0


@dataclass
class ActionArrow:
    dx: float          # image-space delta X (pixels)
    dy: float          # image-space delta Y (pixels)
    grip_close: float  # 0..1 — used to color the arrow head
    z_delta: float     # vertical delta (small visual mark)
    raw: np.ndarray    # original 7-D action

    @property
    def magnitude(self) -> float:
        return float(np.hypot(self.dx, self.dy))


def project_action(action: np.ndarray, scale: float = ARROW_SCALE_PX) -> ActionArrow:
    a = np.asarray(action, dtype=np.float32).flatten()
    if a.size < 7:
        a = np.concatenate([a, np.zeros(7 - a.size, dtype=np.float32)])
    return ActionArrow(
        dx=float(a[1] * scale),
        dy=float(-a[0] * scale),
        grip_close=float(np.clip(a[6], 0, 1)),
        z_delta=float(a[2]),
        raw=a[:7].copy(),
    )


def project_action_sequence(actions: np.ndarray) -> list[ActionArrow]:
    return [project_action(a) for a in actions]


def aggregate_trajectory(actions: np.ndarray) -> ActionArrow:
    """Sum a sequence of per-frame actions into a single 'net motion' arrow.

    Useful for the demo: one arrow per (scene, instruction) summarizing
    where the model wants to go over a short window.
    """
    if len(actions) == 0:
        return project_action(np.zeros(7))
    return project_action(actions.sum(axis=0))
