"""Action-arrow rendering primitive."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ARROW_SCALE_PX = 1200.0


@dataclass
class ProjectedArrow:
    dx: float
    dy: float
    grip_close: float
    z_delta: float


def project_action(action: np.ndarray, scale: float = ARROW_SCALE_PX) -> ProjectedArrow:
    a = np.asarray(action, dtype=np.float32).flatten()
    if a.size < 7:
        a = np.concatenate([a, np.zeros(7 - a.size, dtype=np.float32)])
    return ProjectedArrow(
        dx=float(a[1] * scale),
        dy=float(-a[0] * scale),
        grip_close=float(np.clip(a[6], 0, 1)),
        z_delta=float(a[2]),
    )


def draw_arrow(ax, x: float, y: float, arrow: ProjectedArrow) -> None:
    color = "#d62728" if arrow.grip_close > 0.5 else "#1f77b4"
    ax.annotate("", xy=(x + arrow.dx, y + arrow.dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", lw=4.5, color="white"))
    ax.annotate("", xy=(x + arrow.dx, y + arrow.dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=color))
    ax.plot(x, y, "o", color="white", markersize=10)
    ax.plot(x, y, "o", color=color, markersize=6)
