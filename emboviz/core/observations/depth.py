"""Depth map observation, keyed by camera."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


DepthUnits = Literal["meters", "millimeters"]


@dataclass(frozen=True)
class DepthMap:
    """A depth map aligned to a named camera."""

    data: np.ndarray               # (H, W), float32
    camera_id: str = "primary"
    units: DepthUnits = "meters"
