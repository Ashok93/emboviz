"""Tactile / contact sensor reading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TactileReading:
    """A reading from one named tactile sensor.

    Tactile data is wildly heterogeneous (capacitive grids, optical-based
    GelSight images, vibration arrays). We carry the raw ndarray and a
    sensor_id; per-sensor schema lives in the user's RobotProfile.extras.
    """

    data: np.ndarray
    sensor_id: str = "primary"
