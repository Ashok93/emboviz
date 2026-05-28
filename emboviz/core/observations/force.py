"""Force / torque (wrench) observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


WrenchFrame = Literal["world", "ee", "sensor"]
WrenchUnits = Literal["N_Nm", "kgf_kgfm"]


@dataclass(frozen=True)
class ForceTorque:
    """A 6D wrench (force + torque) at a known frame.

    Sign and frame conventions matter; `frame` says where the wrench is
    expressed. Adapters are responsible for transforming into this
    representation if the sensor reports differently.
    """

    wrench: np.ndarray             # (6,) — [fx, fy, fz, tx, ty, tz]
    frame: WrenchFrame = "ee"
    units: WrenchUnits = "N_Nm"
