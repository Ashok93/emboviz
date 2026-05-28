"""Pre-shipped RobotProfile presets for common single-arm tabletop setups.

Each profile is a small data file; teams pick the closest match and
either use it directly or copy + customize. Custom robots get their own
profile in this directory (community contributions welcome).

Day-one profiles:
    franka_robotiq         — Franka Panda + Robotiq 2F-85 (the most common
                              open-source manipulation setup)
    ur5_robotiq            — UR5/UR10 + Robotiq 2F-85
    trossen_aloha_single   — Trossen ViperX-300 (single-arm ALOHA)
    bridge_orig            — re-export of the Bridge profile from datasets/

Planned: aloha_bimanual, unitree_h1, unitree_g1.
"""

from emboviz.datasets.lerobot_bridge import BRIDGE_PROFILE as bridge_orig
from emboviz.profiles.franka_robotiq import FRANKA_ROBOTIQ
from emboviz.profiles.trossen_aloha_single import TROSSEN_ALOHA_SINGLE
from emboviz.profiles.ur5_robotiq import UR5_ROBOTIQ

PROFILES = {
    "franka_robotiq": FRANKA_ROBOTIQ,
    "ur5_robotiq": UR5_ROBOTIQ,
    "trossen_aloha_single": TROSSEN_ALOHA_SINGLE,
    "bridge_orig": bridge_orig,
}

__all__ = [
    "PROFILES",
    "FRANKA_ROBOTIQ",
    "UR5_ROBOTIQ",
    "TROSSEN_ALOHA_SINGLE",
    "bridge_orig",
]
