"""Episode source adapters — one per data format.

Every team's rollouts live in one of:
  • LeRobot v3       — the dominant open-source robotics dataset format
  • HuggingFace generic — any other HF-hosted dataset
  • Rerun .rrd       — recordings from rerun.io
  • Foxglove .mcap   — rosbag2 / Foxglove recordings
  • (RLDS, ROS bag native — planned)

Each adapter implements `EpisodeSource` and emits Scenes with typed
Observations populated from the format's native fields.
"""

from emboviz.datasets.base import EpisodeSource
from emboviz.datasets.huggingface import HuggingFaceEpisodeSource
from emboviz.datasets.lerobot import LeRobotEpisodeSource
from emboviz.datasets.lerobot_aloha import (
    ALOHA_BIMANUAL_PROFILE_1CAM,
    ALOHA_BIMANUAL_PROFILE_4CAM,
    AlohaSimInsertionSource,
    AlohaSimTransferCubeSource,
    AlohaStatic4CamSource,
)
from emboviz.datasets.lerobot_bridge import BridgeEpisodeSource, BRIDGE_PROFILE
from emboviz.datasets.lerobot_droid import (
    DROID_PROFILE,
    GR00T_DROID_PROFILE,
    Droid100Source,
    DroidFullSource,
    GR00TDroidSampleSource,
)
from emboviz.datasets.lerobot_libero import (
    LIBERO_PROFILE,
    PI_LIBERO_PROFILE,
    Libero10Source,
    LiberoGoalSource,
    LiberoObjectSource,
    LiberoSpatialSource,
    PhysicalIntelligenceLiberoSource,
)

__all__ = [
    "EpisodeSource",
    "HuggingFaceEpisodeSource",
    "LeRobotEpisodeSource",
    "BridgeEpisodeSource",
    "BRIDGE_PROFILE",
    "ALOHA_BIMANUAL_PROFILE_1CAM",
    "ALOHA_BIMANUAL_PROFILE_4CAM",
    "AlohaSimTransferCubeSource",
    "AlohaSimInsertionSource",
    "AlohaStatic4CamSource",
    "LIBERO_PROFILE",
    "LiberoSpatialSource",
    "LiberoObjectSource",
    "LiberoGoalSource",
    "Libero10Source",
    "DROID_PROFILE",
    "Droid100Source",
    "DroidFullSource",
    "GR00T_DROID_PROFILE",
    "GR00TDroidSampleSource",
    "PI_LIBERO_PROFILE",
    "PhysicalIntelligenceLiberoSource",
]


def __getattr__(name):
    # Lazy access to optional-dep adapters so module imports cleanly without
    # `rerun-sdk` or `mcap` installed.
    if name == "RerunEpisodeSource":
        from emboviz.datasets.rerun import RerunEpisodeSource
        return RerunEpisodeSource
    if name == "FoxgloveEpisodeSource":
        from emboviz.datasets.foxglove import FoxgloveEpisodeSource
        return FoxgloveEpisodeSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
