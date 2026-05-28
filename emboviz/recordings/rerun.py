"""Rerun ``.rrd`` deployment-recording adapter.

Reads recordings that the user produced by calling ``rr.log(...)`` in
their model inference loop. The Python rerun-sdk's read APIs are still
experimental (as of rerun-sdk 0.22). This adapter is a stub that errors
helpfully when invoked; full support lands when the read API stabilizes.

Install when implemented:
  pip install 'emboviz[rerun]'

Until then, users with Rerun-only recordings have two options:
  1. Use ``rerun convert <in>.rrd <out>.mcap`` to convert to MCAP and
     read via MCAPRecording.
  2. Pipe the same observations through MCAPRecording directly by
     pointing it at the original ROS bag the inference loop also
     recorded (a common dual-record pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from emboviz.core.profile import RobotProfile
from emboviz.core.types import Scene, Trajectory
from emboviz.datasets.base import EpisodeSource


class RerunRecording(EpisodeSource):
    """Stub adapter — not yet implemented."""

    def __init__(
        self,
        path: str,
        *,
        entity_map: Optional[dict[str, str]] = None,
        target_rate_hz: float = 10.0,
        profile: Optional[RobotProfile] = None,
    ):
        self.path = Path(path)
        self.entity_map = entity_map or {}
        self.target_rate_hz = target_rate_hz
        self.profile = profile
        self.name = f"rerun:{self.path.name}"
        raise NotImplementedError(
            "RerunRecording is not yet implemented — Rerun's read APIs "
            "are still experimental. Workarounds:\n"
            "  1. ``rerun convert path/to/recording.rrd path/to/recording.mcap``\n"
            "     then use emboviz.recordings.MCAPRecording on the .mcap.\n"
            "  2. If you have the original ROS bag, point MCAPRecording at it.\n"
            "Tracking issue: TBD."
        )

    def list_episodes(self) -> list[str]:
        return []

    def load_episode(self, episode_id: str) -> list[Scene]:
        raise NotImplementedError

    def load_trajectory(self, episode_idx: int) -> Trajectory:
        raise NotImplementedError

    def all_instructions(self) -> list[str]:
        return []
