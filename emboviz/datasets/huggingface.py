"""Generic HuggingFace `datasets`-library episode source.

For any HF dataset that isn't in the LeRobot v3 format. The user supplies
a mapping function that turns one HF row into a `Scene`. This is the
escape hatch for community datasets with bespoke schemas — the team
writes ~10 lines to map their fields, everything else works.

Lazy imports the `datasets` package so this module is free to import
without it.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from emboviz.core.profile import RobotProfile
from emboviz.core.types import Scene, Trajectory
from emboviz.datasets.base import EpisodeSource


# Maps one HF row (a dict) + a per-row index to a Scene.
RowToScene = Callable[[dict, int], Scene]


class HuggingFaceEpisodeSource(EpisodeSource):
    """Wrap any HuggingFace `datasets` dataset as an EpisodeSource.

    Designed for datasets where each row is one frame, and episode
    grouping is by an "episode_index" or similar field. The caller
    supplies `row_to_scene` to convert one row to a typed Scene.
    """

    def __init__(
        self,
        repo_id: str,
        profile: RobotProfile,
        row_to_scene: RowToScene,
        *,
        split: str = "train",
        episode_field: str = "episode_index",
    ):
        self.repo_id = repo_id
        self.profile = profile
        self.row_to_scene = row_to_scene
        self.split = split
        self.episode_field = episode_field
        self.name = f"hf:{repo_id}"
        self._dataset = None

    def _load(self):
        if self._dataset is None:
            from datasets import load_dataset
            self._dataset = load_dataset(self.repo_id, split=self.split)
        return self._dataset

    def list_episodes(self) -> list[str]:
        ds = self._load()
        if self.episode_field not in ds.column_names:
            return ["0"]   # single-episode dataset
        eps = sorted(set(int(x) for x in ds[self.episode_field]))
        return [str(e) for e in eps]

    def load_episode(self, episode_id: str) -> list[Scene]:
        ds = self._load()
        target = int(episode_id)
        scenes: list[Scene] = []
        for i, row in enumerate(ds):
            ep = int(row.get(self.episode_field, 0))
            if ep != target:
                continue
            scenes.append(self.row_to_scene(row, len(scenes)))
        return scenes

    def load_trajectory(self, episode_idx: int) -> Trajectory:
        scenes = self.load_episode(str(episode_idx))
        return Trajectory(
            frames=scenes,
            frame_indices=list(range(len(scenes))),
            fps=float(scenes[0].metadata.get("fps", 0.0)) if scenes else 0.0,
            episode_id=str(episode_idx),
            source=f"{self.name}:{episode_idx}",
            metadata={"dataset": self.repo_id},
        )

    def all_instructions(self) -> list[str]:
        ds = self._load()
        # Heuristic: try common instruction fields
        for field in ("instruction", "task", "language_instruction", "text"):
            if field in ds.column_names:
                return sorted(set(str(x) for x in ds[field] if x))
        return []
