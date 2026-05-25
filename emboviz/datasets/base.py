"""Episode source — the protocol every dataset adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from emboviz.core.types import Scene


class EpisodeSource(ABC):
    """A source of Scenes — wraps a HF dataset, a local rollout, etc."""

    name: str

    @abstractmethod
    def list_episodes(self) -> list[str]:
        """All episode IDs available from this source."""

    @abstractmethod
    def load_episode(self, episode_id: str) -> list[Scene]:
        """Materialize one episode as a list of Scenes (one per frame)."""

    def load_first_scene(self, episode_id: str) -> Scene:
        scenes = self.load_episode(episode_id)
        if not scenes:
            raise ValueError(f"Episode {episode_id} is empty")
        return scenes[0]

    @abstractmethod
    def all_instructions(self) -> list[str]:
        """All unique instruction strings — for coverage analysis."""
