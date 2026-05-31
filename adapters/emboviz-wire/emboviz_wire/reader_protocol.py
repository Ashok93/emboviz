"""EpisodeSource — the contract every dataset reader implements.

This is the dataset-side analogue of :class:`emboviz_wire.model_protocol.
VLAModel`. A reader turns a stored dataset (LeRobot / HDF5 / RLDS / a
custom rollout) into the framework's universal :class:`Scene` /
:class:`Trajectory` types. Diagnostics and the runner only ever call the
methods declared here — they never import a specific dataset library — so
swapping formats, or isolating a heavy/conflicting reader into its own
worker venv, is invisible to the rest of the system.

The contract is deliberately tiny and format-agnostic:

  * ``list_episodes``      — which episodes exist.
  * ``load_episode(s)``    — materialize episode(s) as ``Scene`` lists.
  * ``load_trajectory``    — one episode as a typed ``Trajectory``.
  * ``all_instructions``   — every unique instruction string.
  * ``name``               — a stable dataset identity (used for caching).

``load_episodes`` and ``load_trajectory`` ship default implementations on
top of ``load_episode`` so a minimal reader only implements three
abstract methods; readers with a cheaper batched path (LeRobot loads many
episodes from one dataset handle) override ``load_episodes``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from emboviz_wire.types import Scene, Trajectory


class EpisodeSource(ABC):
    """A source of Scenes — wraps a HF/local dataset, a rollout, etc."""

    #: Stable identity for this source (e.g. ``"lerobot:<repo_id>"``).
    #: Used by the modality-pool cache key, so it must be deterministic
    #: for a given dataset.
    name: str = ""

    # ----- abstract: the three things every reader must provide -------

    @abstractmethod
    def list_episodes(self) -> list[str]:
        """All episode IDs available from this source."""

    @abstractmethod
    def load_episode(self, episode_id: str) -> list[Scene]:
        """Materialize one episode as a list of Scenes (one per frame)."""

    @abstractmethod
    def all_instructions(self) -> list[str]:
        """All unique instruction strings declared by the dataset."""

    # ----- concrete defaults (override for efficiency) ----------------

    def load_episodes(self, episode_indices: list[int]) -> dict[int, list[Scene]]:
        """Load several episodes, keyed by index.

        Default loops :meth:`load_episode`; readers with a batched path
        (one dataset handle for many episodes) override this.
        """
        return {int(i): self.load_episode(str(int(i))) for i in episode_indices}

    def episode_lengths(self, episode_indices: list[int]) -> dict[int, int]:
        """Frame count for each episode, keyed by index.

        Default materializes each episode via :meth:`load_episode`; readers
        whose metadata records episode lengths override this to answer
        without decoding any frame.
        """
        return {int(i): len(self.load_episode(str(int(i)))) for i in episode_indices}

    def sample_frames(self, episode_offsets: dict[int, int]) -> dict[int, "Scene"]:
        """Materialize one frame per episode — ``{episode_idx: frame_offset}``
        → ``{episode_idx: Scene}``.

        Default loads each episode and indexes the requested offset; readers
        with random per-frame access override this to decode ONLY the
        requested frames (the modality pool needs one frame per episode, not
        the whole episode). An out-of-range offset is omitted from the result.
        """
        out: dict[int, "Scene"] = {}
        for ep_idx, offset in episode_offsets.items():
            scenes = self.load_episode(str(int(ep_idx)))
            if 0 <= int(offset) < len(scenes):
                out[int(ep_idx)] = scenes[int(offset)]
        return out

    def load_trajectory(self, episode_idx: int) -> Trajectory:
        """Load one episode as a typed :class:`Trajectory`.

        ``fps`` is read from the first frame's ``metadata["fps"]`` if
        present (every shipped reader stamps it), defaulting to 5.0.
        """
        scenes = self.load_episode(str(episode_idx))
        fps = float(scenes[0].metadata.get("fps", 5.0)) if scenes else 5.0
        return Trajectory(
            frames=scenes,
            frame_indices=list(range(len(scenes))),
            fps=fps,
            episode_id=str(episode_idx),
            source=f"{self.name}:{episode_idx}",
            metadata={"dataset": self.name},
        )

    def load_first_scene(self, episode_id: str) -> Scene:
        scenes = self.load_episode(episode_id)
        if not scenes:
            raise ValueError(f"Episode {episode_id} is empty")
        return scenes[0]
