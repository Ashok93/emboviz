"""RLDS / TFDS episode source — for Open-X-Embodiment, RT-X, Octo data.

The community standard for cross-embodiment robot data: an `RLDS`
dataset (built on TensorFlow Datasets) contains episodes; each episode
is a sequence of ``step`` dicts with ``observation``, ``action``,
``is_terminal``, and ``language_instruction`` fields. Different
datasets put images, proprio, and gripper under different sub-keys,
so the adapter takes explicit ``camera_keys`` + ``state_key`` mappings
and refuses to guess.

Install:
  pip install 'emboviz[rlds]'

This pulls ``tensorflow_datasets`` (and a CPU-only tensorflow). We do
not train through this adapter — we only read the recorded episodes —
so the CPU build is enough.

Reference: Open-X-Embodiment (arXiv:2310.08864) for the format
contract; rlds docs at https://github.com/google-research/rlds.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from emboviz.core.observations import (
    GripperState,
    Proprioception,
    RGBImage,
)
from emboviz.core.profile import RobotProfile
from emboviz.core.types import Observations, Scene, Trajectory
from emboviz.datasets.base import EpisodeSource

# ``tensorflow`` + ``tensorflow_datasets`` are heavy ML deps and live
# behind the ``rlds`` extra. They're imported inside methods so the
# module itself loads in a core-only install.


GripperExtractor = Callable[[np.ndarray], tuple[np.ndarray, Optional[float]]]


def _identity_state(state: np.ndarray) -> tuple[np.ndarray, Optional[float]]:
    return state, None


class RLDSEpisodeSource(EpisodeSource):
    """RLDS / TFDS-backed episode source.

    Args:
      builder_name: TFDS builder name. For Open-X-Embodiment datasets
        this is typically the dataset's short name (e.g. ``"bridge"``,
        ``"fractal20220817_data"``, ``"taco_play"``). For a local
        dataset built with the `rlds_dataset_builder` tooling, the
        builder name registered there.
      data_dir: Optional override for TFDS data root. Defaults to the
        TFDS default (``~/tensorflow_datasets`` unless ``TFDS_DATA_DIR``
        is set).
      split: TFDS split to read; default ``"train"`` (the only split
        most OXE datasets ship).
      camera_keys: ``{scene_camera_name → step.observation key}`` map.
        Must include an entry named ``"primary"`` — emboviz never
        auto-aliases the first camera as primary. Example for Bridge:
        ``{"primary": "image_0", "wrist": "image_1"}``.
      state_key: optional key inside ``step.observation`` holding the
        proprioception vector. Set to ``None`` if your model doesn't
        need state.
      action_key: key for the per-step action vector (default
        ``"action"`` — the OXE convention).
      instruction_key: key for the language instruction. Usually
        ``"language_instruction"`` per step, but some OXE datasets put
        it in ``episode_metadata`` instead — pass ``None`` and the
        adapter falls back to ``episode_metadata`` lookup.
      profile: RobotProfile describing the robot's state + action
        convention. Required for diagnostics that consult the profile
        (e.g. action-dim naming in expert delta).
      gripper_extractor: split state into (proprio, gripper_value).
        Defaults to "no split" — pass a custom callable if your state
        layout includes a gripper component you want surfaced
        separately.
    """

    def __init__(
        self,
        builder_name: str,
        *,
        data_dir: Optional[str] = None,
        split: str = "train",
        camera_keys: dict[str, str],
        state_key: Optional[str] = None,
        action_key: str = "action",
        instruction_key: Optional[str] = "language_instruction",
        profile: Optional[RobotProfile] = None,
        gripper_extractor: GripperExtractor = _identity_state,
    ):
        if not camera_keys:
            raise ValueError(
                "RLDSEpisodeSource: camera_keys is required. RLDS datasets "
                "store images under arbitrary sub-keys (image_0 vs image vs "
                "rgb_static vs ...); we never guess."
            )
        if "primary" not in camera_keys:
            raise KeyError(
                f"RLDSEpisodeSource: camera_keys must contain a 'primary' "
                f"entry. Got {sorted(camera_keys)}. We never auto-alias the "
                "first declared camera — different RLDS datasets put the "
                "exterior view under different keys; the adapter caller "
                "must say which is the primary view."
            )
        self.builder_name     = builder_name
        self.data_dir         = data_dir
        self.split            = split
        self.camera_keys      = dict(camera_keys)
        self.state_key        = state_key
        self.action_key       = action_key
        self.instruction_key  = instruction_key
        self.profile          = profile
        self.gripper_extractor = gripper_extractor
        self.name             = f"rlds:{builder_name}"

        # TFDS dataset is cached lazily. Iteration is one-shot (TFDS
        # tf.data iterators), so we materialize episodes into a list on
        # first build. Indexed by episode_index = 0..N-1.
        self._episodes_cache: Optional[list[dict]] = None
        self._builder_info_cache: Optional[Any] = None

    # ── EpisodeSource interface ────────────────────────────────────

    def list_episodes(self) -> list[str]:
        eps = self._episodes()
        return [str(i) for i in range(len(eps))]

    def load_episode(self, episode_id: str) -> list[Scene]:
        eps = self._episodes()
        idx = int(episode_id)
        if idx < 0 or idx >= len(eps):
            raise IndexError(f"episode {idx} out of range (have {len(eps)})")
        return self._episode_to_scenes(eps[idx], idx)

    def load_episodes(self, episode_indices: list[int]) -> dict[int, list[Scene]]:
        eps = self._episodes()
        out: dict[int, list[Scene]] = {}
        for i in episode_indices:
            if i < 0 or i >= len(eps):
                raise IndexError(f"episode {i} out of range (have {len(eps)})")
            out[i] = self._episode_to_scenes(eps[i], i)
        return out

    def load_trajectory(self, episode_idx: int) -> Trajectory:
        scenes = self.load_episode(str(episode_idx))
        fps = float(scenes[0].metadata.get("fps", 10.0)) if scenes else 10.0
        return Trajectory(
            frames=scenes,
            frame_indices=list(range(len(scenes))),
            fps=fps,
            episode_id=str(episode_idx),
            source=f"{self.name}:{episode_idx}",
            metadata={"dataset": self.builder_name},
        )

    def all_instructions(self) -> list[str]:
        """Unique per-step instructions across every episode.

        For coverage analysis. Iterates every episode → every step,
        collects the instruction string. Cheap given the cached
        materialized episodes; first call may be slow on a large
        dataset because it forces full materialization.
        """
        out: set[str] = set()
        for ep in self._episodes():
            ep_meta = ep.get("episode_metadata") or {}
            ep_instr = _bytes_to_str(
                ep_meta.get("language_instruction")
                or ep_meta.get("natural_language_instruction")
                or ""
            )
            saw_step = False
            for step in ep["steps"]:
                instr = ""
                if self.instruction_key and self.instruction_key in step:
                    instr = _bytes_to_str(step[self.instruction_key])
                if not instr:
                    obs = step.get("observation") or {}
                    if self.instruction_key and self.instruction_key in obs:
                        instr = _bytes_to_str(obs[self.instruction_key])
                if instr:
                    out.add(instr)
                    saw_step = True
            if not saw_step and ep_instr:
                out.add(ep_instr)
        return sorted(out)

    def episode_lengths(self, episode_indices: list[int]) -> dict[int, int]:
        """Frame count per episode, from the cached materialized episodes.

        RLDS/TFDS is sequential: episodes are materialized once into
        :meth:`_episodes` (cached), after which the scene count is computed
        from the in-memory steps — no per-frame decode.
        """
        eps = self._episodes()
        out: dict[int, int] = {}
        for i in episode_indices:
            idx = int(i)
            if idx < 0 or idx >= len(eps):
                raise IndexError(f"episode {i} out of range (have {len(eps)})")
            out[idx] = len(self._episode_to_scenes(eps[idx], idx))
        return out

    def sample_frames(self, episode_offsets: dict[int, int]) -> dict[int, Scene]:
        """One frame per episode, from the cached materialized episodes.

        RLDS has no random per-step access, but the episodes are already in
        memory after the one-time materialization, so this builds the scenes
        and indexes the requested offset (no re-decode). An out-of-range offset
        is omitted from the result.
        """
        eps = self._episodes()
        out: dict[int, Scene] = {}
        for ep_idx, offset in episode_offsets.items():
            idx = int(ep_idx)
            if idx < 0 or idx >= len(eps):
                raise IndexError(f"episode {ep_idx} out of range (have {len(eps)})")
            scenes = self._episode_to_scenes(eps[idx], idx)
            if 0 <= int(offset) < len(scenes):
                out[idx] = scenes[int(offset)]
        return out

    # ── internals ──────────────────────────────────────────────────

    def _episodes(self) -> list[dict]:
        """Materialize all episodes into Python dicts. Cached after first call."""
        if self._episodes_cache is not None:
            return self._episodes_cache
        try:
            import tensorflow_datasets as tfds
            import tensorflow as tf  # noqa: F401  — TFDS needs it imported
        except ImportError as e:
            raise ImportError(
                "RLDSEpisodeSource needs the `rlds` extra. Install with: "
                "pip install 'emboviz[rlds]'. Underlying error: " + str(e)
            ) from e

        builder = tfds.builder(self.builder_name, data_dir=self.data_dir)
        builder.download_and_prepare()
        self._builder_info_cache = builder.info
        ds = builder.as_dataset(split=self.split)

        episodes: list[dict] = []
        for ep in ds:
            steps = list(ep["steps"].as_numpy_iterator())
            episodes.append({
                "steps":            steps,
                "episode_metadata": {
                    k: v.numpy() if hasattr(v, "numpy") else v
                    for k, v in (ep.get("episode_metadata") or {}).items()
                },
            })
        self._episodes_cache = episodes
        return episodes

    def _episode_to_scenes(self, ep: dict, ep_idx: int) -> list[Scene]:
        steps = ep["steps"]
        ep_meta = ep.get("episode_metadata") or {}

        # Episode-wide instruction fallback. Most OXE datasets repeat
        # the instruction per step; some put it once in episode_metadata.
        ep_instruction = _bytes_to_str(
            ep_meta.get("language_instruction")
            or ep_meta.get("natural_language_instruction")
            or ""
        )

        scenes: list[Scene] = []
        for fi, step in enumerate(steps):
            obs = step.get("observation") or {}
            images: dict[str, RGBImage] = {}
            for scene_cam, obs_key in self.camera_keys.items():
                arr = obs.get(obs_key)
                if arr is None:
                    raise KeyError(
                        f"RLDSEpisodeSource: camera '{scene_cam}' maps to "
                        f"observation key '{obs_key}' but step {fi} of "
                        f"episode {ep_idx} has no such key. Available "
                        f"observation keys: {sorted(obs)}."
                    )
                pil = Image.fromarray(np.asarray(arr, dtype=np.uint8))
                images[scene_cam] = RGBImage(data=pil, camera_id=scene_cam)

            proprio: Optional[Proprioception] = None
            gripper: Optional[GripperState] = None
            raw_state = None
            if self.state_key is not None and self.state_key in obs:
                raw_state = np.asarray(obs[self.state_key], dtype=np.float32).reshape(-1)
                proprio_vals, gripper_val = self.gripper_extractor(raw_state)
                state_convention = (
                    self.profile.state.convention if self.profile and self.profile.state
                    else "joint_angles"
                )
                proprio = Proprioception(values=proprio_vals.copy(), convention=state_convention)
                if gripper_val is not None and self.profile and self.profile.gripper:
                    gripper = GripperState(
                        value=float(gripper_val),
                        kind=self.profile.gripper.kind,
                        units=self.profile.gripper.units,
                    )

            # Per-step instruction wins over episode-wide fallback.
            instr = ""
            if self.instruction_key and self.instruction_key in step:
                instr = _bytes_to_str(step[self.instruction_key])
            if not instr and self.instruction_key and self.instruction_key in obs:
                instr = _bytes_to_str(obs[self.instruction_key])
            if not instr:
                instr = ep_instruction

            metadata: dict = {
                "fps":            10.0,   # OXE convention; profile can override later
                "frame_index":    fi,
                "episode_index":  ep_idx,
                "dataset":        self.builder_name,
            }
            if raw_state is not None:
                metadata["raw_state"] = raw_state.tolist()
            if self.action_key in step:
                metadata["expert_action"] = (
                    np.asarray(step[self.action_key], dtype=np.float32).reshape(-1).tolist()
                )

            scenes.append(Scene(
                observations=Observations(images=images, state=proprio, gripper=gripper),
                instruction=instr,
                profile=self.profile,
                metadata=metadata,
                scene_id=f"{self.name}:{ep_idx}:{fi}",
            ))
        return scenes


def _bytes_to_str(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray) and v.dtype.kind in ("S", "U", "O"):
        if v.size == 0:
            return ""
        return _bytes_to_str(v.item())
    return str(v) if v is not None else ""
