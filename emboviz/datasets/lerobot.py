"""Generic LeRobot v3 episode source.

Wraps `lerobot.datasets.LeRobotDataset` and emits typed `Scene`s with
properly populated `Observations`. Any team using LeRobot v3 (Bridge,
ALOHA, PushT, custom uploads to HuggingFace under the LeRobot schema)
can plug in by constructing one of these with the right key mappings.

The Bridge adapter (`emboviz.datasets.lerobot_bridge`) is one specific
configuration of this generic class.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

from emboviz.core.observations import (
    GripperState,
    Proprioception,
    RGBImage,
)
from emboviz.core.profile import RobotProfile
from emboviz.core.types import Observations, Scene, Trajectory
from emboviz.datasets.base import EpisodeSource


# Function that, given a raw state ndarray from the dataset, returns
# (proprioception_values, gripper_value_or_None). Lets the user extract a
# gripper component from the state vector without baking it into the schema.
GripperExtractor = Callable[[np.ndarray], tuple[np.ndarray, Optional[float]]]


def _identity_state(state: np.ndarray) -> tuple[np.ndarray, Optional[float]]:
    """Default: proprio is the whole state, no gripper extraction."""
    return state, None


class LeRobotEpisodeSource(EpisodeSource):
    """Episode source backed by a LeRobotDataset on HuggingFace Hub.

    The fields you must supply:
      • `repo_id`     — HF dataset repo
      • `profile`     — RobotProfile for this robot/dataset combination
      • `image_keys`  — {camera_name → dataset key}; the first key is treated
                        as the "primary" camera if no key named "primary" exists

    Optional:
      • `state_key`         — dataset key for proprioception
      • `action_key`        — dataset key for expert action (stored in metadata)
      • `gripper_extractor` — splits raw state into (proprio, gripper); see
                              `_identity_state`
      • `n_episodes`        — total episode count (used by list_episodes)
    """

    def __init__(
        self,
        repo_id: str,
        profile: RobotProfile,
        image_keys: dict[str, str],
        *,
        state_key: Optional[str] = None,
        action_key: Optional[str] = None,
        gripper_extractor: GripperExtractor = _identity_state,
        n_episodes: int = 1_000_000,
    ):
        if not image_keys:
            raise ValueError("image_keys must have at least one entry")
        self.repo_id = repo_id
        self.profile = profile
        self.image_keys = dict(image_keys)
        self.state_key = state_key
        self.action_key = action_key
        self.gripper_extractor = gripper_extractor
        self._n_episodes = n_episodes
        self.name = f"lerobot:{repo_id}"
        self._meta_dataset = None
        # Cache LeRobotDataset instances keyed by the frozen tuple of
        # episode indices. Each instantiation hits HF for ~50 tree-listing
        # API calls; the pool builder samples 8-30 episodes per call, so
        # without this cache we'd burn the 1000-req/5-min rate limit on
        # any non-trivial dataset (Bridge has 50K episodes → paginated
        # tree listing). The cache makes batched loads free on repeat.
        self._dataset_cache: dict[tuple[int, ...], object] = {}
        self._dataset_cache_max = 8

    # ----- EpisodeSource interface -----------------------------------

    def list_episodes(self) -> list[str]:
        return [str(i) for i in range(self._n_episodes)]

    def load_episode(self, episode_id: str) -> list[Scene]:
        return self.load_episodes([int(episode_id)])[int(episode_id)]

    def load_episodes(self, episode_indices: list[int]) -> dict[int, list[Scene]]:
        """Batched load — single LeRobotDataset init for all indices.

        We tolerate both lerobot layouts:
          • ``lerobot.datasets.lerobot_dataset``   (≥0.3 — the modern path)
          • ``lerobot.common.datasets.lerobot_dataset``  (0.1.x — vendored
            by openpi and other VLA projects that pin the legacy stub)

        ``self.repo_id`` may be:
          • a HuggingFace repo id (``namespace/dataset``) — standard case
          • a local filesystem path containing a lerobot-format dataset
            (``meta/``, ``data/``, ``videos/``) — used for NVIDIA-shipped
            demo datasets that don't live on HF
        Local paths are detected and routed via the ``root=`` kwarg so
        lerobot skips its hub lookup entirely.
        """
        import os
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        indices = sorted(set(episode_indices))
        cache_key = tuple(indices)
        dataset = self._dataset_cache.get(cache_key)
        if dataset is None:
            is_local = os.path.isdir(self.repo_id) or self.repo_id.startswith("/")
            if is_local:
                try:
                    dataset = LeRobotDataset(
                        "local", root=self.repo_id, episodes=indices,
                    )
                except TypeError:
                    dataset = LeRobotDataset(
                        "local",
                        download_videos=False,
                        cache_dir=os.path.dirname(self.repo_id.rstrip("/")),
                        episodes=indices,
                    )
            else:
                dataset = LeRobotDataset(self.repo_id, episodes=indices)
            self._dataset_cache[cache_key] = dataset
            if len(self._dataset_cache) > self._dataset_cache_max:
                self._dataset_cache.pop(next(iter(self._dataset_cache)))
        out: dict[int, list[Scene]] = {i: [] for i in indices}

        for i in range(dataset.num_frames):
            sample = dataset[i]
            ep_i = int(sample.get("episode_index", sample.get("episode_idx", indices[0])))
            if ep_i not in out:
                continue
            instruction = self._resolve_instruction(dataset, sample)
            scene = self._build_scene(sample, instruction, ep_i, len(out[ep_i]), dataset.fps)
            out[ep_i].append(scene)
        return out

    def load_trajectory(self, episode_idx: int) -> Trajectory:
        scenes = self.load_episode(str(episode_idx))
        fps = float(scenes[0].metadata.get("fps", 5.0)) if scenes else 5.0
        return Trajectory(
            frames=scenes,
            frame_indices=list(range(len(scenes))),
            fps=fps,
            episode_id=str(episode_idx),
            source=f"{self.name}:{episode_idx}",
            metadata={"dataset": self.repo_id},
        )

    def all_instructions(self) -> list[str]:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        if self._meta_dataset is None:
            self._meta_dataset = LeRobotDataset(self.repo_id, episodes=[0])
        tasks = getattr(self._meta_dataset.meta, "tasks", None)
        if tasks is None:
            return []
        if isinstance(tasks, dict):
            items = list(tasks.values())
        else:
            items = list(tasks)
        out: list[str] = []
        for it in items:
            if isinstance(it, dict) and "task" in it:
                out.append(str(it["task"]))
            elif isinstance(it, str):
                out.append(it)
        return out

    # ----- internals -------------------------------------------------

    def _build_scene(
        self, sample: dict, instruction: str,
        episode_idx: int, frame_offset: int, fps: float,
    ) -> Scene:
        # Multi-cam: pull every camera the user declared.
        images: dict[str, RGBImage] = {}
        for cam_name, key in self.image_keys.items():
            if key not in sample:
                continue
            pil = self._tensor_to_pil(sample[key])
            images[cam_name] = RGBImage(data=pil, camera_id=cam_name)

        # Strict: every dataset adapter MUST declare which camera is "primary"
        # explicitly via image_keys. We do not silently alias the first
        # declared camera as primary — that pattern routinely puts the wrong
        # viewpoint into single-cam diagnostics. If you want a camera named
        # "primary", add it to image_keys: {"primary": "observation.images.X"}.
        if "primary" not in images and images:
            raise KeyError(
                f"Dataset adapter for repo_id={self.repo_id!r} loaded "
                f"cameras {sorted(images)} but none of them are named "
                "'primary'. Add an explicit \"primary\" entry to "
                "image_keys so the framework knows which view is the "
                "main exterior camera. We never auto-alias because the "
                "first declared key is not always the semantically-primary one."
            )

        proprio: Optional[Proprioception] = None
        gripper: Optional[GripperState] = None
        raw_state = None
        if self.state_key and self.state_key in sample:
            raw_state = sample[self.state_key].to(torch.float32).reshape(-1).numpy()
            proprio_vals, gripper_val = self.gripper_extractor(raw_state)
            state_convention = (
                self.profile.state.convention if self.profile.state is not None
                else "joint_angles"
            )
            proprio = Proprioception(values=proprio_vals.copy(), convention=state_convention)
            if gripper_val is not None and self.profile.gripper is not None:
                gripper = GripperState(
                    value=float(gripper_val),
                    kind=self.profile.gripper.kind,
                    units=self.profile.gripper.units,
                )

        obs = Observations(images=images, state=proprio, gripper=gripper)

        metadata: dict = {
            "fps": float(fps),
            "frame_index": int(sample.get("frame_index", frame_offset)),
            "episode_index": episode_idx,
            "dataset": self.repo_id,
        }
        if raw_state is not None:
            metadata["raw_state"] = raw_state.tolist()
        if self.action_key and self.action_key in sample:
            metadata["expert_action"] = (
                sample[self.action_key].to(torch.float32).reshape(-1).tolist()
            )

        return Scene(
            observations=obs,
            instruction=instruction,
            profile=self.profile,
            metadata=metadata,
            scene_id=f"{self.name}:{episode_idx}:{frame_offset}",
        )

    def _tensor_to_pil(self, t) -> Image.Image:
        """Convert a lerobot image tensor → PIL.Image.

        Strict dtype handling: floating tensors are assumed [0, 1] normalized
        and rescaled to [0, 255] uint8; integer tensors are assumed already
        in [0, 255] and only clipped. We never use a "if max ≤ 1.5 multiply"
        heuristic — a genuinely-dark uint8 frame can have max < 2 and the
        heuristic would overflow that frame silently.
        """
        if hasattr(t, "detach"):
            raw = t.detach().cpu().numpy()
        else:
            raw = np.asarray(t)
        if raw.ndim == 3 and raw.shape[0] in (1, 3):
            raw = raw.transpose(1, 2, 0)
        if np.issubdtype(raw.dtype, np.floating):
            a = (raw * 255.0).astype(np.float32)
        elif np.issubdtype(raw.dtype, np.integer):
            a = raw.astype(np.float32)
        else:
            raise TypeError(
                f"LeRobotEpisodeSource._tensor_to_pil: unsupported dtype "
                f"{raw.dtype}. Expected floating ([0,1] normalized) or "
                "integer ([0,255]) image tensor — the dataset adapter must "
                "produce one of these. No silent conversion."
            )
        a = np.clip(a, 0, 255).astype(np.uint8)
        return Image.fromarray(a)

    def _resolve_instruction(self, dataset, sample: dict) -> str:
        """Look up the instruction string for this sample's task_index."""
        meta = dataset.meta
        tasks = getattr(meta, "tasks", None)
        task_idx = int(sample.get("task_index", -1)) if "task_index" in sample else -1
        if task_idx < 0 or tasks is None:
            return ""
        if isinstance(tasks, dict):
            return str(tasks.get(task_idx, ""))
        if isinstance(tasks, (list, tuple)) and task_idx < len(tasks):
            entry = tasks[task_idx]
            return entry["task"] if isinstance(entry, dict) else str(entry)
        return ""
