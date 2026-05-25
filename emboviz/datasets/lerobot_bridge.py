"""BridgeV2 (IPEC-COMMUNITY/bridge_orig_lerobot) episode source.

Bridge has 53k episodes; we lazy-load. `load_episode` accepts integer IDs
and uses batched LeRobotDataset() calls when many episodes are requested.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image

from emboviz.core.types import Scene, Trajectory
from emboviz.datasets.base import EpisodeSource


DATASET_REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"
PRIMARY_IMAGE_KEY = "observation.images.image_0"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


class BridgeEpisodeSource(EpisodeSource):
    """Episode source backed by LeRobotDataset on the IPEC bridge mirror."""

    name = "bridge_v2"

    def __init__(self):
        self._meta_dataset = None  # populated on first call to all_instructions

    def list_episodes(self) -> list[str]:
        # Bridge has 53k. We don't enumerate by default.
        return [str(i) for i in range(53192)]

    def load_episode(self, episode_id: str) -> list[Scene]:
        return self.load_episodes([int(episode_id)])[int(episode_id)]

    def load_episodes(self, episode_indices: list[int]) -> dict[int, list[Scene]]:
        """Batched load — single LeRobotDataset init for all indices."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        indices = sorted(set(episode_indices))
        dataset = LeRobotDataset(DATASET_REPO, episodes=indices)
        out: dict[int, list[Scene]] = {i: [] for i in indices}

        for i in range(dataset.num_frames):
            sample = dataset[i]
            ep_i = int(sample.get("episode_index", sample.get("episode_idx", indices[0])))
            if ep_i not in out:
                continue
            instruction = self._resolve_instruction(dataset, ep_i)
            out[ep_i].append(Scene(
                image=self._tensor_to_pil(sample[PRIMARY_IMAGE_KEY]),
                instruction=instruction,
                metadata={
                    "state": sample[STATE_KEY].to(torch.float32).reshape(-1).tolist(),
                    "expert_action": sample[ACTION_KEY].to(torch.float32).reshape(-1).tolist(),
                    "fps": float(dataset.fps),
                    "frame_index": i,
                    "episode_index": ep_i,
                    "dataset": DATASET_REPO,
                },
                scene_id=f"bridge:{ep_i}:{len(out[ep_i])}",
            ))
        return out

    def load_trajectory(self, episode_idx: int) -> Trajectory:
        """Load one episode as a Trajectory (Scenes in time order)."""
        scenes = self.load_episode(str(episode_idx))
        # FPS comes from the metadata; pull it from any scene that has it.
        fps = float(scenes[0].metadata.get("fps", 5.0)) if scenes else 5.0
        return Trajectory(
            frames=scenes,
            frame_indices=list(range(len(scenes))),
            fps=fps,
            episode_id=str(episode_idx),
            source=f"bridge:{episode_idx}",
            metadata={"dataset": DATASET_REPO},
        )

    def all_instructions(self) -> list[str]:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        if self._meta_dataset is None:
            self._meta_dataset = LeRobotDataset(DATASET_REPO, episodes=[0])
        tasks = getattr(self._meta_dataset.meta, "tasks", None)
        if tasks is None:
            return []
        if isinstance(tasks, dict):
            items = list(tasks.values())
        else:
            items = list(tasks)
        out = []
        for it in items:
            if isinstance(it, dict) and "task" in it:
                out.append(str(it["task"]))
            elif isinstance(it, str):
                out.append(it)
        return out

    # ---- helpers ----------------------------------------------------------

    def _tensor_to_pil(self, t) -> Image.Image:
        a = t.detach().cpu().float().numpy() if hasattr(t, "detach") else np.asarray(t)
        if a.ndim == 3 and a.shape[0] in (1, 3):
            a = a.transpose(1, 2, 0)
        if a.max() <= 1.5:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
        return Image.fromarray(a)

    def _resolve_instruction(self, dataset, episode_idx: int) -> str:
        meta = dataset.meta
        tasks = getattr(meta, "tasks", None)
        target_idx: Optional[int] = None
        for i in range(dataset.num_frames):
            sample = dataset[i]
            ep_i = int(sample.get("episode_index", sample.get("episode_idx", -1)))
            if ep_i == episode_idx and "task_index" in sample:
                target_idx = int(sample["task_index"])
                break
        if target_idx is None or tasks is None:
            return f"(task #{target_idx})" if target_idx is not None else "(no instruction)"
        if isinstance(tasks, dict):
            return str(tasks.get(target_idx, f"(task #{target_idx})"))
        if isinstance(tasks, (list, tuple)) and target_idx < len(tasks):
            entry = tasks[target_idx]
            return entry["task"] if isinstance(entry, dict) else str(entry)
        return f"(task #{target_idx})"
