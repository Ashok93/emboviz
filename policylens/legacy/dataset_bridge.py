"""Load one BridgeV2 episode (`IPEC-COMMUNITY/bridge_orig_lerobot`) with its
language instruction. OpenVLA was trained on this dataset (unnorm_key
"bridge_orig"), so it should produce sensible actions zero-shot.

We grab the primary camera view (`observation.images.image_0`) since that's
the one OpenVLA expects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


DATASET_REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"
PRIMARY_IMAGE_KEY = "observation.images.image_0"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


@dataclass
class BridgeEpisode:
    images: list[Image.Image]    # T PIL RGB images (256x256 native)
    states: torch.Tensor          # (T, 8) float32
    expert_actions: torch.Tensor  # (T, 7) float32
    instruction: str              # natural-language task description
    fps: float
    num_frames: int
    episode_idx: int


def load_bridge_episode(episode_idx: int = 0) -> BridgeEpisode:
    """Convenience wrapper: load one episode."""
    return load_bridge_episodes([episode_idx])[episode_idx]


def load_bridge_episodes(episode_indices: list[int]) -> dict[int, BridgeEpisode]:
    """Load several BridgeV2 episodes in ONE LeRobotDataset call.

    Loading episodes one-at-a-time is dramatically slower because each
    LeRobotDataset() init re-fetches the dataset metadata from HuggingFace.
    Batching them into a single call cuts per-episode cost from ~30s to ~2s.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    indices = sorted(set(episode_indices))
    dataset = LeRobotDataset(DATASET_REPO, episodes=indices)

    # `episodes=[...]` makes the dataset enumerate ONLY those episodes in
    # contiguous order. We need a per-episode-index slicing into the dataset
    # — use the episode_index column the sample carries.
    by_ep: dict[int, dict] = {idx: {"images": [], "states": [], "actions": []} for idx in indices}
    for i in range(dataset.num_frames):
        sample = dataset[i]
        ep_i = int(sample.get("episode_index", sample.get("episode_idx", -1)))
        if ep_i not in by_ep:
            # Fallback: when episode_index isn't surfaced for old datasets.
            ep_i = indices[0]
        by_ep[ep_i]["images"].append(_tensor_to_pil(sample[PRIMARY_IMAGE_KEY]))
        by_ep[ep_i]["states"].append(sample[STATE_KEY].to(torch.float32).reshape(-1))
        by_ep[ep_i]["actions"].append(sample[ACTION_KEY].to(torch.float32).reshape(-1))

    out: dict[int, BridgeEpisode] = {}
    for idx in indices:
        buf = by_ep[idx]
        if not buf["images"]:
            continue
        instruction = _resolve_instruction_for_episode(dataset, idx)
        out[idx] = BridgeEpisode(
            images=buf["images"],
            states=torch.stack(buf["states"]),
            expert_actions=torch.stack(buf["actions"]),
            instruction=instruction,
            fps=float(dataset.fps),
            num_frames=len(buf["images"]),
            episode_idx=idx,
        )
    return out


def _resolve_instruction_for_episode(dataset, episode_idx: int) -> str:
    """Look up the instruction for a specific episode (handles multi-episode loads)."""
    meta = dataset.meta
    tasks = getattr(meta, "tasks", None)

    # Find a sample belonging to this episode to read its task_index.
    target_task_idx: int | None = None
    for i in range(dataset.num_frames):
        sample = dataset[i]
        ep_i = int(sample.get("episode_index", sample.get("episode_idx", -1)))
        if ep_i == episode_idx and "task_index" in sample:
            target_task_idx = int(sample["task_index"])
            break
    if target_task_idx is None:
        return "(no instruction available)"
    if tasks is None:
        return f"(task #{target_task_idx})"
    if isinstance(tasks, dict):
        return str(tasks.get(target_task_idx, f"(task #{target_task_idx})"))
    if isinstance(tasks, (list, tuple)) and target_task_idx < len(tasks):
        entry = tasks[target_task_idx]
        return entry["task"] if isinstance(entry, dict) else str(entry)
    return f"(task #{target_task_idx})"


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """LeRobot returns video frames as (C, H, W) float in [0,1]."""
    a = t.detach().cpu().float().numpy()
    if a.ndim == 3 and a.shape[0] in (1, 3):  # CHW -> HWC
        a = a.transpose(1, 2, 0)
    if a.max() <= 1.5:
        a = a * 255.0
    a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def _resolve_instruction(dataset) -> str:
    """LeRobot stores instructions in `meta/tasks.jsonl`; per-frame samples
    carry only a `task_index`. We look up the single task that this episode is
    associated with (Bridge episodes are single-task) and return its string.
    """
    meta = dataset.meta
    # Newer lerobot exposes `tasks` as a dict {task_index: instruction}; older
    # versions expose a list. We handle both.
    tasks = getattr(meta, "tasks", None)
    if tasks is None:
        return "(no instruction available)"

    # Pull task indices that appear in this episode.
    sample = dataset[0]
    if "task_index" in sample:
        idx = int(sample["task_index"])
        if isinstance(tasks, dict):
            return str(tasks.get(idx, f"(task #{idx} not in tasks.jsonl)"))
        if isinstance(tasks, (list, tuple)) and idx < len(tasks):
            entry = tasks[idx]
            return entry["task"] if isinstance(entry, dict) else str(entry)

    # Fallback: episode-level task list
    episodes_meta = getattr(meta, "episodes", None)
    if episodes_meta is not None and len(episodes_meta) > 0:
        ep = episodes_meta[0] if isinstance(episodes_meta, list) else episodes_meta
        for k in ("task", "tasks"):
            if k in ep:
                v = ep[k]
                return v[0] if isinstance(v, (list, tuple)) else str(v)

    return "(no instruction available)"
