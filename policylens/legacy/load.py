"""Load the pretrained Diffusion Policy and one episode of LeRobot PushT.

PushT is a 2D top-down task: push a T-shaped block onto a T-shaped target.
The pretrained checkpoint `lerobot/diffusion_pusht` consumes:
  - `observation.image`  : (B, C, H, W) RGB image, [0,1]
  - `observation.state`  : (B, 2) end-effector xy
and produces a 2D action (xy target).

This module hides all the lerobot import/version quirks behind two functions:
  - load_policy(device) -> policy
  - load_episode(episode_idx) -> EpisodeFrames
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


PRETRAINED_REPO = "lerobot/diffusion_pusht"
DATASET_REPO = "lerobot/pusht"


@dataclass
class EpisodeFrames:
    """One full episode, ordered by timestep."""

    images: torch.Tensor          # (T, C, H, W) float32 in [0,1]
    states: torch.Tensor          # (T, state_dim) float32
    expert_actions: torch.Tensor  # (T, action_dim) float32 — recorded ground-truth
    fps: float
    num_frames: int


def load_policy(device: str = "cuda"):
    """Load the pretrained Diffusion Policy checkpoint and put it in eval mode.

    Diffusion Policy denoises action chunks conditioned on (image, state).
    For attribution we want the underlying network to be gradient-traceable,
    so we explicitly do NOT torch.no_grad() at the model level here — callers
    are responsible for grad context when computing attribution vs inference.
    """
    # Import inside the function so import errors surface where they're actionable.
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    policy = DiffusionPolicy.from_pretrained(PRETRAINED_REPO)
    policy.to(device)
    policy.eval()
    return policy


def load_episode(episode_idx: int = 0) -> EpisodeFrames:
    """Load one full episode from the PushT dataset, materialized into tensors.

    Returns frames in temporal order. The dataset is small (~few hundred MB)
    and one episode is at most a few hundred frames, so we materialize eagerly.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # episodes=[episode_idx] restricts to a single episode — fast load.
    dataset = LeRobotDataset(DATASET_REPO, episodes=[episode_idx])

    images = []
    states = []
    actions = []
    for i in range(dataset.num_frames):
        sample = dataset[i]
        images.append(_as_chw_float(sample["observation.image"]))
        states.append(_as_1d_float(sample["observation.state"]))
        actions.append(_as_1d_float(sample["action"]))

    return EpisodeFrames(
        images=torch.stack(images),
        states=torch.stack(states),
        expert_actions=torch.stack(actions),
        fps=float(dataset.fps),
        num_frames=int(dataset.num_frames),
    )


def _as_chw_float(x: Any) -> torch.Tensor:
    """Coerce an observation image into a (C, H, W) float32 tensor in [0,1]."""
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if t.dtype != torch.float32:
        t = t.to(torch.float32)
    if t.max() > 1.5:  # uint8-ish range
        t = t / 255.0
    if t.dim() == 3 and t.shape[-1] in (1, 3):  # HWC -> CHW
        t = t.permute(2, 0, 1)
    return t.contiguous()


def _as_1d_float(x: Any) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    return t.to(torch.float32).reshape(-1)


def build_observation_batch(
    image: torch.Tensor, state: torch.Tensor, device: str
) -> dict[str, torch.Tensor]:
    """Wrap a single timestep into the dict shape Diffusion Policy expects.

    The policy is trained on batched inputs, so we add a leading B=1 axis.
    """
    return {
        "observation.image": image.unsqueeze(0).to(device),
        "observation.state": state.unsqueeze(0).to(device),
    }
