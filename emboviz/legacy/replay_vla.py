"""Replay an OpenVLA policy over a recorded BridgeV2 episode.

OpenVLA inference at bf16 is ~1–2s/frame on a 3090. Bridge episodes are
~30–80 frames at 5fps. Rolling a full episode is a couple minutes — fine.

The goal of this pass is two-fold:
  1. Confirm the policy outputs sensible actions on the recorded scene
     (sanity).
  2. Locate the most-divergent timestep (largest L2 distance between
     predicted action and recorded expert action) as the failure-frame
     candidate — that's where attribution is most interesting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from emboviz.dataset_bridge import BridgeEpisode
from emboviz.openvla import OpenVLAInference, VLAPrediction


@dataclass
class VLAReplayResult:
    predictions: list[VLAPrediction]   # one per frame
    predicted_actions: np.ndarray      # (T, 7)
    action_deviations: np.ndarray      # (T,) L2 distance to expert
    failure_frame_idx: int             # argmax of deviations (post-warmup)


def replay_vla(
    vla: OpenVLAInference,
    episode: BridgeEpisode,
    warmup_frames: int = 0,
    max_frames: int | None = None,
) -> VLAReplayResult:
    """Predict an action at every (sub-sampled) frame; record everything.

    `max_frames` lets us cap the rollout for cost — IG over a 70-frame episode
    means 70 expensive predictions; for a hypothesis check 20 is plenty.
    """
    T = episode.num_frames if max_frames is None else min(episode.num_frames, max_frames)

    predictions: list[VLAPrediction] = []
    for t in tqdm(range(T), desc="vla replay", unit="frame"):
        pred = vla.predict(episode.images[t], episode.instruction)
        predictions.append(pred)

    predicted_actions = np.stack([p.action for p in predictions])
    expert = episode.expert_actions[:T].cpu().numpy()
    deviations = np.linalg.norm(predicted_actions - expert, axis=-1)

    warmup = min(warmup_frames, T - 1)
    search = deviations.copy()
    search[:warmup] = -1.0
    failure_idx = int(np.argmax(search))

    return VLAReplayResult(
        predictions=predictions,
        predicted_actions=predicted_actions,
        action_deviations=deviations,
        failure_frame_idx=failure_idx,
    )


def pick_keyframes(result: VLAReplayResult, n: int) -> list[int]:
    """Pick n keyframes spanning the episode + guaranteeing the failure frame."""
    T = len(result.predictions)
    if n >= T:
        return list(range(T))
    spaced = np.linspace(0, T - 1, num=n - 1).round().astype(int).tolist()
    return sorted(set(spaced + [result.failure_frame_idx]))
