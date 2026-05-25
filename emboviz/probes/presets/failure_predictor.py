"""Failure-prediction probe preset.

Predicts P(failure) per frame from the model's hidden states at the
action-prediction position. Trained on (success, failure) labeled frames
extracted from real rollouts.

Labelling strategy (default — `label_frames_from_deviation`):
  • Per episode, compare each frame's PREDICTED action to the dataset's
    expert action.
  • Per-episode max deviation; episodes with max > threshold → "failure"
    episode, contributing failure-labeled frames around the divergence
    moment.

Reference: SAFE (Gu et al., arXiv 2506.09937) — VLA failure probes trained
on fusion-band internal features generalize across tasks.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


FAILURE_PROBE_NAME = "failure_predictor"


def label_frames_from_deviation(
    per_episode_predicted: list[np.ndarray],     # list of (T, action_dim) arrays
    per_episode_expert: list[np.ndarray],        # same shape
    failure_threshold: float = 0.30,             # max-deviation cutoff per episode
    spread_frames: int = 3,                      # widen the failure label around the spike
) -> tuple[list[int], list[tuple[int, int]]]:
    """Produce per-frame labels: 1 = failure, 0 = success.

    Returns:
      labels — list of 0/1 ints, flattened over (episode, frame)
      indices — list of (episode_idx, frame_idx) for each label, in order
    """
    labels: list[int] = []
    indices: list[tuple[int, int]] = []
    for ep_i, (pred, expert) in enumerate(zip(per_episode_predicted, per_episode_expert)):
        T = min(len(pred), len(expert))
        per_frame_dev = np.linalg.norm(pred[:T] - expert[:T], axis=-1)
        max_dev = float(per_frame_dev.max()) if T > 0 else 0.0
        if max_dev < failure_threshold:
            # All frames in this episode are SUCCESS-labeled.
            for t in range(T):
                labels.append(0)
                indices.append((ep_i, t))
        else:
            # Mark frames within `spread_frames` of the spike as FAILURE,
            # rest as SUCCESS within the same episode.
            spike = int(np.argmax(per_frame_dev))
            for t in range(T):
                is_failure = abs(t - spike) <= spread_frames
                labels.append(1 if is_failure else 0)
                indices.append((ep_i, t))
    return labels, indices
