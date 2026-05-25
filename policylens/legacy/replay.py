"""Roll a policy over a recorded episode and locate the most interesting frame.

For each timestep we ask the policy to predict an action from the recorded
observation. We compare those predicted actions to the expert actions in the
dataset; the largest L2 deviation is our "failure-frame candidate" — the moment
the policy departs most from what the demonstrator did. That's the frame whose
attribution is most worth eyeballing.

Note: Diffusion Policy uses internal action chunking and an action queue. We
explicitly call `policy.reset()` once at the start of the episode and rely on
its standard `select_action` interface, which is exactly how it would be used
in a real rollout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm

from policylens.load import EpisodeFrames, build_observation_batch


@dataclass
class ReplayResult:
    predicted_actions: torch.Tensor  # (T, action_dim) on CPU
    action_deviations: torch.Tensor  # (T,) L2 distance to expert action
    failure_frame_idx: int           # argmax of action_deviations


def replay_episode(
    policy, episode: EpisodeFrames, device: str = "cuda", warmup_frames: int = 4
) -> ReplayResult:
    """Run the policy over every frame of the episode and record predictions.

    `warmup_frames` are excluded from failure-frame selection: DiffusionPolicy
    keeps an internal obs queue of length n_obs_steps and an action chunk of
    length horizon, so the first few steps are dominated by queue-warmup and
    chunk-boundary artefacts rather than meaningful policy disagreement.
    """
    if hasattr(policy, "reset"):
        policy.reset()

    predicted = []
    with torch.inference_mode():
        for t in tqdm(range(episode.num_frames), desc="replay", unit="frame"):
            obs = build_observation_batch(episode.images[t], episode.states[t], device)
            action = policy.select_action(obs)  # (1, action_dim)
            predicted.append(action.squeeze(0).detach().cpu().float())

    predicted_actions = torch.stack(predicted)  # (T, action_dim)
    deviations = (predicted_actions - episode.expert_actions).norm(dim=-1)

    # Mask out warmup region before picking failure frame.
    warmup = min(warmup_frames, episode.num_frames - 1)
    search_devs = deviations.clone()
    search_devs[:warmup] = -1.0
    failure_idx = int(torch.argmax(search_devs).item())

    return ReplayResult(
        predicted_actions=predicted_actions,
        action_deviations=deviations,
        failure_frame_idx=failure_idx,
    )


def pick_keyframes(result: ReplayResult, num_frames: int, total_frames: int) -> list[int]:
    """Pick `num_frames` frame indices to feature in the grid.

    Includes the failure frame plus an evenly-spaced sweep across the episode so
    the grid tells a temporal story rather than a single moment.
    """
    if num_frames >= total_frames:
        return list(range(total_frames))

    # Evenly-spaced sweep, then guarantee the failure frame is in there.
    spaced = torch.linspace(0, total_frames - 1, steps=num_frames - 1).round().long().tolist()
    keyframes = sorted(set(spaced + [result.failure_frame_idx]))
    # If dedup shrank us, pad from the densest gap. (Cheap to skip — fine for v1.)
    return keyframes
