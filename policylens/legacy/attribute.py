"""Per-frame input-pixel attribution for Diffusion Policy on PushT.

Three methods, all computed on the same scalar target so they're comparable:

  IG       — Integrated Gradients (captum). Path integral of gradients from a
             black-image baseline to the actual image. Generally the most
             trustworthy gradient-based attribution.
  Saliency — |d target / d input|. Cheap, noisy, often a useful sanity check
             against IG.
  Random   — Random-noise heatmap. Anything passing this baseline is doing
             *something*; anything failing it is theatre.

Attribution target — Diffusion Policy denoises an action chunk over many
steps. For a single-step, differentiable scalar we use the policy's underlying
conditional UNet to predict noise from (random_noisy_action, fixed_timestep,
conditioning(image, state)), then take the L2 norm of that noise prediction.
This captures "which pixels matter for the policy's noise estimate at this
denoising step" — a faithful answer to one well-defined model question rather
than a hand-wavy answer to the whole rollout.

If lerobot's diffusion policy internals change, the only thing to update is
`_predict_noise_norm` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from captum.attr import IntegratedGradients, Saliency
from tqdm import tqdm

from policylens.load import EpisodeFrames


# Fixed seeds so attribution is reproducible across frames and runs.
_NOISE_SEED = 0
_TIMESTEP_FRACTION = 0.5  # mid-denoising step — late enough to be informative


@dataclass
class FrameAttributions:
    """Per-frame heatmaps, all shape (H, W) float32 in [0,1] after normalization."""

    ig: np.ndarray
    saliency: np.ndarray
    random: np.ndarray


def compute_attributions(
    policy,
    episode: EpisodeFrames,
    frame_indices: list[int],
    device: str = "cuda",
    ig_steps: int = 16,
) -> dict[int, FrameAttributions]:
    """Compute IG + Saliency + Random heatmaps for the requested frames only.

    Attribution is expensive (~O(ig_steps) forward+backward passes per frame for
    IG); we don't run it on every timestep, just the keyframes we'll render.
    """
    forward = _make_scalar_forward(policy, device=device)

    ig_attributor = IntegratedGradients(forward)
    saliency_attributor = Saliency(forward)

    rng = np.random.default_rng(_NOISE_SEED)
    out: dict[int, FrameAttributions] = {}

    for idx in tqdm(frame_indices, desc="attribute", unit="frame"):
        image = episode.images[idx].to(device).unsqueeze(0).requires_grad_(True)
        state = episode.states[idx].to(device).unsqueeze(0)

        baseline = torch.zeros_like(image)
        ig_attr = ig_attributor.attribute(
            image,
            baselines=baseline,
            additional_forward_args=(state,),
            n_steps=ig_steps,
        )
        sal_attr = saliency_attributor.attribute(
            image,
            additional_forward_args=(state,),
            abs=True,
        )

        ig_map = _to_heatmap(ig_attr)
        sal_map = _to_heatmap(sal_attr)
        rand_map = rng.random(ig_map.shape, dtype=np.float32)

        out[idx] = FrameAttributions(ig=ig_map, saliency=sal_map, random=rand_map)

    return out


def _make_scalar_forward(policy, device: str) -> Callable:
    """Returns f(image, state) -> scalar — the function we attribute.

    The scalar is the L2 norm of the noise prediction at a fixed mid-denoising
    timestep with a fixed noise sample, so the only thing varying across calls
    is the conditioning (image, state). That's the right signal for "what about
    this input drove this denoising step's correction."
    """
    diffusion = policy.diffusion  # the inner DiffusionModel (lerobot name)

    # Pull static config so we can build the right noisy_action shape.
    horizon = diffusion.config.horizon
    action_dim = diffusion.config.action_feature.shape[0]
    num_train_timesteps = diffusion.noise_scheduler.config.num_train_timesteps
    timestep_idx = int(_TIMESTEP_FRACTION * num_train_timesteps)

    # Fixed noise so the only differentiable input is (image, state).
    gen = torch.Generator(device=device).manual_seed(_NOISE_SEED)
    fixed_noisy_action = torch.randn(
        (1, horizon, action_dim), generator=gen, device=device
    )
    fixed_timesteps = torch.tensor([timestep_idx], device=device, dtype=torch.long)

    # lerobot's DiffusionPolicy keeps an internal obs queue of length n_obs_steps
    # and expects already-stacked inputs at the internal API layer:
    #   observation.images : (B, n_obs_steps, n_cameras, C, H, W)
    #   observation.state  : (B, n_obs_steps, state_dim)
    # For attribution we have one image per timestep, so we tile that image
    # across the obs-steps dim. Gradients flow through expand() correctly,
    # which means d(noise_pred)/d(image) accumulates over the tile dim — the
    # right thing to do (the model uses this image at both obs positions).
    n_obs_steps = diffusion.config.n_obs_steps
    n_cameras = 1  # single-camera task; pusht has one `observation.image`

    def forward(image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        stacked_images = image.unsqueeze(1).unsqueeze(2).expand(
            b, n_obs_steps, n_cameras, *image.shape[1:]
        )
        stacked_state = state.unsqueeze(1).expand(b, n_obs_steps, *state.shape[1:])
        batch = {
            "observation.images": stacked_images,
            "observation.state": stacked_state,
        }
        global_cond = diffusion._prepare_global_conditioning(batch)

        noise_pred = diffusion.unet(
            fixed_noisy_action.expand(b, -1, -1),
            fixed_timesteps.expand(b),
            global_cond=global_cond,
        )
        return noise_pred.flatten(1).pow(2).sum(dim=1).sqrt()

    return forward


def _to_heatmap(attr: torch.Tensor) -> np.ndarray:
    """Collapse (1, C, H, W) attribution → (H, W) normalized to [0,1]."""
    a = attr.detach().abs().sum(dim=1).squeeze(0).cpu().float().numpy()
    a_min, a_max = float(a.min()), float(a.max())
    if a_max - a_min < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - a_min) / (a_max - a_min)).astype(np.float32)
