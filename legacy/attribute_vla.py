"""Two attribution paths for OpenVLA, sharing one differentiable target.

Target = sum of log-probabilities of the 7 *chosen* action tokens, i.e.
"how confidently does the model produce *this* action from these inputs."

  • Image  : Saliency + Integrated Gradients over `pixel_values`.
  • Tokens : LayerIntegratedGradients on the text-embedding layer
             (input ids are discrete, so we integrate in embedding space).

Both are memory-controlled via `internal_batch_size=1`: captum runs IG steps
sequentially instead of in a single fat batch, keeping a 7B-param model
backward pass inside 24GB VRAM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients, Saliency

from emboviz.openvla import OpenVLAInference, VLAPrediction


@dataclass
class ImageAttribution:
    saliency: np.ndarray   # (H, W) [0,1]
    ig: np.ndarray         # (H, W) [0,1]


@dataclass
class TokenAttribution:
    tokens: list[str]      # human-readable strings, length = num_input_tokens
    scores: np.ndarray     # per-token importance (>=0). For ablation method:
                           # ||Δaction|| when that token is replaced by BOS.
    norm_scores: np.ndarray  # same length, normalized to [-1, 1] for display
    method: str            # "ablation" or "ig" — labels the visualization


def attribute_image(
    vla: OpenVLAInference,
    pred: VLAPrediction,
    ig_steps: int = 8,
) -> ImageAttribution:
    """Saliency (1 backward pass) + IG (`ig_steps` backward passes)."""

    def forward(pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values is the only thing captum differentiates; everything
        # else is held fixed (text tokens, decoded action tokens).
        return vla.scalar_attribution_target(
            pixel_values=pixel_values,
            full_input_ids=pred.full_input_ids,
            action_token_ids=pred.action_token_ids,
            prompt_len=pred.prompt_len,
        )

    pix = pred.pixel_values.clone().detach().requires_grad_(True)

    sal = Saliency(forward)
    sal_attr = sal.attribute(pix, abs=True)

    ig = IntegratedGradients(forward)
    ig_attr = ig.attribute(
        pix,
        baselines=torch.zeros_like(pix),
        n_steps=ig_steps,
        internal_batch_size=1,
    )

    return ImageAttribution(
        saliency=_collapse_pixel_map(sal_attr),
        ig=_collapse_pixel_map(ig_attr),
    )


def attribute_tokens(
    vla: OpenVLAInference,
    pred: VLAPrediction,
    ig_steps: int = 8,  # kept for API compat; unused by ablation path
) -> TokenAttribution:
    """Per-token attribution via direct causal ablation.

    Why not LayerIntegratedGradients? Because OpenVLA-7B in bf16 has so much
    multiplicative depth (Llama2 backbone + dual ViTs + projector) that
    gradients at the embedding layer routinely underflow to zero — we saw
    this empirically and dropped it. Ablation sidesteps gradients entirely:
    we *measure* how much the model's action changes when each input token is
    silenced (replaced with the BOS embedding). That's a causal score by
    construction, which is what we'd want to verify a gradient with anyway.

    Side benefit: the resulting bars are directly interpretable as
    "||Δaction|| if you delete this word."
    """
    prefix_ids = pred.full_input_ids[:, : pred.prompt_len].clone()
    n_tokens = int(prefix_ids.shape[1])
    bos_id = vla.processor.tokenizer.bos_token_id or 1
    base_action = pred.action
    action_dim = int(pred.action_token_ids.shape[-1])

    scores = np.zeros(n_tokens, dtype=np.float32)
    with torch.inference_mode():
        for i in range(n_tokens):
            perturbed = prefix_ids.clone()
            perturbed[0, i] = bos_id
            generated = vla.model.generate(
                input_ids=perturbed,
                pixel_values=pred.pixel_values,
                max_new_tokens=action_dim,
                do_sample=False,
            )
            new_tokens = generated[0, -action_dim:].cpu().numpy()
            new_action = vla._decode_action(new_tokens, unnorm_key="bridge_orig")
            scores[i] = float(np.linalg.norm(new_action - base_action))

    tokens = vla.processor.tokenizer.convert_ids_to_tokens(prefix_ids[0].tolist())
    abs_max = float(scores.max()) or 1.0
    norm = scores / abs_max
    return TokenAttribution(tokens=tokens, scores=scores, norm_scores=norm, method="ablation")


# ---- helpers ---------------------------------------------------------------


def _collapse_pixel_map(attr: torch.Tensor) -> np.ndarray:
    """OpenVLA's pixel_values is (B, C, H, W) where C may be 6 (SigLIP+DinoV2
    fused). Collapse to (H, W) by summing absolute value across channels and
    normalizing to [0, 1]. The fused-encoder channel-stacking is an OpenVLA
    quirk — see modeling_prismatic.py."""
    a = attr.detach().abs().sum(dim=1).squeeze(0).cpu().float().numpy()
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)
