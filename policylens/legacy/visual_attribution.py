"""Cross-modal attribution for OpenVLA — per-neuron *image* maps.

Stage B v2 told us *which concepts* fire and *when*. This module answers
the missing question: **where in the image is each concept anchored?**

Two methods, used together:

  • `compute_attention_rollout`     — Aggregates attention from the action
    position back to the image-token positions across all layers/heads.
    Cheap (~1 forward, output_attentions=True). Shows the *macro* picture:
    "this is what the model is looking at right now."

  • `compute_per_neuron_image_attribution` — For each smoking-gun neuron,
    patch-ablate the image (16×16 grid by default) and measure the drop in
    that neuron's activation at the action position. The drop heatmap is
    causal: *these patches drive this neuron*.

The product UI then pairs each anomalous neuron's language label (from the
logit-lens dictionary) with its image map → "the model's `transition` concept
is anchored to these pixels, and it fired 4× above baseline at the failure."
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from policylens.openvla import OpenVLAInference, VLAPrediction


# OpenVLA inserts image tokens after BOS (position 0). The number of image
# tokens equals the vision backbone's patch count — for SigLIP-ViT-L+DinoV2
# at 224×224 with patch_size=14 it's 16*16 = 256. We don't hardcode this;
# we infer it at runtime from the actual forward.
DEFAULT_VISION_TOKEN_START = 1


# ---------------------------------------------------------------------------
# Path A — Attention rollout
# ---------------------------------------------------------------------------


def compute_attention_rollout(
    vla: OpenVLAInference,
    pred: VLAPrediction,
    image_token_start: int = DEFAULT_VISION_TOKEN_START,
    aggregate: str = "mean",
) -> np.ndarray:
    """One macro heatmap: 'where the model is looking when picking this action.'

    Method: forward pass with output_attentions=True; for each layer take the
    attention from the action-prediction position (last prompt token) to the
    image-token positions; average across heads and layers; reshape to a
    square grid. Standard ViT-style rollout.

    Returns (grid, grid) float32 in [0,1].
    """
    with torch.inference_mode():
        outputs = vla.model(
            input_ids=pred.full_input_ids,
            pixel_values=pred.pixel_values,
            output_attentions=True,
        )
    # outputs.attentions: tuple of per-layer attention tensors,
    # each shape (B, heads, seq, seq).
    full_seq_len = outputs.attentions[0].shape[-1]
    text_seq_len = pred.full_input_ids.shape[1]
    # Number of inserted image tokens = total seq len − text seq len.
    n_image_tokens = full_seq_len - text_seq_len
    # The action-prediction position is the LAST POSITION of the prompt
    # within the *multimodal* sequence (after image insertion).
    # Original prompt has `prompt_len` text tokens including the trailing
    # space. After image insertion, position p in the text becomes p+n_image
    # for p>=1 (BOS stays at 0). The action-prediction read position is at
    # prompt_len - 1 in the text → (prompt_len - 1) + n_image_tokens in the
    # multimodal sequence.
    action_pos_mm = pred.prompt_len - 1 + n_image_tokens
    img_slice = slice(image_token_start, image_token_start + n_image_tokens)

    contribs = []
    for layer_attn in outputs.attentions:
        # (B, heads, seq, seq) → mean over heads, take row=action_pos_mm
        row = layer_attn[0, :, action_pos_mm, img_slice].float().mean(dim=0)  # (n_image,)
        contribs.append(row.cpu().numpy())
    contribs = np.stack(contribs)  # (n_layers, n_image)
    agg = contribs.mean(axis=0) if aggregate == "mean" else contribs.max(axis=0)

    grid_side = int(round(math.sqrt(n_image_tokens)))
    if grid_side * grid_side != n_image_tokens:
        # Padding / non-square — pad with zeros to nearest square.
        side = grid_side + 1
        padded = np.zeros(side * side, dtype=np.float32)
        padded[:n_image_tokens] = agg
        agg = padded
        grid_side = side
    grid = agg.reshape(grid_side, grid_side)
    # Normalize.
    lo, hi = float(grid.min()), float(grid.max())
    return ((grid - lo) / (hi - lo + 1e-9)).astype(np.float32)


# ---------------------------------------------------------------------------
# Path B — Per-neuron causal image attribution
# ---------------------------------------------------------------------------


@dataclass
class NeuronImageMap:
    layer: int
    neuron: int
    heatmap: np.ndarray         # (grid, grid) — activation drop per patch
    baseline_activation: float  # raw |activation| × ||value|| at unperturbed input
    grid_side: int


def compute_per_neuron_image_attribution(
    vla: OpenVLAInference,
    pred: VLAPrediction,
    neurons: list[tuple[int, int]],
    dictionary: dict,
    grid_side: int = 16,
) -> list[NeuronImageMap]:
    """For each requested neuron (layer, idx), produce a causal image map.

    Method: split the image into a `grid_side × grid_side` patch grid; for
    each patch, mask it out (replace with channel-mean) and re-run the
    forward; capture every requested neuron's activation at the action
    position via hooks (one forward → many neurons, all in same pass).
    The drop in activation = "this patch drives this neuron."

    Cost: grid_side² forward passes. For 16×16 = 256 passes ≈ 75 s/frame.
    """
    if not neurons:
        return []
    model = vla.model
    layers = model.language_model.model.layers

    # Pre-compute value-vector norms so we can produce contributions matched
    # to the same units (|act| × ||value||) as the smoking-gun plot.
    value_norms = {
        (li, ni): float(dictionary["value_norms"][str(li)][ni])
        for (li, ni) in neurons
    }

    # Build per-layer set of neurons to capture.
    by_layer: dict[int, list[int]] = {}
    for li, ni in neurons:
        by_layer.setdefault(li, []).append(ni)

    read_position = pred.prompt_len - 1  # in TEXT sequence; will adjust later

    # We need the *multimodal* read position because hooks fire on the LLM's
    # internal sequence (after image-token insertion). We probe it with a
    # dry-run forward to learn n_image_tokens.
    with torch.inference_mode():
        probe = model(
            input_ids=pred.full_input_ids,
            pixel_values=pred.pixel_values,
            output_attentions=False,
        )
    full_seq = probe.logits.shape[1]
    n_image = full_seq - pred.full_input_ids.shape[1]
    read_position_mm = pred.prompt_len - 1 + n_image

    # Run with hooks once on the unperturbed input to get baseline activations.
    captured_baseline = _run_with_hooks(model, layers, by_layer, pred.full_input_ids,
                                        pred.pixel_values, read_position_mm)
    baselines = {
        (li, ni): abs(captured_baseline[li][ni]) * value_norms[(li, ni)]
        for (li, ni) in neurons
    }

    pix = pred.pixel_values
    ph, pw = pix.shape[-2:]
    chan_means = pix.mean(dim=(2, 3), keepdim=True)
    patch_h = ph // grid_side
    patch_w = pw // grid_side

    drops: dict[tuple[int, int], np.ndarray] = {
        k: np.zeros((grid_side, grid_side), dtype=np.float32) for k in neurons
    }

    for gi in tqdm(range(grid_side), desc="image attr (rows)", leave=False):
        for gj in range(grid_side):
            perturbed = pix.clone()
            r0, r1 = gi * patch_h, (gi + 1) * patch_h
            c0, c1 = gj * patch_w, (gj + 1) * patch_w
            perturbed[..., r0:r1, c0:c1] = chan_means
            captured = _run_with_hooks(model, layers, by_layer,
                                       pred.full_input_ids, perturbed, read_position_mm)
            for (li, ni) in neurons:
                cur = abs(captured[li][ni]) * value_norms[(li, ni)]
                drops[(li, ni)][gi, gj] = baselines[(li, ni)] - cur

    out: list[NeuronImageMap] = []
    for (li, ni) in neurons:
        h = drops[(li, ni)]
        # Clamp negative drops (perturbation raised activation) to 0 for
        # visualization — we want "what makes the neuron fire", not "what
        # silences it." Negative drops can be plotted separately if needed.
        h = np.clip(h, 0, None)
        lo, hi = float(h.min()), float(h.max())
        norm = ((h - lo) / (hi - lo + 1e-9)).astype(np.float32)
        out.append(NeuronImageMap(
            layer=li, neuron=ni,
            heatmap=norm,
            baseline_activation=float(baselines[(li, ni)]),
            grid_side=grid_side,
        ))
    return out


def _run_with_hooks(
    model,
    layers,
    by_layer: dict[int, list[int]],
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    read_position_mm: int,
) -> dict[int, dict[int, float]]:
    """Forward with FFN hooks; return {layer: {neuron: scalar_activation}}."""
    captured_layer: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(li: int) -> Callable:
        def hook(module, inputs, output):
            captured_layer[li] = inputs[0].detach()
            return None
        return hook

    for li in by_layer:
        h = layers[li].mlp.down_proj.register_forward_hook(make_hook(li))
        handles.append(h)

    try:
        with torch.inference_mode():
            model(input_ids=input_ids, pixel_values=pixel_values)
    finally:
        for h in handles:
            h.remove()

    out: dict[int, dict[int, float]] = {}
    for li, neuron_list in by_layer.items():
        act = captured_layer[li][0, read_position_mm].float().cpu().numpy()
        out[li] = {ni: float(act[ni]) for ni in neuron_list}
    return out
