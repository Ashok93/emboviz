"""Causal sanity checks — *the* moat differentiator for PolicyLens.

A heatmap that *looks* meaningful might still be misleading: the gradient can
fixate on features the model doesn't actually rely on (a common failure of
saliency methods). We rule that out with two perturbation tests.

  • Image — Insertion/Deletion-style occlusion curve.
      Sort pixels by IG magnitude. For k = 0…100% in steps, replace the top-k%
      pixels with the per-channel mean color and re-run the policy. Measure
      ||action_perturbed − action_original||₂. Repeat with a *random* pixel
      ordering. If the IG-ordered curve grows much faster than random, our
      heatmap is causally faithful.

  • Tokens — Per-token ablation.
      For each input token in the instruction span, replace its embedding
      with the BOS embedding (a "soft delete") and re-run the policy. Measure
      action delta. Rank-correlate per-token deltas with the token attribution
      scores. High correlation ⇒ the attribution rank predicts which words
      actually drive behavior.

A demo without this section is just another heatmap. *This* is what makes
PolicyLens defensible against arXiv-style screenshots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr

from policylens.attribute_vla import ImageAttribution, TokenAttribution
from policylens.openvla import OpenVLAInference, VLAPrediction


@dataclass
class ImageFaithfulness:
    coverage_pct: np.ndarray        # (K,) — e.g. [0, 10, 20, ..., 100]
    delta_ig: np.ndarray            # (K,) — action delta when masking top-IG pixels
    delta_random: np.ndarray        # (K,) — action delta when masking random pixels (mean over seeds)
    auc_ratio: float                # area(delta_ig) / area(delta_random) — >>1 is good


@dataclass
class TokenFaithfulness:
    tokens: list[str]
    attribution_scores: np.ndarray  # per-token IG score
    measured_deltas: np.ndarray     # per-token action delta when ablated
    spearman_rho: float             # rank correlation (attribution vs measured)
    spearman_p: float


# ---- image -----------------------------------------------------------------


def image_occlusion_curve(
    vla: OpenVLAInference,
    pred: VLAPrediction,
    attribution: ImageAttribution,
    coverages: list[int] | None = None,
    random_seeds: int = 3,
    patch_size: int = 32,
) -> ImageFaithfulness:
    """Patch-level occlusion curve.

    Why patch-level? OpenVLA discretizes actions into 256 bins per DOF, so
    masking individual pixels rarely shifts the chosen bin. Patches at the
    ViT-input-patch scale (14–16px) are the smallest unit that meaningfully
    moves the model's logits. This is also what RISE / Insertion-Deletion
    canonically do.

    We aggregate IG per patch by mean (or sum — equivalent for ranking).
    """
    if coverages is None:
        coverages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    pix = pred.pixel_values  # (1, C, H, W)
    base_action = pred.action
    ph, pw = pix.shape[-2:]

    # Build a patch grid aligned to pixel_values resolution.
    h_patches = ph // patch_size
    w_patches = pw // patch_size
    n_patches = h_patches * w_patches

    # Resize IG heatmap to (h_patches, w_patches) by mean-pooling so each
    # patch gets a single importance score.
    ig_at_patch_res = _mean_pool(attribution.ig, h_patches, w_patches)
    ig_order = np.argsort(-ig_at_patch_res.flatten())  # descending importance

    delta_ig = np.zeros(len(coverages))
    delta_random = np.zeros(len(coverages))

    rng = np.random.default_rng(0)
    random_orders = [rng.permutation(n_patches) for _ in range(random_seeds)]

    chan_means = pix.mean(dim=(2, 3), keepdim=True)

    for i, cov in enumerate(coverages):
        k = int(round(cov / 100.0 * n_patches))
        delta_ig[i] = _mask_patches_and_measure(
            vla, pred, ig_order[:k], h_patches, w_patches, patch_size,
            chan_means, base_action,
        )
        rand_deltas = [
            _mask_patches_and_measure(
                vla, pred, order[:k], h_patches, w_patches, patch_size,
                chan_means, base_action,
            )
            for order in random_orders
        ]
        delta_random[i] = float(np.mean(rand_deltas))

    auc_ig = float(np.trapz(delta_ig, coverages))
    auc_rand = float(np.trapz(delta_random, coverages))
    ratio = auc_ig / max(auc_rand, 1e-6)

    return ImageFaithfulness(
        coverage_pct=np.array(coverages, dtype=np.float32),
        delta_ig=delta_ig,
        delta_random=delta_random,
        auc_ratio=ratio,
    )


def _mask_patches_and_measure(
    vla: OpenVLAInference,
    pred: VLAPrediction,
    patch_indices: np.ndarray,
    h_patches: int,
    w_patches: int,
    patch_size: int,
    chan_means: torch.Tensor,
    base_action: np.ndarray,
) -> float:
    """Mask a set of patches and re-run the policy; return ||Δaction||₂."""
    perturbed = pred.pixel_values.clone()
    if len(patch_indices) > 0:
        for p in patch_indices:
            r = (p // w_patches) * patch_size
            c = (p % w_patches) * patch_size
            perturbed[..., r : r + patch_size, c : c + patch_size] = chan_means

    with torch.inference_mode():
        action_dim = pred.action_token_ids.shape[-1]
        prompt_ids = pred.full_input_ids[:, : pred.prompt_len]
        generated = vla.model.generate(
            input_ids=prompt_ids,
            pixel_values=perturbed,
            max_new_tokens=action_dim,
            do_sample=False,
        )
        new_action_tokens = generated[0, -action_dim:].cpu().numpy()
    new_action = vla._decode_action(new_action_tokens, unnorm_key="bridge_orig")
    return float(np.linalg.norm(new_action - base_action))


def _mean_pool(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Downsample 2D array to (target_h, target_w) by mean-pooling fixed-size
    blocks. We do this rather than bilinear resize so each patch's score is
    the average IG inside it — the right summary for occlusion.
    """
    h, w = arr.shape
    # Crop to a size divisible by the target grid, then mean-pool.
    bh = h // target_h
    bw = w // target_w
    arr = arr[: bh * target_h, : bw * target_w]
    return arr.reshape(target_h, bh, target_w, bw).mean(axis=(1, 3))


# ---- tokens ----------------------------------------------------------------


def token_ablation(
    vla: OpenVLAInference,
    pred: VLAPrediction,
    token_attr: TokenAttribution,
    instruction_token_span: tuple[int, int],
) -> TokenFaithfulness:
    """Ablate each instruction-span token, measure action delta, rank-correlate.

    `instruction_token_span` is (start, end) into `pred.full_input_ids[0]`.
    We only ablate within that span to avoid touching the chat-template
    framing ("In: What action ...") which would always change the answer.
    """
    start, end = instruction_token_span
    base_action = pred.action

    bos_id = vla.processor.tokenizer.bos_token_id or 1
    deltas = np.zeros(end - start, dtype=np.float32)

    for offset, t_idx in enumerate(range(start, end)):
        perturbed = pred.full_input_ids[:, : pred.prompt_len].clone()
        perturbed[0, t_idx] = bos_id
        with torch.inference_mode():
            action_dim = pred.action_token_ids.shape[-1]
            generated = vla.model.generate(
                input_ids=perturbed,
                pixel_values=pred.pixel_values,
                max_new_tokens=action_dim,
                do_sample=False,
            )
            new_action_tokens = generated[0, -action_dim:].cpu().numpy()
        new_action = vla._decode_action(new_action_tokens, unnorm_key="bridge_orig")
        deltas[offset] = float(np.linalg.norm(new_action - base_action))

    # Compare to attribution scores in the same span.
    span_scores = token_attr.scores[start:end]
    if len(span_scores) >= 2 and np.std(span_scores) > 0 and np.std(deltas) > 0:
        rho, pval = spearmanr(np.abs(span_scores), deltas)
    else:
        rho, pval = float("nan"), float("nan")

    return TokenFaithfulness(
        tokens=token_attr.tokens[start:end],
        attribution_scores=span_scores,
        measured_deltas=deltas,
        spearman_rho=float(rho),
        spearman_p=float(pval),
    )


def locate_instruction_span(vla: OpenVLAInference, pred: VLAPrediction) -> tuple[int, int]:
    """Find the (start, end) token indices of the instruction inside the prompt.

    We re-tokenize the bare instruction and locate its first occurrence in the
    prompt token sequence. Robust to tokenizer prefix-space quirks because we
    look for the suffix of the instruction tokens.
    """
    tok = vla.processor.tokenizer
    instr_ids = tok(pred.instruction_text, add_special_tokens=False)["input_ids"]
    prompt_ids = pred.full_input_ids[0, : pred.prompt_len].tolist()
    n = len(instr_ids)
    # Strip leading BOS-like tokens that the bare-tokenize would add but the
    # in-context tokenization wouldn't.
    for start_offset in (0, 1):
        needle = instr_ids[start_offset:]
        for i in range(len(prompt_ids) - len(needle) + 1):
            if prompt_ids[i : i + len(needle)] == needle:
                return (i, i + len(needle))
    # Fallback: middle of prompt
    return (max(0, pred.prompt_len // 2 - 5), min(pred.prompt_len, pred.prompt_len // 2 + 5))
