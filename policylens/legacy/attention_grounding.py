"""Attention-based binding diagnostic.

We extract attention weights from OpenVLA's Llama backbone at the position
of a *noun token* in the prompt (e.g., "spoon") and look at where that
noun-token's attention concentrates over the image patches.

The diagnostic question: **does the model look at different image patches
when we change the noun?** (e.g., "spoon" → "fork")

If yes — the model is language-sensitive: it routes visual attention based
on what was named. If no — the model treats "spoon" and "fork" identically;
attention is dictated by the visual scene alone. That's wrong-binding
evidence even if action output ends up similar (since action depends on
later FFN layers too).

References:
  • Kang et al. CVPR 2025 — "Your LVLM Only Needs a Few Attention Heads for
    Visual Grounding" (arXiv 2503.06287). Shows that 3 of thousands of heads
    in LLaVA carry near-all grounding signal; we identify analogous heads
    in OpenVLA implicitly by ranking them on language-sensitivity here.
  • "Seeing but Not Believing" (arXiv 2510.17771). Attention is necessary
    but not sufficient — agreeing attention + diverging action = downstream
    failure; that's its own important diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import math
import numpy as np
import torch

from policylens.openvla import OpenVLAInference, VLAPrediction


# Image tokens are inserted right after BOS (position 0); see modeling_prismatic.py.
DEFAULT_IMAGE_TOKEN_START = 1


@dataclass
class AttentionGroundingResult:
    """Per-(layer, head) attention heatmaps over the image, for one noun token.

    attention_maps shape: (n_layers, n_heads, grid, grid). Normalized so each
    head's map sums to 1 (after sliced to image tokens only).
    """

    noun: str
    noun_token_positions: list[int]   # all sub-token positions for that noun (handles BPE splits)
    attention_maps: np.ndarray         # (L, H, G, G)
    grid_side: int


def find_noun_token_positions(
    vla: OpenVLAInference, pred: VLAPrediction, noun: str
) -> list[int]:
    """Locate ALL sub-token positions of `noun` in the prompt.

    Llama BPE often splits short words ("spoon" → ["▁sp", "oon"]; "fork"
    → ["▁fork"]; "tray" → ["▁t", "ray"]). We return every position whose
    decoded form is part of `noun`. The caller can attend over them as a set.

    If the noun isn't found, returns an empty list and the caller should
    skip this noun rather than fall back to a bogus position.
    """
    tokenizer = vla.processor.tokenizer
    prompt_ids = pred.full_input_ids[0, : pred.prompt_len].tolist()
    tokens = tokenizer.convert_ids_to_tokens(prompt_ids)
    needle = noun.lower().strip()

    positions: list[int] = []
    # Walk word boundaries: in Llama BPE, '▁' prefix marks a new word.
    i = 0
    while i < len(tokens):
        t = tokens[i] or ""
        # Word starts here if token begins with '▁'.
        if t.startswith("▁"):
            # Accumulate sub-tokens until next word boundary.
            j = i
            piece = t.replace("▁", "")
            while j + 1 < len(tokens) and not (tokens[j + 1] or "").startswith("▁"):
                j += 1
                piece += tokens[j] or ""
            if piece.lower().strip(".,!?:;") == needle:
                positions.extend(range(i, j + 1))
            i = j + 1
        else:
            i += 1
    return positions


def extract_attention_to_image(
    vla: OpenVLAInference,
    pred: VLAPrediction,
    noun_token_positions: list[int],
    image_token_start: int = DEFAULT_IMAGE_TOKEN_START,
) -> AttentionGroundingResult | None:
    """Run forward with output_attentions; return per-head image-attention.

    Returned shape (L, H, grid, grid) is normalized — each head's image-token
    attention vector sums to 1, so heads with "spiky" attention dominate
    visualization vs heads that spread uniformly.

    Returns None if `noun_token_positions` is empty.
    """
    if not noun_token_positions:
        return None

    with torch.inference_mode():
        outputs = vla.model(
            input_ids=pred.full_input_ids,
            pixel_values=pred.pixel_values,
            output_attentions=True,
        )
    # attentions: tuple of (B, heads, seq, seq), one per layer.
    full_seq = outputs.attentions[0].shape[-1]
    text_seq = pred.full_input_ids.shape[1]
    n_image = full_seq - text_seq

    # In the multimodal sequence, image tokens occupy [1, 1+n_image).
    # Original prompt position `p` (for p>=1) becomes p+n_image after image
    # insertion. The noun lives in the text region (p >= 1 most likely).
    noun_pos_mm = [(p + n_image) if p >= 1 else p for p in noun_token_positions]
    img_slice = slice(image_token_start, image_token_start + n_image)
    grid_side = int(round(math.sqrt(n_image)))
    grid_total = grid_side * grid_side

    L = len(outputs.attentions)
    H = outputs.attentions[0].shape[1]
    maps = np.zeros((L, H, grid_total), dtype=np.float32)
    for li, attn in enumerate(outputs.attentions):
        # (B=1, H, seq, seq) → take rows = noun_pos_mm, slice cols = img_slice
        rows = attn[0, :, :, :][:, noun_pos_mm, img_slice]  # (H, len(noun_pos), n_image)
        # Sum sub-tokens together; this treats multi-token nouns as one entity.
        per_head = rows.sum(dim=1).float().cpu().numpy()    # (H, n_image)
        # Normalize each head so head-magnitude doesn't dominate.
        sums = per_head.sum(axis=1, keepdims=True)
        per_head = np.where(sums > 0, per_head / np.maximum(sums, 1e-9), per_head)
        # Pad or truncate to a square grid.
        if per_head.shape[1] < grid_total:
            pad = np.zeros((H, grid_total - per_head.shape[1]), dtype=np.float32)
            per_head = np.concatenate([per_head, pad], axis=1)
        elif per_head.shape[1] > grid_total:
            per_head = per_head[:, :grid_total]
        maps[li] = per_head

    return AttentionGroundingResult(
        noun="",  # caller fills
        noun_token_positions=noun_token_positions,
        attention_maps=maps.reshape(L, H, grid_side, grid_side),
        grid_side=grid_side,
    )


@dataclass
class HeadLanguageSensitivity:
    """Per-head measure of how different attention is between two nouns."""

    layer: int
    head: int
    js_divergence: float          # Jensen-Shannon between the two attention dists
    primary_focus_a: tuple[int, int]   # (row, col) in grid for noun A
    primary_focus_b: tuple[int, int]   # (row, col) in grid for noun B
    concentration_a: float        # 1 - normalized entropy for A
    concentration_b: float        # 1 - normalized entropy for B


def score_head_language_sensitivity(
    result_a: AttentionGroundingResult,
    result_b: AttentionGroundingResult,
) -> list[HeadLanguageSensitivity]:
    """For each (layer, head), measure how much its attention pattern changes
    when the noun is swapped from A to B. Heads with high JS divergence are
    'language-sensitive' — they route attention based on the noun. Heads with
    low JS treat both nouns identically (visual default).
    """
    L, H, G, _ = result_a.attention_maps.shape
    assert result_b.attention_maps.shape == result_a.attention_maps.shape

    out: list[HeadLanguageSensitivity] = []
    for li in range(L):
        for hi in range(H):
            a = result_a.attention_maps[li, hi].flatten()
            b = result_b.attention_maps[li, hi].flatten()
            js = _jensen_shannon(a, b)
            focus_a = np.unravel_index(np.argmax(result_a.attention_maps[li, hi]), (G, G))
            focus_b = np.unravel_index(np.argmax(result_b.attention_maps[li, hi]), (G, G))
            out.append(HeadLanguageSensitivity(
                layer=li, head=hi,
                js_divergence=float(js),
                primary_focus_a=tuple(int(x) for x in focus_a),
                primary_focus_b=tuple(int(x) for x in focus_b),
                concentration_a=_concentration(a),
                concentration_b=_concentration(b),
            ))
    return out


def aggregate_attention_across_heads(
    result: AttentionGroundingResult,
    layer_fraction: tuple[float, float] = (0.3, 1.0),
) -> np.ndarray:
    """Mean attention across heads + a slice of layers → one (G, G) heatmap.

    Defaults to the middle-and-late layers (where binding generally happens
    per the MINT fusion-band finding). Returns float32 in [0, 1].
    """
    L, H, G, _ = result.attention_maps.shape
    lo = int(L * layer_fraction[0])
    hi = int(L * layer_fraction[1])
    aggregated = result.attention_maps[lo:hi].mean(axis=(0, 1))  # (G, G)
    lo_v, hi_v = float(aggregated.min()), float(aggregated.max())
    if hi_v - lo_v < 1e-9:
        return np.zeros_like(aggregated, dtype=np.float32)
    return ((aggregated - lo_v) / (hi_v - lo_v)).astype(np.float32)


# ---- small utilities ------------------------------------------------------


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """JS divergence (base-2, in [0,1])."""
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log2(p) - np.log2(m))))
    kl_qm = float(np.sum(q * (np.log2(q) - np.log2(m))))
    return 0.5 * (kl_pm + kl_qm)


def _concentration(dist: np.ndarray) -> float:
    """1 - normalized entropy. High = concentrated; low = uniform."""
    p = np.clip(dist, 1e-12, 1.0)
    p = p / p.sum()
    H_max = math.log2(p.size)
    return float(1.0 - (-np.sum(p * np.log2(p))) / max(H_max, 1e-9))
