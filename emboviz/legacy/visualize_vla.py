"""Stage B visualizations. These are the demo artifacts a non-author should
be able to open and immediately understand.

Three plots:
  • Frame grid PNG    : raw frame + IG overlay + token bar chart per keyframe.
  • Faithfulness PNG  : occlusion curves (IG vs random) — the moat plot.
  • Token chart PNG   : attribution vs measured ablation delta side-by-side
                        for the failure frame.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from emboviz.attribute_vla import ImageAttribution, TokenAttribution
from emboviz.faithfulness import ImageFaithfulness, TokenFaithfulness


# ---- frame grid ------------------------------------------------------------


def render_frame_grid(
    frames: list[Image.Image],
    keyframe_indices: list[int],
    image_attrs: dict[int, ImageAttribution],
    token_attrs: dict[int, TokenAttribution],
    failure_idx: int,
    instruction: str,
    out_path: Path,
) -> None:
    """N rows × 3 cols: frame | IG overlay | per-token bar chart."""
    n = len(keyframe_indices)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.5 * n), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 2]})
    fig.suptitle(f'OpenVLA  ·  instruction: "{instruction}"', fontsize=11, y=1.0)

    for row, idx in enumerate(keyframe_indices):
        frame = np.array(frames[idx])
        ig_overlay = _overlay(frame, image_attrs[idx].ig)

        is_failure = idx == failure_idx
        label = f"t={idx}" + ("  (failure)" if is_failure else "")

        axes[row, 0].imshow(frame)
        axes[row, 0].set_title("frame" if row == 0 else "", fontsize=10)
        axes[row, 0].set_ylabel(label, fontsize=9, color="red" if is_failure else "black")
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        axes[row, 1].imshow(ig_overlay)
        axes[row, 1].set_title("IG over image" if row == 0 else "", fontsize=10)
        axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])

        ta = token_attrs[idx]
        _draw_token_bars(axes[row, 2], ta, title="token attribution" if row == 0 else "")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---- faithfulness ---------------------------------------------------------


def render_image_faithfulness(
    faithfulness_per_frame: dict[int, ImageFaithfulness],
    failure_idx: int,
    out_path: Path,
) -> None:
    """Two lines: average occlusion-curve for IG-ordering vs random-ordering.

    Each line is averaged across keyframes (each keyframe is its own
    independent test). The mean AUC ratio is the headline metric.
    """
    keyframes = sorted(faithfulness_per_frame.keys())
    if not keyframes:
        return
    sample = next(iter(faithfulness_per_frame.values()))
    coverage = sample.coverage_pct

    delta_ig = np.stack([faithfulness_per_frame[k].delta_ig for k in keyframes])
    delta_rand = np.stack([faithfulness_per_frame[k].delta_random for k in keyframes])
    ratios = np.array([faithfulness_per_frame[k].auc_ratio for k in keyframes])

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(coverage, delta_ig.mean(0), "o-", color="#d62728", label="mask top-IG pixels", lw=2)
    ax.fill_between(coverage, delta_ig.min(0), delta_ig.max(0), color="#d62728", alpha=0.15)
    ax.plot(coverage, delta_rand.mean(0), "s-", color="#888888", label="mask random pixels", lw=2)
    ax.fill_between(coverage, delta_rand.min(0), delta_rand.max(0), color="#888888", alpha=0.15)

    ax.set_xlabel("% of pixels masked (ordered by IG score, or random)")
    ax.set_ylabel("||action change||₂ from original prediction")
    ax.set_title(
        f"Faithfulness: masking IG-ranked pixels changes the action  "
        f"{ratios.mean():.2f}×  more than random  (across {len(keyframes)} keyframes)"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_token_faithfulness(
    token_faith: TokenFaithfulness,
    out_path: Path,
) -> None:
    """Side-by-side: attribution magnitude vs measured ablation delta per token.

    Spearman ρ in the title quantifies how well the heatmap *predicts* the
    causal ranking. We want ρ > 0.5 with p < 0.1 ish for a clear story.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), sharey=False)
    x = np.arange(len(token_faith.tokens))
    labels = [_clean_token(t) for t in token_faith.tokens]

    axes[0].bar(x, np.abs(token_faith.attribution_scores), color="#1f77b4")
    axes[0].set_title("predicted importance (|IG attribution|)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, token_faith.measured_deltas, color="#d62728")
    axes[1].set_title(
        f"measured importance (||Δaction|| when ablated)  ·  Spearman ρ = {token_faith.spearman_rho:.2f}  (p={token_faith.spearman_p:.2g})"
    )
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---- helpers ---------------------------------------------------------------


def _overlay(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    if heatmap.shape != frame.shape[:2]:
        heatmap = _resize_2d(heatmap, frame.shape[:2])
    cmap = plt.get_cmap("jet")
    colored = (cmap(heatmap)[..., :3] * 255).astype(np.uint8)
    blended = frame.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _resize_2d(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _draw_token_bars(ax, ta: TokenAttribution, title: str) -> None:
    # Drop the BOS token (always position 0 with Llama) — it's the baseline
    # anchor and contributes no information for the reader.
    start = 1 if ta.tokens and ta.tokens[0] in ("<s>", "<bos>") else 0
    tokens = ta.tokens[start:]
    scores = ta.scores[start:]
    # Re-normalize to [-1, 1] within the displayed span so content tokens
    # actually show — otherwise a single outlier crushes everything.
    norm = scores / (max(np.abs(scores).max(), 1e-9))

    x = np.arange(len(tokens))
    labels = [_clean_token(t) for t in tokens]
    colors = ["#d62728" if s > 0 else "#1f77b4" for s in norm]
    ax.bar(x, norm, color=colors)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("normalized IG", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def _clean_token(t: str) -> str:
    """Render Llama BPE tokens readably (▁ marks word boundary)."""
    return t.replace("▁", " ").strip() or t
