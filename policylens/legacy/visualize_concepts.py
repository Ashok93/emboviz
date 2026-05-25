"""Concept-level visualizations for Stage B v2.

Three plots that translate the FFN-neuron-level analysis into something a
robotics engineer can actually read:

  • `render_top_concepts_bars`     — per-keyframe horizontal bar chart of
    the model's most-used concepts (with their semantic labels).
  • `render_concept_timeline`      — concepts on Y axis, time on X axis;
    color = activation magnitude. Shows which concepts come and go.
  • `render_anomaly_comparison`    — failure-frame vs baseline-mean for the
    smoking-gun anomalous concepts. The headline plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from policylens.concepts import ConceptAnomaly, FrameConcepts
from policylens.visual_attribution import NeuronImageMap


def render_top_concepts_bars(
    per_frame: dict[int, FrameConcepts],
    keyframes: list[int],
    out_path: Path,
    title: str = "",
    top_n: int = 12,
) -> None:
    """N rows of horizontal bar charts — one per keyframe.

    Each row: the top-N concepts driving the action at that frame, labeled
    with their semantic dictionary entry. Bars are |activation| × ||value||.
    """
    n = len(keyframes)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 + 0.5 * top_n * n / max(1, n)),
                             squeeze=False)
    if title:
        fig.suptitle(title, fontsize=12, y=1.0)

    for row, k in enumerate(keyframes):
        ax = axes[row, 0]
        hits = per_frame[k].top_hits[:top_n]
        labels = [f"L{h.layer}.{h.neuron}  {h.short_label()}" for h in hits]
        values = [h.contribution for h in hits]
        y = np.arange(len(hits))[::-1]
        ax.barh(y, values, color="#d62728")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("|activation| × ||value vector||", fontsize=9)
        ax.set_title(f"t={k}  ·  top concepts driving this action", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_concept_timeline(
    per_frame: dict[int, FrameConcepts],
    out_path: Path,
    failure_idx: int | None = None,
    top_n: int = 25,
) -> None:
    """Concepts on Y, time on X, heatmap = contribution magnitude.

    We pick the top-N concepts by max contribution across the whole episode,
    so the displayed concepts are always meaningful and the heatmap actually
    shows the rise and fall of the model's "thoughts" over time.
    """
    frame_indices = sorted(per_frame.keys())

    # Union of all (layer, neuron) keys.
    union = set()
    for c in per_frame.values():
        union.update(c.full.keys())

    # Build a (concept, frame) matrix.
    keys = list(union)
    mat = np.zeros((len(keys), len(frame_indices)), dtype=np.float32)
    for fi, ti in enumerate(frame_indices):
        for ki, key in enumerate(keys):
            mat[ki, fi] = per_frame[ti].full.get(key, 0.0)

    # Rank concepts by their max-over-time contribution.
    order = np.argsort(-mat.max(axis=1))[:top_n]
    mat = mat[order]
    keys = [keys[i] for i in order]

    # Labels — pull from dictionary via per_frame's top_hits cache when
    # available (each ConceptHit carries its labels).
    label_cache: dict[tuple[int, int], str] = {}
    for c in per_frame.values():
        for h in c.top_hits:
            label_cache[(h.layer, h.neuron)] = h.short_label()
    labels = [
        f"L{li}.{ni}  {label_cache.get((li, ni), '?')}" for (li, ni) in keys
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(frame_indices) * 0.6), 0.4 * top_n + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(np.arange(len(frame_indices)))
    ax.set_xticklabels([f"t={t}" for t in frame_indices], rotation=45, fontsize=8)
    ax.set_title(f"Concept activation over time  (top {top_n} concepts in this episode)")
    fig.colorbar(im, ax=ax, label="|activation| × ||value||")
    if failure_idx is not None and failure_idx in frame_indices:
        col = frame_indices.index(failure_idx)
        ax.axvline(col, color="cyan", lw=2, ls="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_cross_modal_binding(
    frame_image: Image.Image,
    anomalies: list[ConceptAnomaly],
    per_neuron_maps: list[NeuronImageMap],
    attention_rollout: np.ndarray | None,
    out_path: Path,
    title: str = "",
) -> None:
    """The headline cross-modal panel.

    Layout: top row = original image + (optional) attention-rollout overlay.
    Then one row per smoking-gun neuron: name + language label + per-neuron
    image map (the patches that drive that specific named concept).
    """
    n = len(anomalies)
    rows = 1 + n
    fig = plt.figure(figsize=(11, 3.0 * rows))

    # --- Top row: raw frame + attention rollout ---
    ax_img = fig.add_subplot(rows, 2, 1)
    ax_img.imshow(np.array(frame_image))
    ax_img.set_title("input frame", fontsize=11)
    ax_img.set_xticks([]); ax_img.set_yticks([])

    if attention_rollout is not None:
        ax_att = fig.add_subplot(rows, 2, 2)
        ax_att.imshow(_overlay(np.array(frame_image), attention_rollout))
        ax_att.set_title("attention rollout — where the model is looking (macro)", fontsize=11)
        ax_att.set_xticks([]); ax_att.set_yticks([])

    # --- One row per smoking-gun neuron ---
    map_by_key = {(m.layer, m.neuron): m for m in per_neuron_maps}
    for i, a in enumerate(anomalies):
        row_idx = i + 1
        label = a.short_label()
        info = (f"L{a.layer}.N{a.neuron}    z={a.z_score:.1f}    "
                f"failure={a.failure_contribution:.2f}  baseline={a.baseline_mean:.2f}±{a.baseline_std:.2f}")
        ax_l = fig.add_subplot(rows, 2, row_idx * 2 + 1)
        ax_l.axis("off")
        ax_l.text(0.02, 0.92, label, fontsize=14, fontweight="bold",
                  transform=ax_l.transAxes, color="#d62728")
        ax_l.text(0.02, 0.72, info, fontsize=9, transform=ax_l.transAxes)
        # Top tokens table
        tokens_line1 = " ".join(a.label_tokens[:6])
        tokens_line2 = " ".join(a.label_tokens[6:12])
        ax_l.text(0.02, 0.45, "top tokens:", fontsize=8, transform=ax_l.transAxes, color="#555")
        ax_l.text(0.02, 0.30, tokens_line1, fontsize=9, transform=ax_l.transAxes)
        ax_l.text(0.02, 0.15, tokens_line2, fontsize=9, transform=ax_l.transAxes, color="#555")

        ax_r = fig.add_subplot(rows, 2, row_idx * 2 + 2)
        nmap = map_by_key.get((a.layer, a.neuron))
        if nmap is not None:
            ax_r.imshow(_overlay(np.array(frame_image), nmap.heatmap))
            ax_r.set_title(f"image patches driving this neuron", fontsize=10)
        else:
            ax_r.text(0.5, 0.5, "(no image map computed)",
                      ha="center", va="center", transform=ax_r.transAxes, fontsize=10)
        ax_r.set_xticks([]); ax_r.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _overlay(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a normalized heatmap onto an RGB frame using the 'jet' colormap."""
    if heatmap.shape != frame.shape[:2]:
        # Nearest-neighbour-style resize via PIL.
        pil = Image.fromarray((heatmap * 255).clip(0, 255).astype(np.uint8), mode="L")
        pil = pil.resize((frame.shape[1], frame.shape[0]), Image.BILINEAR)
        heatmap = np.asarray(pil, dtype=np.float32) / 255.0
    cmap = plt.get_cmap("jet")
    colored = (cmap(heatmap)[..., :3] * 255).astype(np.uint8)
    blended = frame.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def render_anomaly_comparison(
    anomalies: list[ConceptAnomaly],
    failure_idx: int,
    out_path: Path,
    title: str = "",
) -> None:
    """The headline 'smoking gun' plot. Each anomalous concept gets a row:
    failure-frame contribution as a tall bar, baseline mean ± std as a
    horizontal range. Z-score annotated.
    """
    if not anomalies:
        # Render an empty placeholder so the script doesn't crash mid-flight.
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No anomalous concepts found (z < threshold).",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return

    n = len(anomalies)
    fig, ax = plt.subplots(figsize=(11, 0.5 * n + 2))
    y = np.arange(n)[::-1]
    failure_vals = [a.failure_contribution for a in anomalies]
    baselines = [a.baseline_mean for a in anomalies]
    errs = [a.baseline_std for a in anomalies]

    ax.errorbar(baselines, y, xerr=errs, fmt="o", color="#888",
                ecolor="#bbb", markersize=6, label="baseline frames (mean ± std)")
    ax.barh(y, failure_vals, color="#d62728", alpha=0.85,
            label=f"failure frame t={failure_idx}")

    for i, a in enumerate(anomalies):
        ax.text(failure_vals[i], y[i], f"  z={a.z_score:.1f}",
                va="center", fontsize=9, color="black")

    labels = [f"L{a.layer}.{a.neuron}  {a.short_label()}" for a in anomalies]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("|activation| × ||value vector||", fontsize=10)
    ax.set_title(title or
        f"Smoking-gun concepts: unusually active at failure frame t={failure_idx}", fontsize=11)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
