"""Poster-quality failure-scenario demo.

A single PNG that walks an engineer through:

  1. **The scene** + the WRONG instruction we'll test, with a huge headline.
  2. **The counterfactual matrix**: same scene, 6 instruction variants, each
     with its predicted action arrow overlaid. Identical arrows ⇒ noun blindness.
  3. **The mechanism**: top language-sensitive attention head's image attention
     under noun A vs noun B (the model DOES route on the noun internally,
     but downstream FFN ignores it).
  4. **The root cause**: a clean bar chart over BridgeV2's per-category
     coverage, with severity colour coding.
  5. **The recommendation**: the priority gap with a concrete recording brief.

Implementation note: we render each section into its own matplotlib figure,
save to PNG, and stitch them vertically with PIL. This gives us reliable
layout without fighting matplotlib's nested gridspec quirks.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from emboviz.action_viz import ActionArrow
from emboviz.attention_grounding import (
    AttentionGroundingResult,
    HeadLanguageSensitivity,
)
from emboviz.coverage_analysis import CoverageGap


@dataclass
class FailureDemoPayload:
    scene_image: Image.Image
    correct_instruction: str
    wrong_instruction: str
    variant_panels: list["VariantPanel"]
    attn_a: AttentionGroundingResult | None
    attn_b: AttentionGroundingResult | None
    noun_a: str
    noun_b: str
    top_heads: list[HeadLanguageSensitivity]
    coverage_gaps: list[CoverageGap]
    total_dataset_episodes: int
    paired_summary: dict | None
    headline_iss_noun_swap: float
    headline_iss_correct: float
    headline_iss_ood: float


@dataclass
class VariantPanel:
    instruction: str
    axis: str
    iss: float
    arrow: ActionArrow
    is_baseline: bool = False


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def render_failure_demo(payload: FailureDemoPayload, out_path: Path) -> None:
    """Render every section to a separate PNG, then stitch vertically."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        section_paths: list[Path] = []

        section_paths.append(_render_header(payload, tmp / "01_header.png"))
        section_paths.append(_render_cf_matrix(payload, tmp / "02_cf.png"))
        section_paths.append(_render_mechanism(payload, tmp / "03_mech.png"))
        section_paths.append(_render_root_cause(payload, tmp / "04_root.png"))
        section_paths.append(_render_recommendation(payload, tmp / "05_rec.png"))

        # Stitch with PIL on a white canvas with a small spacer between.
        imgs = [Image.open(p).convert("RGB") for p in section_paths]
        max_w = max(im.width for im in imgs)
        gap = 24
        total_h = sum(im.height for im in imgs) + gap * (len(imgs) - 1) + 40
        canvas = Image.new("RGB", (max_w + 80, total_h), "white")
        y = 20
        for im in imgs:
            # Centre horizontally
            x = (canvas.width - im.width) // 2
            canvas.paste(im, (x, y))
            y += im.height + gap
        canvas.save(out_path)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_header(p: FailureDemoPayload, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 1.3]})
    scene_np = np.array(p.scene_image)
    axes[0].imshow(scene_np)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("THE SCENE", fontsize=11, loc="left", fontweight="bold", color="#666")

    ax = axes[1]
    ax.axis("off")
    # Title
    ax.text(0.0, 1.0, "Emboviz — Real Failure Scenario", fontsize=22,
            fontweight="bold", color="#0d0d0d", transform=ax.transAxes,
            verticalalignment="top")

    # Wrong-instruction call-out
    ax.text(0.0, 0.83, "Engineer asked OpenVLA:", fontsize=10, color="#666",
            transform=ax.transAxes)
    ax.text(0.0, 0.75, f'"{p.wrong_instruction}"', fontsize=14,
            color="#c92a2a", fontweight="bold", transform=ax.transAxes)

    ax.text(0.0, 0.65, "Reality of the scene:", fontsize=10, color="#666",
            transform=ax.transAxes)
    ax.text(0.0, 0.57, f'There is no {p.noun_b}. There is a {p.noun_a}.',
            fontsize=12, color="#2b8a3e", fontweight="bold",
            transform=ax.transAxes)

    # The big numbers
    ax.text(0.0, 0.42, "DIAGNOSIS:", fontsize=10, color="#666", fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.0, 0.32, "Noun-blindness — model is producing the same action",
            fontsize=12, color="#c92a2a", fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.24, f"whether you say '{p.noun_a}' or '{p.noun_b}'.",
            fontsize=12, color="#c92a2a", fontweight="bold", transform=ax.transAxes)

    # Number callouts side by side
    ax.text(0.0, 0.10, "Action divergence under noun swap:", fontsize=9, color="#666",
            transform=ax.transAxes)
    ax.text(0.0, 0.00, f"{p.headline_iss_noun_swap:.3f}", fontsize=24,
            fontweight="bold", color="#c92a2a", transform=ax.transAxes)

    ax.text(0.55, 0.10, "OOD-task reference:", fontsize=9, color="#666",
            transform=ax.transAxes)
    ax.text(0.55, 0.00, f"{p.headline_iss_ood:.3f}", fontsize=24,
            fontweight="bold", color="#1971c2", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _render_cf_matrix(p: FailureDemoPayload, out: Path) -> Path:
    panels = p.variant_panels[:6]
    n = len(panels)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows), squeeze=False)
    fig.suptitle(
        "1.  The counterfactual matrix — same scene, different words",
        fontsize=14, fontweight="bold", color="#1971c2",
        x=0.02, y=0.995, ha="left",
    )
    fig.text(
        0.02, 0.965,
        "Each panel: predicted end-effector motion arrow on the same scene "
        "under a different instruction. Identical arrows ⇒ the model isn't listening to the words.",
        fontsize=10, color="#555", ha="left",
    )

    scene_np = np.array(p.scene_image)
    h, w = scene_np.shape[:2]
    cx, cy = w * 0.5, h * 0.7

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        if i >= n:
            ax.axis("off"); continue
        panel = panels[i]
        ax.imshow(scene_np)
        ax.set_xticks([]); ax.set_yticks([])
        _draw_arrow(ax, cx, cy, panel.arrow)
        if panel.is_baseline:
            tag, color = "BASELINE", "#2b8a3e"
        elif panel.iss < 0.10:
            tag, color = "IGNORED", "#c92a2a"
        elif panel.iss < 0.30:
            tag, color = "PARTIAL", "#fab005"
        else:
            tag, color = "FOLLOWED", "#1971c2"
        wrap = panel.instruction or "(empty)"
        if len(wrap) > 42:
            wrap = wrap[:40] + "…"
        ax.set_title(f"{tag}   ISS={panel.iss:.3f}\n\"{wrap}\"",
                     fontsize=10, color=color, loc="left")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _render_mechanism(p: FailureDemoPayload, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    fig.suptitle(
        "2.  The mechanism — does the model SEE the noun difference?",
        fontsize=14, fontweight="bold", color="#1971c2",
        x=0.02, y=0.99, ha="left",
    )
    fig.text(0.02, 0.93,
             "Top language-sensitive attention head: image patches the model "
             "looks at when each noun is spoken. Different focus regions "
             "⇒ the input pathway routes on the noun. "
             "Action stays the same anyway ⇒ downstream FFN is overriding it.",
             fontsize=10, color="#555", ha="left", wrap=True)

    scene_np = np.array(p.scene_image)
    _render_head_attention(axes[0], scene_np, p.attn_a, p.top_heads,
                           label=f'attention from "{p.noun_a}"', color="#2b8a3e")
    _render_head_attention(axes[1], scene_np, p.attn_b, p.top_heads,
                           label=f'attention from "{p.noun_b}"', color="#c92a2a")
    _render_head_js_bars(axes[2], p.top_heads[:10])

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _render_root_cause(p: FailureDemoPayload, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.suptitle(
        "3.  The root cause — your training data",
        fontsize=14, fontweight="bold", color="#1971c2",
        x=0.02, y=0.99, ha="left",
    )
    fig.text(
        0.02, 0.92,
        f"For each object category, how many of {p.total_dataset_episodes:,} unique "
        f"task descriptions in BridgeV2 contain ≥2 in-category objects? "
        f"(The pattern needed for noun-grounding to emerge.)",
        fontsize=10, color="#555", ha="left",
    )

    if not p.coverage_gaps:
        ax.text(0.5, 0.5, "(no coverage gaps to render)",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        gaps = sorted(p.coverage_gaps, key=lambda g: g.observed_count)
        cats = [g.failure_axis.replace("noun_swap on ", "") for g in gaps]
        counts = [g.observed_count for g in gaps]
        severities = [g.severity for g in gaps]
        palette = {"critical": "#c92a2a", "moderate": "#fab005", "ok": "#2b8a3e"}
        colors = [palette[s] for s in severities]
        y = np.arange(len(cats))[::-1]
        ax.barh(y, counts, color=colors, height=0.65)
        ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=11)
        for i, (cnt, sev) in enumerate(zip(counts, severities)):
            ax.text(cnt + max(counts) * 0.02, y[i],
                    f" {cnt:,}  ({sev.upper()})",
                    va="center", fontsize=10, color=palette[sev],
                    fontweight="bold")
        ax.set_xlabel(f"# of demos with ≥2 in-category objects "
                      f"(out of {p.total_dataset_episodes:,} BridgeV2 tasks)", fontsize=10)
        ax.set_xlim(0, max(counts) * 1.25 if counts else 1)
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _render_recommendation(p: FailureDemoPayload, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.axis("off")
    fig.suptitle(
        "4.  What to record — concrete data-collection brief",
        fontsize=14, fontweight="bold", color="#1971c2",
        x=0.02, y=0.97, ha="left",
    )

    # The caller has already promoted the *tested* category to the front of
    # the list; use it as the priority recommendation. This anchors the
    # recommendation to the actual failure axis, not the worst-coverage one.
    top = p.coverage_gaps[0] if p.coverage_gaps else None

    if top is None:
        ax.text(0.5, 0.5, "(no concrete recommendation generated)",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out

    palette = {"critical": "#c92a2a", "moderate": "#fab005", "ok": "#2b8a3e"}
    cat = top.failure_axis.split("on ")[-1]
    ax.text(0.0, 0.92,
            f"PRIORITY: noun_swap on '{cat}'  —  severity {top.severity.upper()}  "
            f"({top.observed_count:,} of {p.total_dataset_episodes:,} demos cover it)",
            fontsize=12, fontweight="bold", color=palette[top.severity],
            transform=ax.transAxes)

    y = 0.78
    for line in top.recommendation.splitlines()[:10]:
        ax.text(0.0, y, line, fontsize=10, color="#222", transform=ax.transAxes)
        y -= 0.07

    missing = top.details.get("pairs_missing_examples", [])[:10]
    if missing:
        y -= 0.02
        ax.text(0.0, y, "Missing object pairs to record (concrete list):",
                fontsize=10, fontweight="bold", color="#1971c2", transform=ax.transAxes)
        y -= 0.07
        pair_str = " · ".join(f"{a}+{b}" for a, b in missing)
        ax.text(0.0, y, pair_str, fontsize=10, color="#222", transform=ax.transAxes,
                wrap=True)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _draw_arrow(ax, x, y, arrow: ActionArrow) -> None:
    dx, dy = arrow.dx, arrow.dy
    color = "#d62728" if arrow.grip_close > 0.5 else "#1f77b4"
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", lw=4.5, color="white"))
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=color))
    ax.plot(x, y, "o", color="white", markersize=10)
    ax.plot(x, y, "o", color=color, markersize=6)


def _render_head_attention(ax, frame_np, attn_result, top_heads, label, color):
    if attn_result is None or not top_heads:
        ax.text(0.5, 0.5, "(no attention available)",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    top = top_heads[0]
    a_map = attn_result.attention_maps[top.layer, top.head]
    a_norm = (a_map - a_map.min()) / (a_map.max() - a_map.min() + 1e-9)
    overlaid = _overlay(frame_np, a_norm)
    ax.imshow(overlaid)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{label}\nL{top.layer}.H{top.head}", fontsize=11, color=color)


def _render_head_js_bars(ax, heads):
    if not heads:
        ax.text(0.5, 0.5, "(no head data)", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    labels = [f"L{h.layer}.H{h.head}" for h in heads]
    vals = [h.js_divergence for h in heads]
    y = np.arange(len(heads))[::-1]
    ax.barh(y, vals, color="#1971c2")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("JS divergence (noun A vs noun B attention)", fontsize=10)
    ax.set_title(f"Top {len(heads)} heads that DO route on the noun",
                 fontsize=11, color="#1971c2")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _overlay(frame, heatmap, alpha=0.55):
    if heatmap.shape != frame.shape[:2]:
        pil = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8), mode="L")
        pil = pil.resize((frame.shape[1], frame.shape[0]), Image.BILINEAR)
        heatmap = np.asarray(pil, dtype=np.float32) / 255.0
    cmap = plt.get_cmap("jet")
    colored = (cmap(heatmap)[..., :3] * 255).astype(np.uint8)
    blended = frame.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
