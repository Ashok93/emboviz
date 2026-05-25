"""Verdict-card visualization for Module B.

The output of this module is what a robotics engineer would see for one
rollout. It must answer three questions at a glance:

  1. Did the model actually listen to the instruction?  (counterfactual
     divergence bar chart + verdict text)
  2. WHERE in the image was it looking when you said the noun vs an
     alternative noun?  (attention overlays for both nouns side by side)
  3. What should I do about it?  (recommendation block)

Layout (one PNG):
    +----------------------------+----------------------------+
    |  scene image (large)       |  attention: noun_A on img  |
    |  + action arrows           |  attention: noun_B on img  |
    +----------------------------+----------------------------+
    |  bar chart: action divergence per counterfactual variant |
    +-----------------------------------------------------------+
    |  VERDICT (large)  +  diagnostic numbers                   |
    |  RECOMMENDATION                                           |
    +-----------------------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from policylens.attention_grounding import (
    AttentionGroundingResult,
    HeadLanguageSensitivity,
    aggregate_attention_across_heads,
)
from policylens.counterfactual import CounterfactualResult


@dataclass
class VerdictPayload:
    base_image: Image.Image
    base_action: np.ndarray
    cf_result: CounterfactualResult
    verdict_tag: str
    verdict_text: str
    attn_noun_a: AttentionGroundingResult | None
    attn_noun_b: AttentionGroundingResult | None
    noun_a: str
    noun_b: str
    head_sensitivities: list[HeadLanguageSensitivity]
    recommendation: str


def render_verdict_card(payload: VerdictPayload, out_path: Path) -> None:
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(
        nrows=4, ncols=3,
        height_ratios=[3, 2, 0.8, 1.2],
        hspace=0.45, wspace=0.30,
    )

    # ---- Row 0: scene image + two attention overlays --------------------
    ax_scene = fig.add_subplot(gs[0, 0])
    frame_np = np.array(payload.base_image)
    ax_scene.imshow(frame_np)
    ax_scene.set_title(f'scene\nbaseline instruction:\n"{payload.cf_result.base_instruction}"',
                       fontsize=10)
    ax_scene.set_xticks([]); ax_scene.set_yticks([])

    ax_a = fig.add_subplot(gs[0, 1])
    if payload.attn_noun_a is not None:
        heat = aggregate_attention_across_heads(payload.attn_noun_a)
        ax_a.imshow(_overlay(frame_np, heat))
        ax_a.set_title(f'attention from noun "{payload.noun_a}"\n(mean over heads, late layers)',
                       fontsize=10, color="#2b8a3e")
    else:
        ax_a.text(0.5, 0.5, f'"{payload.noun_a}" not in prompt',
                  ha="center", va="center", fontsize=10)
    ax_a.set_xticks([]); ax_a.set_yticks([])

    ax_b = fig.add_subplot(gs[0, 2])
    if payload.attn_noun_b is not None:
        heat = aggregate_attention_across_heads(payload.attn_noun_b)
        ax_b.imshow(_overlay(frame_np, heat))
        ax_b.set_title(f'attention from noun "{payload.noun_b}"\n(under counterfactual prompt)',
                       fontsize=10, color="#c92a2a")
    else:
        ax_b.text(0.5, 0.5, f'"{payload.noun_b}" not in prompt',
                  ha="center", va="center", fontsize=10)
    ax_b.set_xticks([]); ax_b.set_yticks([])

    # ---- Row 1: divergence bar chart + head-sensitivity bar chart -------
    ax_bars = fig.add_subplot(gs[1, :2])
    variants = list(payload.cf_result.instruction_sensitivity.keys())
    iss_values = [payload.cf_result.instruction_sensitivity[v] for v in variants]
    y = np.arange(len(variants))[::-1]
    colors = ["#2b8a3e" if v >= 0.30 else ("#fab005" if v >= 0.05 else "#c92a2a") for v in iss_values]
    ax_bars.barh(y, iss_values, color=colors)
    ax_bars.set_yticks(y)
    ax_bars.set_yticklabels([_short(v) for v in variants], fontsize=9)
    ax_bars.axvline(0.05, color="#888", linestyle="--", lw=1, label="noise floor (0.05)")
    ax_bars.axvline(0.30, color="#444", linestyle="--", lw=1, label="grounded (0.30)")
    ax_bars.set_xlabel("Instruction Sensitivity Score (mean ||Δaction||₂ vs baseline)", fontsize=10)
    ax_bars.set_title("How much does the action change when we swap the instruction?", fontsize=11)
    ax_bars.legend(loc="lower right", fontsize=8)
    ax_bars.grid(axis="x", alpha=0.3)

    ax_heads = fig.add_subplot(gs[1, 2])
    if payload.head_sensitivities:
        top = sorted(payload.head_sensitivities, key=lambda h: -h.js_divergence)[:10]
        labels = [f"L{h.layer}.H{h.head}" for h in top]
        vals = [h.js_divergence for h in top]
        y2 = np.arange(len(top))[::-1]
        ax_heads.barh(y2, vals, color="#1971c2")
        ax_heads.set_yticks(y2)
        ax_heads.set_yticklabels(labels, fontsize=8)
        ax_heads.set_xlabel("JS divergence (A vs B attention)", fontsize=9)
        ax_heads.set_title("Top 10 language-sensitive heads", fontsize=10)
        ax_heads.grid(axis="x", alpha=0.3)

    # ---- Row 2: verdict text -------------------------------------------
    ax_verd = fig.add_subplot(gs[2, :])
    ax_verd.axis("off")
    tag_text = {
        "language_blind": "LANGUAGE BLINDNESS CONFIRMED",
        "partial": "PARTIAL GROUNDING",
        "grounded": "GROUNDED",
        "unknown": "VERDICT UNAVAILABLE",
    }[payload.verdict_tag]
    tag_color = {
        "language_blind": "#c92a2a",
        "partial": "#fab005",
        "grounded": "#2b8a3e",
        "unknown": "#666",
    }[payload.verdict_tag]
    ax_verd.text(0.0, 0.7, tag_text, fontsize=18, fontweight="bold",
                 color=tag_color, transform=ax_verd.transAxes)
    ax_verd.text(0.0, 0.05, payload.verdict_text, fontsize=10,
                 transform=ax_verd.transAxes, wrap=True)

    # ---- Row 3: recommendation ----------------------------------------
    ax_rec = fig.add_subplot(gs[3, :])
    ax_rec.axis("off")
    ax_rec.text(0.0, 0.92, "RECOMMENDATION", fontsize=12, fontweight="bold",
                color="#1971c2", transform=ax_rec.transAxes)
    ax_rec.text(0.0, 0.0, payload.recommendation, fontsize=9,
                transform=ax_rec.transAxes, verticalalignment="bottom", wrap=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---- helpers ---------------------------------------------------------------


def _overlay(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    if heatmap.shape != frame.shape[:2]:
        pil = Image.fromarray((heatmap * 255).clip(0, 255).astype(np.uint8), mode="L")
        pil = pil.resize((frame.shape[1], frame.shape[0]), Image.BILINEAR)
        heatmap = np.asarray(pil, dtype=np.float32) / 255.0
    cmap = plt.get_cmap("jet")
    colored = (cmap(heatmap)[..., :3] * 255).astype(np.uint8)
    blended = frame.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _short(s: str, n: int = 50) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"
