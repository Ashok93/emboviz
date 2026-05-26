"""Inspect GR00T per-layer + per-head attention to diagnose sink pattern.

Saves a 16×16 grid of mini-heatmaps (one per layer×head) for the primary
camera's FIRST tile (no temporal summing). If most heads concentrate on
corners → sink phenomenon. If some heads show clean table-center
attention → averaging is washing out the real signal.

Also reports per-tile attention masses BEFORE summing, so we can see if
tile 0 and tile 2 (both primary, same image replicated) show identical
attention or different patterns.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from emboviz.core.types import TokenSelector
from emboviz.datasets.lerobot_droid import GR00TDroidSampleSource
from emboviz.models.gr00t import Gr00tAdapter

scene = GR00TDroidSampleSource().load_trajectory(1).frames[0]
adapter = Gr00tAdapter(camera_mapping={
    "primary":    "exterior_image_1_left",
    "wrist_left": "wrist_image_left",
})
am = adapter.extract_attention(scene, TokenSelector(relative="before_action"))

print(f"weights shape:  {am.weights.shape}")
print(f"image_token_ranges: {am.image_token_ranges}")
print(f"image_grid_sides:   {am.image_grid_sides}")

# Primary has 2 tile ranges. Inspect them SEPARATELY.
ranges = am.image_token_ranges["primary"]
side = am.image_grid_sides["primary"]
print(f"\nPrimary tile ranges: {ranges}, grid_side={side}")

# Per-tile attention BEFORE summing
per_tile = []
for ti, (s, e) in enumerate(ranges):
    tile_attn = am.weights[..., s:e].reshape(*am.weights.shape[:-1], side, side)
    agg = tile_attn.mean(axis=(0, 1))  # mean over layers + heads
    mass = float(agg.sum())
    print(f"  tile {ti} (positions {s}..{e}):  shape={tile_attn.shape}  "
          f"mean-across-LH range=[{agg.min():.4e}, {agg.max():.4e}]  mass={mass:.4f}")
    per_tile.append(tile_attn)

# Compare: are the 2 tiles' attentions IDENTICAL (same image replicated)?
diff = np.abs(per_tile[0].mean(axis=(0, 1)) - per_tile[1].mean(axis=(0, 1))).sum()
print(f"\nL1 diff between tile 0 and tile 1 (avg-LH heatmaps): {diff:.4e}")

# Grid of per-layer × per-head heatmaps for tile 0 (the primary's first tile)
tile0 = per_tile[0]   # shape (L, H, side, side)
L, H, _, _ = tile0.shape
print(f"\nRendering {L}x{H} per-layer × per-head grid ...")
fig, axes = plt.subplots(L, H, figsize=(H * 0.8, L * 0.8))
for li in range(L):
    for hi in range(H):
        ax = axes[li, hi]
        ax.imshow(tile0[li, hi], cmap="hot")
        ax.set_xticks([]); ax.set_yticks([])
        if hi == 0:
            ax.set_ylabel(f"L{li}", fontsize=4)
        if li == 0:
            ax.set_title(f"H{hi}", fontsize=4)
plt.tight_layout()
plt.savefig("/root/probes/per_layer_head_gr00t_primary_tile0.png", dpi=80)
plt.close()
print("Saved /root/probes/per_layer_head_gr00t_primary_tile0.png")

# Save raw stats per layer + head
stats = []
for li in range(L):
    for hi in range(H):
        m = tile0[li, hi]
        stats.append({
            "layer": li, "head": hi,
            "min": float(m.min()), "max": float(m.max()),
            "mass": float(m.sum()),
            "argmax": [int(x) for x in np.unravel_index(int(m.argmax()), m.shape)],
        })
with open("/root/probes/per_layer_head_gr00t_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# Most-concentrated head per layer (one that has the LEAST uniform mass)
print("\nMost-focused head per layer (max-cell / mean-cell ratio):")
for li in range(L):
    ratios = []
    for hi in range(H):
        m = tile0[li, hi]
        ratios.append((hi, float(m.max() / (m.mean() + 1e-9))))
    ratios.sort(key=lambda x: -x[1])
    print(f"  L{li}: best head={ratios[0][0]} ratio={ratios[0][1]:.2f}  "
          f"argmax={np.unravel_index(int(tile0[li, ratios[0][0]].argmax()), (side, side))}")
