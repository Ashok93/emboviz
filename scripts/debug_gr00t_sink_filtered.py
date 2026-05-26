"""Apply sink-mask + late-layer-only filter to GR00T attention, render
the resulting heatmap. If filtered signal pops on the manipulation area,
the corner brightness in the unfiltered map IS the documented RoPE sink.
"""
from __future__ import annotations

import numpy as np
import matplotlib.cm as cm
from PIL import Image

from emboviz.core.types import TokenSelector
from emboviz.datasets.lerobot_droid import GR00TDroidSampleSource
from emboviz.models.gr00t import Gr00tAdapter


def overlay(img, heat, out_path):
    h_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
    h_full = np.array(Image.fromarray((h_norm * 255).astype(np.uint8))
                      .resize(img.size, Image.BILINEAR)).astype(np.float32) / 255.0
    hot = (cm.hot(h_full)[..., :3] * 255).astype(np.uint8)
    base = np.array(img).astype(np.float32)
    blend = (base * 0.55 + hot * 0.45).clip(0, 255).astype(np.uint8)
    Image.fromarray(blend).save(out_path)


scene = GR00TDroidSampleSource().load_trajectory(1).frames[0]
adapter = Gr00tAdapter(camera_mapping={
    "primary":    "exterior_image_1_left",
    "wrist_left": "wrist_image_left",
})
am = adapter.extract_attention(scene, TokenSelector(relative="before_action"))

for cam in ["primary", "wrist_left"]:
    src_img = scene.observations.images[cam].data
    weights = am.image_weights(cam)  # (L, H, side, side)
    L, H, S, _ = weights.shape

    # 1. RAW (current default): mean over all L × H
    raw = weights.mean(axis=(0, 1))
    overlay(src_img, raw, f"/root/probes/gr00t_FILTER_{cam}_raw.png")

    # 2. LATE LAYERS only (drop early sink-heavy layers — L0..L7)
    late = weights[L // 2 :].mean(axis=(0, 1))
    overlay(src_img, late, f"/root/probes/gr00t_FILTER_{cam}_late_layers.png")

    # 3. LATE LAYERS + SINK MASK (zero out the rightmost column, the
    #    documented RoPE recency-bias dump position)
    late_masked = late.copy()
    late_masked[:, -1] = 0   # zero rightmost column
    late_masked[-1, :] = 0   # zero bottom row (bottom-right also sinks)
    overlay(src_img, late_masked, f"/root/probes/gr00t_FILTER_{cam}_late_layers_sink_masked.png")

    print(f"=== {cam} ===")
    print(f"  raw            mass={raw.sum():.4f}  range=[{raw.min():.4e},{raw.max():.4e}]")
    print(f"  late only      mass={late.sum():.4f}  range=[{late.min():.4e},{late.max():.4e}]")
    print(f"  late + nosink  mass={late_masked.sum():.4f}  range=[{late_masked.min():.4e},{late_masked.max():.4e}]")
    print(f"  argmax late+nosink: {np.unravel_index(int(late_masked.argmax()), late_masked.shape)}")
