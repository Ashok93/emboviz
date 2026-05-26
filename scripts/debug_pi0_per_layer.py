"""Per-layer attention pattern probe for π0 — checks if early layers
dominate the visualization via sinks (like GR00T) and late-layer-only
filter cleans things up.
"""
from __future__ import annotations

import numpy as np
import matplotlib.cm as cm
from PIL import Image

from emboviz.core.types import TokenSelector
from emboviz.datasets.lerobot_libero import PhysicalIntelligenceLiberoSource
from emboviz.models.pi0 import Pi0Adapter


def overlay(img, heat, out_path):
    h_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
    h_full = np.array(Image.fromarray((h_norm * 255).astype(np.uint8))
                      .resize(img.size, Image.BILINEAR)).astype(np.float32) / 255.0
    hot = (cm.hot(h_full)[..., :3] * 255).astype(np.uint8)
    base = np.array(img).astype(np.float32)
    blend = (base * 0.55 + hot * 0.45).clip(0, 255).astype(np.uint8)
    Image.fromarray(blend).save(out_path)


scene = PhysicalIntelligenceLiberoSource().load_trajectory(0).frames[0]
adapter = Pi0Adapter(config_name="pi0_libero", use_pytorch=True)
am = adapter.extract_attention(scene, TokenSelector(relative="before_action"))

print(f"weights shape: {am.weights.shape}")
print(f"image_token_ranges: {am.image_token_ranges}")
print()

# Per-layer mass to image tokens (sum across heads + per-cam tiles)
for cam in ["primary", "wrist"]:
    print(f"=== {cam} per-layer mass ===")
    weights = am.image_weights(cam)  # (L, H, side, side)
    L = weights.shape[0]
    for li in range(L):
        layer_attn = weights[li].mean(axis=0)  # mean over heads → (side, side)
        mass = float(layer_attn.sum())
        argmax = np.unravel_index(int(layer_attn.argmax()), layer_attn.shape)
        ratio = float(layer_attn.max() / (layer_attn.mean() + 1e-9))
        print(f"  L{li:2d}  mass={mass:.4f}  argmax={argmax}  peak/mean ratio={ratio:6.2f}")

# Render comparisons for primary
weights = am.image_weights("primary")
L = weights.shape[0]
src_img = scene.observations.images["primary"].data

raw = weights.mean(axis=(0, 1))
late = weights[L // 2:].mean(axis=(0, 1))    # late half
mid = weights[L // 3 : 2 * L // 3].mean(axis=(0, 1))  # middle third
overlay(src_img, raw, "/root/probes/pi0_FILTER_primary_raw.png")
overlay(src_img, late, "/root/probes/pi0_FILTER_primary_late.png")
overlay(src_img, mid, "/root/probes/pi0_FILTER_primary_mid.png")
print("\nsaved /root/probes/pi0_FILTER_primary_{raw,late,mid}.png")

weights_w = am.image_weights("wrist")
src_img_w = scene.observations.images["wrist"].data
raw_w = weights_w.mean(axis=(0, 1))
late_w = weights_w[L // 2:].mean(axis=(0, 1))
mid_w = weights_w[L // 3 : 2 * L // 3].mean(axis=(0, 1))
overlay(src_img_w, raw_w, "/root/probes/pi0_FILTER_wrist_raw.png")
overlay(src_img_w, late_w, "/root/probes/pi0_FILTER_wrist_late.png")
overlay(src_img_w, mid_w, "/root/probes/pi0_FILTER_wrist_mid.png")
print("saved /root/probes/pi0_FILTER_wrist_{raw,late,mid}.png")
