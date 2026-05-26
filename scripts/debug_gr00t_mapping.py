"""Definitive test for GR00T camera-vs-tile ordering bug.

We extract attention twice:
  1. Original frame (primary + wrist_left as-is)
  2. Primary REPLACED with pure red (255,0,0)

If our tile-to-camera mapping is correct (camera-major), then between
the two runs:
  - "primary" attention mass changes substantially (different image)
  - "wrist_left" attention mass barely changes (same image)

If the mapping is WRONG (time-major), the opposite — wrist_left changes,
primary doesn't.

This is the definitive empirical test.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from PIL import Image

from emboviz.core.types import TokenSelector
from emboviz.datasets.lerobot_droid import GR00TDroidSampleSource
from emboviz.models.gr00t import Gr00tAdapter

# Load
scene = GR00TDroidSampleSource().load_trajectory(1).frames[0]
adapter = Gr00tAdapter(camera_mapping={
    "primary":    "exterior_image_1_left",
    "wrist_left": "wrist_image_left",
})

print("=== Run 1: original ===")
am1 = adapter.extract_attention(scene, TokenSelector(relative="before_action"))
mass1_p = float(am1.image_weights("primary").mean(axis=(0, 1)).sum())
mass1_w = float(am1.image_weights("wrist_left").mean(axis=(0, 1)).sum())
print(f"  primary mass:    {mass1_p:.4f}")
print(f"  wrist_left mass: {mass1_w:.4f}")
print(f"  per-camera ranges: {am1.image_token_ranges}")

# Construct the modified scene: primary → pure red
print("\n=== Run 2: primary replaced with PURE RED ===")
orig_primary = np.asarray(scene.observations.images["primary"].data)
red_primary = np.zeros_like(orig_primary)
red_primary[..., 0] = 255   # pure red
scene2 = scene.with_image(Image.fromarray(red_primary), camera="primary")

am2 = adapter.extract_attention(scene2, TokenSelector(relative="before_action"))
mass2_p = float(am2.image_weights("primary").mean(axis=(0, 1)).sum())
mass2_w = float(am2.image_weights("wrist_left").mean(axis=(0, 1)).sum())
print(f"  primary mass:    {mass2_p:.4f}  (Δ from run1: {mass2_p - mass1_p:+.4f})")
print(f"  wrist_left mass: {mass2_w:.4f}  (Δ from run1: {mass2_w - mass1_w:+.4f})")

# Compute per-cell deltas: which heatmap CHANGED?
hm1_p = am1.image_weights("primary").mean(axis=(0, 1))
hm2_p = am2.image_weights("primary").mean(axis=(0, 1))
hm1_w = am1.image_weights("wrist_left").mean(axis=(0, 1))
hm2_w = am2.image_weights("wrist_left").mean(axis=(0, 1))

primary_change = float(np.abs(hm2_p - hm1_p).sum())
wrist_change   = float(np.abs(hm2_w - hm1_w).sum())

print(f"\n=== Δ when primary→red ===")
print(f"primary mass:    {mass1_p:.4f} → {mass2_p:.4f}  (Δ {mass2_p - mass1_p:+.4f})")
print(f"wrist_left mass: {mass1_w:.4f} → {mass2_w:.4f}  (Δ {mass2_w - mass1_w:+.4f})")

# Symmetric test: now mask wrist_left
print("\n=== Run 3: wrist_left replaced with PURE RED (primary restored) ===")
orig_wrist = np.asarray(scene.observations.images["wrist_left"].data)
red_wrist = np.zeros_like(orig_wrist)
red_wrist[..., 0] = 255
scene3 = scene.with_image(Image.fromarray(red_wrist), camera="wrist_left")
am3 = adapter.extract_attention(scene3, TokenSelector(relative="before_action"))
mass3_p = float(am3.image_weights("primary").mean(axis=(0, 1)).sum())
mass3_w = float(am3.image_weights("wrist_left").mean(axis=(0, 1)).sum())
print(f"primary mass:    {mass1_p:.4f} → {mass3_p:.4f}  (Δ {mass3_p - mass1_p:+.4f})")
print(f"wrist_left mass: {mass1_w:.4f} → {mass3_w:.4f}  (Δ {mass3_w - mass1_w:+.4f})")

print("\n=== Verdict ===")
# Test predicts: masking camera X should DECREASE camera X's mass (red is uninformative)
primary_test_correct = (mass2_p - mass1_p) < (mass2_w - mass1_w)
wrist_test_correct   = (mass3_w - mass1_w) < (mass3_p - mass1_p)
print(f"masking primary → primary mass went down more than wrist?     {primary_test_correct}")
print(f"masking wrist_left → wrist mass went down more than primary? {wrist_test_correct}")
if primary_test_correct and wrist_test_correct:
    print("\n✓ MAPPING CORRECT — each camera's mass responds to its own input")
elif (not primary_test_correct) and (not wrist_test_correct):
    print("\n✗ MAPPING SWAPPED — masking X changes Y's mass and vice versa. "
          "Swap the tile-to-camera mapping in extract_attention.")
else:
    print("\n? PARTIAL — only one direction looks right; investigate per-tile mapping further")
