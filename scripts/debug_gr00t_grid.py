"""Debug helper: print Qwen3-VL input_ids + image_grid_thw shapes for GR00T.

Reveals the layout Qwen3-VL receives so we can fix image-token-to-camera
mapping in extract_attention.
"""
import torch
import numpy as np

from emboviz.datasets.lerobot_droid import GR00TDroidSampleSource
from emboviz.models.gr00t import Gr00tAdapter

scene = GR00TDroidSampleSource().load_trajectory(1).frames[0]
adapter = Gr00tAdapter(camera_mapping={
    "primary":    "exterior_image_1_left",
    "wrist_left": "wrist_image_left",
})
qwen = adapter.policy.model.backbone.model

captured = {}
orig_forward = qwen.forward

def patched(*args, **kwargs):
    kwargs["output_attentions"] = True
    result = orig_forward(*args, **kwargs)
    captured["input_ids"] = kwargs.get("input_ids")
    captured["image_grid_thw"] = kwargs.get("image_grid_thw")
    captured["pixel_values"] = kwargs.get("pixel_values")
    captured["attentions"] = result.attentions
    return result

qwen.set_attn_implementation("eager")
qwen.forward = patched
try:
    obs = adapter._build_observation(scene)
    with torch.inference_mode():
        _ = adapter.policy.get_action(obs)
finally:
    qwen.forward = orig_forward

ids = captured["input_ids"][0].cpu().numpy()
itok = qwen.config.image_token_id
grid = captured["image_grid_thw"].cpu().numpy()
pix = captured["pixel_values"].shape if captured["pixel_values"] is not None else None
merge = int(qwen.config.vision_config.spatial_merge_size)

print(f"input_ids shape         : {captured['input_ids'].shape}")
print(f"image_token_id          : {itok}")
print(f"# image tokens in ids   : {int((ids == itok).sum())}")
print(f"image_grid_thw shape    : {captured['image_grid_thw'].shape}")
print(f"image_grid_thw values   :")
for i, (t, h, w) in enumerate(grid):
    print(f"  image[{i}]: T={t} H={h} W={w}  → tokens = {t * (h // merge) * (w // merge)}")
print(f"pixel_values shape      : {pix}")
print(f"spatial_merge_size      : {merge}")
print(f"# attention layers (kept): {len(captured['attentions'])}")
print(f"attn[0] shape            : {captured['attentions'][0].shape}")

# Inspect the contiguous runs of image tokens
runs = []
in_run, start = False, 0
for i, t in enumerate(ids):
    if t == itok and not in_run:
        in_run = True
        start = i
    elif t != itok and in_run:
        in_run = False
        runs.append((start, i))
if in_run:
    runs.append((start, len(ids)))
print(f"\n# contiguous image-token runs in ids: {len(runs)}")
for i, (s, e) in enumerate(runs):
    print(f"  run[{i}]: positions {s}..{e}  ({e-s} tokens)")
