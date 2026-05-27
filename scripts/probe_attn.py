"""Probe: dump RAW (mean over all heads/layers, no selection, no sink removal)
vs CLEANED per-camera attention overlays, and print the image-token mapping
numbers — to tell apart (a) a mapping bug, (b) over-filtering, (c) genuinely
diffuse attention. One frame only.
"""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
import numpy as np
from PIL import Image

from emboviz.core.types import TokenSelector
from emboviz.exporters.rerun import _heatmap_rgba


def _resolve(spec: str, kwargs_json: str = ""):
    parts = spec.split(":")
    module = importlib.import_module(parts[0])
    obj = getattr(module, parts[1])
    kwargs = json.loads(kwargs_json) if kwargs_json else {}
    if len(parts) == 2:
        return obj(**kwargs)
    intermediate = obj(parts[2])
    return intermediate(**kwargs) if isinstance(intermediate, type) else intermediate


def _overlay(rgb, heat2d, path):
    H, W = rgb.shape[:2]
    rgba = _heatmap_rgba(heat2d, (H, W), cmap_name="turbo")
    if rgba is None:
        print(f"    (flat/zero -> no overlay) {path.name}")
        return
    base = Image.fromarray(rgb).convert("RGBA")
    Image.alpha_composite(base, Image.fromarray(rgba, "RGBA")).convert("RGB").save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-builder", required=True)
    p.add_argument("--model-kwargs", default="")
    p.add_argument("--dataset-builder", required=True)
    p.add_argument("--dataset-kwargs", default="")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--heads-cache", default="", help="JSON from calibrate_attention.py (calibrated localization heads)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = _resolve(args.model_builder, args.model_kwargs)
    dataset = _resolve(args.dataset_builder, args.dataset_kwargs)
    scene = dataset.load_episode(str(args.episode))[args.frame]
    print(f"[probe] ep{args.episode} f{args.frame} cams={sorted(scene.observations.images)} "
          f"instr={scene.instruction!r}")

    attn = model.extract_attention(scene, TokenSelector(relative="before_action"))
    if args.heads_cache:
        cache = json.loads(Path(args.heads_cache).read_text())
        hbc = {c: [tuple(h) for h in hs] for c, hs in cache.get("heads_by_camera", {}).items()}
        attn.metadata["localization_heads_by_camera"] = hbc
        print(f"[probe] injected calibrated heads: {hbc}")
    print(f"[probe] weights.shape (L,H,n_keys) = {attn.weights.shape}  n_keys={attn.n_keys}")
    print(f"[probe] metadata: grid_thw={attn.metadata.get('image_grid_thw')} "
          f"n_image_tokens_total={attn.metadata.get('n_image_tokens_total')} "
          f"tokens_per_image={attn.metadata.get('tokens_per_image')}")
    print(f"[probe] image_token_ranges={attn.image_token_ranges} grid_sides={attn.image_grid_sides}")

    for cam in attn.cameras:
        if cam not in scene.observations.images:
            print(f"[probe] skip padding slot {cam!r}"); continue
        rgb = np.asarray(scene.observations.images[cam].data)
        Image.fromarray(rgb).save(out / f"{cam}_rgb.png")
        raw = attn.image_weights(cam)             # (L,H,side,side)
        raw_mean = raw.mean(axis=(0, 1))          # NO selection, NO sink removal
        _overlay(rgb, raw_mean, out / f"{cam}_RAW.png")
        cleaned, dbg = attn.image_weights_clean(cam)
        _overlay(rgb, cleaned, out / f"{cam}_CLEAN.png")
        print(f"[probe] {cam}: side={raw.shape[-1]} "
              f"raw_peak={np.unravel_index(int(raw_mean.argmax()), raw_mean.shape)} "
              f"clean_peak={np.unravel_index(int(cleaned.argmax()), cleaned.shape)}")
        print(f"        selected_heads={dbg.get('selected_heads')} "
              f"sink_cells={int(np.array(dbg.get('selected_head_imgmass',[])).size and 0)} "
              f"degenerate={dbg.get('degenerate_head_selection')}")
    print(f"[probe] DONE -> {out}")


if __name__ == "__main__":
    main()
