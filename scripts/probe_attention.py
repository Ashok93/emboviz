"""Per-model attention-extraction probe.

Loads the named model + a one-frame scene from the named dataset, calls
extract_attention(), and saves per-camera heatmap overlays + a JSON of
the raw extraction metadata. Used as the smoke test as we wire
attention extraction for each new adapter.

Run inside the model's venv. Examples:

    /root/venvs/openvla/bin/python scripts/probe_attention.py \\
        --model openvla --dataset bridge --episode 0 --frame 0 \\
        --out /root/probes/attn_openvla_bridge

    /root/repos/openvla-oft/.venv/bin/python scripts/probe_attention.py \\
        --model oft --dataset libero-spatial --episode 0 --frame 0 \\
        --out /root/probes/attn_oft_libero
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from emboviz.core.types import TokenSelector
from emboviz.models.protocol import Capability


def load_model(name: str):
    if name == "openvla":
        from emboviz.models.openvla import OpenVLAAdapter
        return OpenVLAAdapter()
    if name == "oft":
        from emboviz.models.openvla_oft import OpenVLAOFTAdapter
        return OpenVLAOFTAdapter()
    if name == "pi0":
        from emboviz.models.pi0 import Pi0Adapter
        return Pi0Adapter(config_name="pi0_libero", use_pytorch=True)
    if name == "gr00t":
        from emboviz.models.gr00t import Gr00tAdapter
        # droid_sample has 2 cameras; map them to GR00T's video keys.
        # primary  → exterior_image_1_left (front exterior view)
        # wrist_left → wrist_image_left
        return Gr00tAdapter(camera_mapping={
            "primary":    "exterior_image_1_left",
            "wrist_left": "wrist_image_left",
        })
    raise SystemExit(f"unknown --model {name!r}; supported: openvla, oft, pi0, gr00t")


def load_scene(dataset: str, episode: int, frame: int):
    if dataset == "bridge":
        from emboviz.datasets.lerobot_bridge import BridgeEpisodeSource
        traj = BridgeEpisodeSource().load_trajectory(episode)
    elif dataset == "libero-spatial":
        from emboviz.datasets.lerobot_libero import LiberoSpatialSource
        traj = LiberoSpatialSource().load_trajectory(episode)
    elif dataset == "pi-libero":
        from emboviz.datasets.lerobot_libero import PhysicalIntelligenceLiberoSource
        traj = PhysicalIntelligenceLiberoSource().load_trajectory(episode)
    elif dataset == "droid-sample":
        from emboviz.datasets.lerobot_droid import GR00TDroidSampleSource
        traj = GR00TDroidSampleSource().load_trajectory(episode)
    else:
        raise SystemExit(f"unknown --dataset {dataset!r}")
    if frame >= len(traj.frames):
        raise SystemExit(f"frame {frame} out of range (traj has {len(traj.frames)})")
    return traj.frames[frame]


def overlay_heatmap(img: Image.Image, heat: np.ndarray, out_path: Path):
    """Resize 2-D heatmap to image size, render as red-hot, blend."""
    import matplotlib.cm as cm
    h_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
    h_full = np.array(
        Image.fromarray((h_norm * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    ).astype(np.float32) / 255.0
    hot = (cm.hot(h_full)[..., :3] * 255).astype(np.uint8)
    base = np.array(img).astype(np.float32)
    blend = (base * 0.55 + hot * 0.45).clip(0, 255).astype(np.uint8)
    Image.fromarray(blend).save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["openvla", "oft", "pi0", "gr00t"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading scene: dataset={args.dataset} ep{args.episode} frame{args.frame}", flush=True)
    scene = load_scene(args.dataset, args.episode, args.frame)
    print(f"  instruction: {scene.instruction!r}")
    print(f"  cameras in scene: {sorted(scene.observations.images)}")
    (out / "instruction.txt").write_text(f"{scene.instruction}\n")
    for cam in sorted(scene.observations.images):
        scene.observations.images[cam].data.save(out / f"original_{cam}.png")

    print(f"[2/4] loading model: {args.model}", flush=True)
    model = load_model(args.model)
    print(f"  model_id: {model.model_id}")
    print(f"  caps:     {model.capabilities}")
    if not (model.capabilities & Capability.ATTENTION):
        raise SystemExit(
            f"model {args.model!r} does not declare Capability.ATTENTION yet. "
            "Add extract_attention() to its adapter first."
        )

    print(f"[3/4] extract_attention(before_action)", flush=True)
    am = model.extract_attention(scene, TokenSelector(relative="before_action"))
    print(f"  weights shape:       {am.weights.shape}  (layers, heads, n_keys)")
    print(f"  query_position:      {am.query_position}")
    print(f"  n_keys:              {am.n_keys}")
    print(f"  cameras (attention): {am.cameras}")
    print(f"  per-camera ranges:   {am.image_token_ranges}")
    print(f"  per-camera grids:    {am.image_grid_sides}")
    if am.metadata:
        print(f"  metadata:            {am.metadata}")

    print(f"[4/4] save per-camera heatmap overlays (clean + raw)", flush=True)
    per_cam_stats: dict = {}
    for cam in am.cameras:
        raw_side = am.image_weights(cam)
        raw_agg = raw_side.mean(axis=(0, 1))
        clean_agg, debug = am.image_weights_clean(cam)
        per_cam_stats[cam] = {
            "raw":   {
                "shape": list(raw_side.shape),
                "min":   float(raw_agg.min()),
                "max":   float(raw_agg.max()),
                "mean":  float(raw_agg.mean()),
                "mass":  float(raw_agg.sum()),
            },
            "clean": {
                "shape": list(clean_agg.shape),
                "min":   float(clean_agg.min()),
                "max":   float(clean_agg.max()),
                "mean":  float(clean_agg.mean()),
                "mass":  float(clean_agg.sum()),
            },
            "debug": debug,
        }
        print(f"  {cam:>10s}: raw mass={raw_agg.sum():.4f}  clean mass={clean_agg.sum():.4f}  "
              f"sink cells masked={debug['n_sink_cells_masked']}  "
              f"layers used={debug['layer_range']}  "
              f"({debug['profile_source'][:60]}...)")
        # Overlay on whichever scene-image best matches this camera name
        # (e.g. for π0 the "padding" slot has no real scene image — fall back to primary).
        target_cam = cam if cam in scene.observations.images else "primary"
        overlay_heatmap(
            scene.observations.images[target_cam].data, raw_agg,
            out / f"heatmap_{cam}_on_{target_cam}_raw.png",
        )
        overlay_heatmap(
            scene.observations.images[target_cam].data, clean_agg,
            out / f"heatmap_{cam}_on_{target_cam}_clean.png",
        )

    summary = {
        "model":        args.model,
        "model_id":     model.model_id,
        "dataset":      args.dataset,
        "episode":      args.episode,
        "frame":        args.frame,
        "instruction":  scene.instruction,
        "weights_shape": list(am.weights.shape),
        "query_position": int(am.query_position),
        "n_keys":       int(am.n_keys),
        "image_token_ranges": {k: list(v) for k, v in am.image_token_ranges.items()},
        "image_grid_sides":   am.image_grid_sides,
        "metadata":     am.metadata,
        "per_camera":   per_cam_stats,
    }
    (out / "result.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n=== ALL OUTPUTS IN {out} ===")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
