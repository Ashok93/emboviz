"""Render an attention-only Rerun .rrd for ONE episode: per frame, extract the
model's attention, apply the calibrated localization heads, and log the RGB +
attention overlay. Streaming + memory-safe — we keep only the tiny (side×side)
heatmap per frame and free the full attention tensors immediately, so it does
NOT blow up memory / starve the host like the full diagnostic suite.

Usage (per model venv, on the VM):
    <venv>/bin/python scripts/render_attention.py \
        --model-builder emboviz.models.registry:get_model:pi0 \
        --model-kwargs '{"config_name":"pi0_libero","use_pytorch":true}' \
        --dataset-builder emboviz.datasets.lerobot_libero:PhysicalIntelligenceLiberoSource \
        --episode 617 --heads-cache /root/attn_dbg/pi0_heads.json \
        --out /root/attn_dbg/rrd/pi0_ep617.rrd
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
from pathlib import Path

import numpy as np

from emboviz.core.types import TokenSelector, Trajectory
from emboviz.exporters.rerun import export_rerun


def _resolve(spec: str, kwargs_json: str = ""):
    parts = spec.split(":")
    module = importlib.import_module(parts[0])
    obj = getattr(module, parts[1])
    kwargs = json.loads(kwargs_json) if kwargs_json else {}
    if len(parts) == 2:
        return obj(**kwargs)
    intermediate = obj(parts[2])
    return intermediate(**kwargs) if isinstance(intermediate, type) else intermediate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-builder", required=True)
    p.add_argument("--model-kwargs", default="")
    p.add_argument("--dataset-builder", required=True)
    p.add_argument("--dataset-kwargs", default="")
    p.add_argument("--episode", required=True)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--heads-cache", default="")
    p.add_argument("--baseline-subtract", action="store_true",
                   help="subtract the content-free (gray-image) attention sink (computed once)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model = _resolve(args.model_builder, args.model_kwargs)
    dataset = _resolve(args.dataset_builder, args.dataset_kwargs)

    hbc = None
    if args.heads_cache:
        cache = json.loads(Path(args.heads_cache).read_text())
        hbc = {c: [tuple(h) for h in hs] for c, hs in cache.get("heads_by_camera", {}).items()}
        print(f"[render] calibrated heads: {hbc}", flush=True)

    frames_all = dataset.load_episode(str(args.episode))
    idxs = list(range(0, len(frames_all), max(1, args.frame_stride)))
    if args.max_frames > 0:
        idxs = idxs[: args.max_frames]
    scenes = [frames_all[i] for i in idxs]
    print(f"[render] ep {args.episode}: {len(scenes)} frames "
          f"(stride {args.frame_stride}), instr={scenes[0].instruction!r}", flush=True)

    sel = TokenSelector(relative="before_action")

    # Optional content-independent attention-sink baseline: the model's
    # attention on a CONTENT-FREE (gray) image with the same instruction is the
    # pure positional/register sink (image-independent). Subtracting it isolates
    # the content-driven grounding (standard attribution baseline). It does NOT
    # depend on the frame, so we compute it ONCE for the episode and reuse it.
    gray_clean: dict[str, np.ndarray] = {}
    if args.baseline_subtract:
        import numpy as _np
        from PIL import Image as _PILImage
        s0 = scenes[0]
        gray = s0
        for cam in s0.observations.images:
            arr = _np.asarray(s0.observations.images[cam].data)
            gray = gray.with_image(_PILImage.fromarray(_np.full_like(arr, 128)), camera=cam)
        gattn = model.extract_attention(gray, sel)
        for cam in gattn.cameras:
            if cam in s0.observations.images:
                gray_clean[cam], _ = gattn.image_weights_clean(cam)
        del gattn; gc.collect()
        print(f"[render] computed gray-image sink baseline for {sorted(gray_clean)}", flush=True)

    attention_per_frame: dict[int, dict] = {}
    for i, scene in enumerate(scenes):
        attn = model.extract_attention(scene, sel)
        if hbc is not None:
            attn.metadata["localization_heads_by_camera"] = hbc
        per_cam = {}
        for cam in attn.cameras:
            if cam not in scene.observations.images:
                continue   # padding slot
            heat, _ = attn.image_weights_clean(cam)
            if cam in gray_clean:
                heat = np.clip(heat - gray_clean[cam], 0.0, None)   # remove sink
            per_cam[cam] = heat                        # small (side, side)
        attention_per_frame[i] = per_cam
        del attn
        gc.collect()
        if (i + 1) % 10 == 0 or i + 1 == len(scenes):
            print(f"[render] {i+1}/{len(scenes)}", flush=True)

    traj = Trajectory(frames=scenes, frame_indices=list(range(len(scenes))),
                      fps=10.0, episode_id=str(args.episode), source="attention_render")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_rerun(traj, {}, out_path, attention_per_frame=attention_per_frame)
    print(f"[render] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
