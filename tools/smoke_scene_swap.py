"""Smoke-test the scene swap in isolation — SAM 3 + SD inpaint (or LaMa), no dream.

Confirms the inpainting half of the closed-loop dream WITHOUT a Cosmos server, a
policy, or forward kinematics: it loads one real frame, runs the configured
``scene_swap`` across its concat cameras, and writes the before/after images so
you can eyeball the edit (e.g. marker -> spoon) and read the per-camera status.

This is the fast path to validate the SAM 3 + sd-inpaint/LaMa workers on a fresh
GPU box before paying for the full dream run.

Run::

    uv run python tools/smoke_scene_swap.py --config configs/droid_pi0.yaml \
        --episode 312 --near-frame 60 --out outputs/scene_swap_smoke

Outputs (per region wrist / exterior_left / exterior_right): ``<region>_before.png``,
``<region>_after.png`` (only when edited), plus ``concat_before.png`` /
``concat_after.png`` — the full stitched seed before and after the swap.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from emboviz.config import load_run_config
from emboviz.datasets.manifest import build_source
from emboviz.world_models.keyframes import detect_keyframes


def _img(frame, role: str) -> np.ndarray:
    if role not in frame.observations.images:
        raise SystemExit(
            f"camera role {role!r} (from cosmos_stress.concat_cameras) is not in the "
            f"episode (available: {sorted(frame.observations.images)})."
        )
    return np.asarray(frame.observations.images[role].data, dtype=np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--frame", type=int, default=None, help="Exact frame index to swap.")
    p.add_argument(
        "--near-frame", type=int, default=None,
        help="Pick the gripper_change keyframe nearest this index (the grasp). "
             "Ignored if --frame is given.",
    )
    p.add_argument("--steps", type=int, default=None, help="SD inpaint steps (default: worker default).")
    p.add_argument("--out", default="outputs/scene_swap_smoke")
    args = p.parse_args()

    cfg = load_run_config(args.config)
    cs = cfg.analysis.cosmos_stress
    if cs is None or cs.scene_swap is None:
        raise SystemExit(
            "config has no analysis.cosmos_stress.scene_swap section — nothing to swap."
        )
    sw = cs.scene_swap

    episode = args.episode
    if episode is None:
        episode = int(str(cfg.analysis.episodes).split(",")[0].split("-")[0])

    print(f"[smoke] loading episode {episode} via {cfg.dataset.format} reader ...")
    real = build_source(**cfg.dataset_build_kwargs()).load_trajectory(episode)
    print(f"[smoke] {len(real.frames)} frames, fps {real.fps:g}")

    # Choose the frame to swap.
    if args.frame is not None:
        idx = args.frame
    elif args.near_frame is not None:
        kfs = [k for k in detect_keyframes(real) if k.kind == "gripper_change"]
        if not kfs:
            raise SystemExit("no gripper_change keyframes found; pass --frame instead.")
        idx = min(kfs, key=lambda k: abs(k.index - args.near_frame)).index
        print(f"[smoke] --near-frame {args.near_frame}: nearest grasp keyframe is {idx}")
    else:
        idx = len(real.frames) // 2
        print(f"[smoke] no --frame/--near-frame; using mid-episode frame {idx}")
    if not 0 <= idx < len(real.frames):
        raise SystemExit(f"frame {idx} out of range (episode has {len(real.frames)} frames).")
    frame = real.frames[idx]

    # Bring up the workers + build the swapper, exactly as the dream driver does.
    from emboviz.adapters import connect
    from emboviz.perturb._target_detection import SAM3Detector
    from emboviz.world_models.scene_swap import SceneSwapper
    from emboviz_cosmos3.concat_view import build_concat_view

    connect("sam3", auto_spawn=True, auto_install=True)
    detector = SAM3Detector(
        target_text=sw.mask_query,
        score_threshold=sw.detector_score_threshold,
        mask_threshold=sw.detector_mask_threshold,
    )
    if sw.replace_query:
        from emboviz.perturb.image._inpaint import SDInpaintInserter
        connect("sd-inpaint", auto_spawn=True, auto_install=True)
        swapper = SceneSwapper(
            mask_query=sw.mask_query, replace_query=sw.replace_query,
            detector=detector, inserter=SDInpaintInserter(num_inference_steps=args.steps),
        )
        print(f"[smoke] swap: {sw.mask_query!r} -> {sw.replace_query!r} (SAM 3 + SD inpaint)")
    else:
        from emboviz.perturb.image._inpaint import LamaInpainter
        connect("lama", auto_spawn=True, auto_install=True)
        swapper = SceneSwapper(
            mask_query=sw.mask_query, detector=detector, inpainter=LamaInpainter(),
        )
        print(f"[smoke] swap: remove {sw.mask_query!r} (SAM 3 + LaMa)")

    result = swapper.swap(frame, cs.concat_cameras)
    print(f"[smoke] {result.summary()}")

    # Write before/after per camera + the full concat.
    from PIL import Image
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for cam in result.per_camera:
        before = _img(frame, cam.role)
        Image.fromarray(before, mode="RGB").save(out / f"{cam.region}_before.png")
        if cam.edited:
            Image.fromarray(result.images_by_region[cam.region], mode="RGB").save(
                out / f"{cam.region}_after.png"
            )

    concat_before = build_concat_view(
        _img(frame, cs.concat_cameras["wrist"]),
        _img(frame, cs.concat_cameras["exterior_left"]),
        _img(frame, cs.concat_cameras["exterior_right"]),
        wrist_size=cs.concat_resolution,
    )
    concat_after = build_concat_view(
        result.images_by_region["wrist"],
        result.images_by_region["exterior_left"],
        result.images_by_region["exterior_right"],
        wrist_size=cs.concat_resolution,
    )
    Image.fromarray(concat_before, mode="RGB").save(out / "concat_before.png")
    Image.fromarray(concat_after, mode="RGB").save(out / "concat_after.png")

    (out / "swap.json").write_text(json.dumps({
        "episode": episode, "frame": idx,
        "mask_query": sw.mask_query, "replace_query": sw.replace_query,
        "any_edited": result.any_edited,
        "per_camera": [asdict(c) for c in result.per_camera],
    }, indent=2))

    print(f"[smoke] wrote before/after to {out}/ "
          f"({'edited ' + ', '.join(result.edited_regions) if result.any_edited else 'NO camera detected the target'})")


if __name__ == "__main__":
    main()
