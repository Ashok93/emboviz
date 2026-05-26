"""Focused memorization-diagnostic probe — verify Pass 3 implementation
works end-to-end on a real model with a real target.

Run inside the openvla venv:
    /root/venvs/openvla/bin/python scripts/probe_memorization.py \
        --episode 0 --frame 0 --target-text "the spoon" --out /root/probes

Outputs to ``$OUT``:
    instruction.txt              — what the dataset says the task is
    detection.txt                — GroundingDINO bbox + confidence + label
    mask_overlay.png             — original primary image with mask alpha overlay
    masked_channel_mean.png      — primary image with channel-mean fill
    masked_gaussian_blur.png     — primary image with blur fill
    result.json                  — per-fill normalized_delta, mask_contrast, verdict
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from emboviz.calibration import calibrate_model
from emboviz.datasets.lerobot_bridge import BridgeEpisodeSource
from emboviz.diagnostics.memorization import MemorizationDiagnostic
from emboviz.models.openvla import OpenVLAAdapter
from emboviz.perturb._target_detection import GroundingDINOSAMDetector


def overlay_mask(img: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.4) -> Image.Image:
    """Return img with red translucent mask overlay."""
    arr = np.array(img).astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    blend = arr * (1 - alpha) + color_arr * alpha
    out = np.where(mask[..., None], blend, arr).astype(np.uint8)
    return Image.fromarray(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--target-text", required=True,
                   help="What to mask, e.g. 'the spoon'")
    p.add_argument("--out", default="/root/probes/memorization")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] loading Bridge episode {args.episode}", flush=True)
    src = BridgeEpisodeSource()
    traj = src.load_trajectory(args.episode)
    if args.frame >= len(traj.frames):
        raise SystemExit(f"frame {args.frame} out of range (traj has {len(traj.frames)})")
    scene = traj.frames[args.frame]
    (out / "instruction.txt").write_text(f"{scene.instruction}\n")
    print(f"      instruction: {scene.instruction!r}", flush=True)
    print(f"      cameras: {sorted(scene.observations.images)}", flush=True)
    primary = scene.observations.images["primary"].data
    primary.save(out / "primary_original.png")
    print(f"      target_text override: {args.target_text!r}", flush=True)

    print(f"[2/6] loading OpenVLA (this takes ~30s)", flush=True)
    model = OpenVLAAdapter()
    print(f"      ready: {model.model_id}", flush=True)

    print(f"[3/6] calibrating on this trajectory (n_noise_probes=3 for speed)", flush=True)
    cal = calibrate_model(model, traj, n_noise_probes=3)
    print(f"      noise_floor={cal.noise_floor:.4f}  "
          f"typical_action={cal.typical_action_magnitude:.4f}  "
          f"n_samples={cal.n_samples}", flush=True)

    print(f"[4/6] running GroundingDINO+SAM with target_text={args.target_text!r}", flush=True)
    detector = GroundingDINOSAMDetector(target_text=args.target_text, device="cuda")
    detection = detector(scene)
    if detection is None:
        (out / "detection.txt").write_text(
            f"NO DETECTION above confidence {detector.min_confidence} "
            f"for phrase {args.target_text!r} in frame {args.frame}.\n"
        )
        print(f"      DETECTION FAILED — see {out}/detection.txt", flush=True)
        raise SystemExit(2)
    (out / "detection.txt").write_text(
        f"label:      {detection.label}\n"
        f"bbox:       {detection.bbox}\n"
        f"confidence: {detection.confidence:.4f}\n"
        f"mask shape: {detection.mask.shape if detection.mask is not None else None}\n"
        f"mask area:  {int(detection.mask.sum()) if detection.mask is not None else 'n/a'} px\n"
    )
    print(f"      detected: bbox={detection.bbox} conf={detection.confidence:.3f} "
          f"mask_area={int(detection.mask.sum())} px", flush=True)
    overlay_mask(primary, detection.mask).save(out / "mask_overlay.png")

    print(f"[5/6] running MemorizationDiagnostic (fill ensemble)", flush=True)
    diag = MemorizationDiagnostic(target_detector=detector, calibration=cal)
    result = diag.run(model, scene)
    print(f"      severity:    {result.severity.value}", flush=True)
    print(f"      scalar:      {result.scalar_score:.4f}", flush=True)
    print(f"      explanation: {result.explanation}", flush=True)

    # Save per-fill masked images for visual inspection.
    if result.raw and "per_fill" in result.raw:
        from emboviz.perturb.image._image_utils import to_array, to_pil
        from emboviz.diagnostics.memorization import _apply_fill
        arr = to_array(primary)
        for fill_mode in ["channel_mean", "gaussian_blur"]:
            masked = _apply_fill(arr, detection.mask, fill_mode)
            to_pil(masked).save(out / f"masked_{fill_mode}.png")

    print(f"[6/6] writing result.json", flush=True)
    serializable = {
        "instruction":  scene.instruction,
        "target_text":  args.target_text,
        "detection": {
            "label":      detection.label,
            "bbox":       list(detection.bbox),
            "confidence": float(detection.confidence),
            "mask_area_px": int(detection.mask.sum()),
        },
        "calibration": cal.to_summary(),
        "severity":    result.severity.value,
        "scalar":      float(result.scalar_score),
        "explanation": result.explanation,
        "raw":         result.raw,
    }
    (out / "result.json").write_text(json.dumps(serializable, indent=2, default=str))
    print(f"\n=== ALL OUTPUTS IN {out} ===", flush=True)
    for f in sorted(out.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
