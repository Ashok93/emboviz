"""Re-anchored (closed-loop) world-model rollout over a real episode.

Predict K frames from a real frame, snap back to the next real frame, predict K
more, … — so prediction error never compounds past the trust horizon. The result
is a world model that stays coherent over a long rollout instead of drifting into
hallucination.

Crucially, this writes **each segment to disk the moment it is generated** (frames
+ a running ``divergence.json``) and prints per-segment progress — a long
generation run must never buffer everything and lose it on a late failure.

Run (host side; needs the reader + world-model workers reachable)::

    uv run python -m emboviz.world_models.reanchor_cli \
        --config configs/droid.yaml --episode 0 \
        --world-model cosmos3 --server-url https://<podid>-8000.proxy.runpod.net \
        --domain droid_lerobot --action-dim 10 \
        --frame-start 0 --n-actions -1 --reanchor-every 3 \
        --out outputs/cosmos_reanchored
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from emboviz.config import load_run_config
from emboviz.datasets.manifest import build_source
from emboviz.adapters import connect_world_model
from emboviz.world_models.rollout import reanchored_rollout
from emboviz.world_models.trust import frame_divergence
from emboviz.world_models.viz import save_frame_comparison


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--world-model", default="cosmos3")
    p.add_argument("--server-url", required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--action-dim", type=int, required=True)
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--n-actions", type=int, default=-1, help="-1 = the whole episode")
    p.add_argument("--reanchor-every", type=int, default=3)
    p.add_argument("--camera", default="primary")
    p.add_argument("--metric", default="pixel_l2", choices=["pixel_l2", "ssim"])
    p.add_argument("--out", default="outputs/cosmos_reanchored")
    args = p.parse_args()

    cfg = load_run_config(args.config)
    episode = args.episode
    if episode is None:
        episode = int(str(cfg.analysis.episodes).split(",")[0].split("-")[0])

    print(f"[reanchor] loading episode {episode} via {cfg.dataset.format} reader ...")
    real = build_source(**cfg.dataset_build_kwargs()).load_trajectory(episode)
    n_actions = args.n_actions if args.n_actions > 0 else len(real.frames) - 1 - args.frame_start
    print(f"[reanchor] episode {len(real.frames)} frames; rolling {n_actions} from "
          f"{args.frame_start}, re-anchor every {args.reanchor_every}")

    wm = connect_world_model(args.world_model, world_model_kwargs={
        "server_url": args.server_url, "domain_name": args.domain,
        "action_dim": args.action_dim, "conditioning_camera": args.camera,
    })

    out = Path(args.out)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    all_divs: list[float] = []
    band: dict = {"value": None}

    def _img(scene):
        return np.asarray(scene.observations.images[args.camera].data, dtype=np.uint8)

    def on_segment(out_start: int, predicted, real_frames) -> None:
        seg_divs = [
            frame_divergence(_img(p), _img(r), args.metric)
            for p, r in zip(predicted, real_frames)
        ]
        all_divs.extend(seg_divs)
        # Estimate the trust band once, from the first frames' irreducible error.
        if band["value"] is None and len(all_divs) >= 2:
            band["value"] = round(2.0 * float(np.mean(all_divs[:2])), 4)
        # SAVE NOW — frames for this segment + the running divergence — so a later
        # failure never throws away what's already been generated.
        save_frame_comparison(
            predicted, real_frames, seg_divs, frames_dir,
            camera=args.camera, trust_band=band["value"], start_index=out_start,
        )
        (out / "divergence.json").write_text(json.dumps({
            "reanchor_every": args.reanchor_every, "trust_band": band["value"],
            "divergence": all_divs,
        }, indent=2))
        print(f"  frames {out_start:3d}..{out_start + len(seg_divs) - 1:3d}  "
              f"div {seg_divs[0]:.3f}..{seg_divs[-1]:.3f}  (saved)", flush=True)

    reanchored_rollout(
        wm, real, frame_start=args.frame_start, n_actions=n_actions,
        reanchor_every=args.reanchor_every, on_segment=on_segment,
    )

    if all_divs:
        print(f"\n[reanchor] DONE: {len(all_divs)} frames -> {out}/")
        print(f"  divergence  max {max(all_divs):.3f}  mean {sum(all_divs)/len(all_divs):.3f}  "
              f"(an open-loop rollout drifts unbounded; re-anchoring keeps it bounded)")
    else:
        print("[reanchor] no frames produced")


if __name__ == "__main__":
    main()
