"""Critical-moment stress test driver.

Find the decisive instants of a recorded episode (grasps, releases, settles) and,
at each one, roll a short world-model clip from just before the event and compare
it to what really happened. Two action sources:

  * ``--source recorded`` (default) — the episode's own logged actions. The clip
    should track reality; this is the faithfulness baseline and needs no policy.
  * ``--source policy`` — the user's policy drives (via the Cosmos action bridge),
    so each clip shows what the policy would do at that moment. Requires
    ``--policy-adapter`` and ``--action-convention``.

Each clip is written the moment it is generated (frames + divergence.json) so a
long run never buffers everything and loses it on a late failure.

Run (host side; needs the reader + Cosmos world-model workers reachable)::

    uv run python -m emboviz.world_models.stress_cli \
        --config configs/droid.yaml --episode 0 \
        --world-model cosmos3 --server-url https://<podid>-8000.proxy.runpod.net \
        --domain droid_lerobot --action-dim 10 \
        --source recorded --n-actions 16 --lead-s 0.5 \
        --out outputs/cosmos_stress
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emboviz.adapters import connect_world_model
from emboviz.config import load_run_config
from emboviz.datasets.manifest import build_source
from emboviz.world_models.keyframes import detect_keyframes
from emboviz.world_models.stress import StressClip, recorded_action_source, stress_test
from emboviz.world_models.viz import save_frame_comparison


def _build_action_source(args, world_model):
    if args.source == "recorded":
        return recorded_action_source(world_model)
    if args.source == "policy":
        if not args.policy_adapter or not args.action_convention:
            raise SystemExit(
                "--source policy requires --policy-adapter and --action-convention."
            )
        # Lazy import: the Cosmos action bridge lives in the adapter, so the core
        # CLI only touches it on the explicit policy path.
        from emboviz.adapters import connect
        from emboviz_cosmos3.bridge import policy_action_source

        print(f"[stress] connecting policy '{args.policy_adapter}' ...")
        policy = connect(args.policy_adapter)
        return policy_action_source(policy.client.predict, convention=args.action_convention)
    raise SystemExit(f"unknown --source {args.source!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--world-model", default="cosmos3")
    p.add_argument("--server-url", required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--action-dim", type=int, required=True)
    p.add_argument("--source", default="recorded", choices=["recorded", "policy"])
    p.add_argument("--policy-adapter", default=None, help="VLA adapter name for --source policy")
    p.add_argument("--action-convention", default=None,
                   help="policy action convention for the bridge "
                        "(absolute_xyz_euler | delta_xyz_euler_base)")
    p.add_argument("--n-actions", type=int, default=16, help="rollout length per clip")
    p.add_argument("--lead-s", type=float, default=0.5, help="seconds before each keyframe to seed")
    p.add_argument("--camera", default="primary")
    p.add_argument("--metric", default="pixel_l2", choices=["pixel_l2", "ssim"])
    p.add_argument("--out", default="outputs/cosmos_stress")
    args = p.parse_args()

    cfg = load_run_config(args.config)
    episode = args.episode
    if episode is None:
        episode = int(str(cfg.analysis.episodes).split(",")[0].split("-")[0])

    print(f"[stress] loading episode {episode} via {cfg.dataset.format} reader ...")
    real = build_source(**cfg.dataset_build_kwargs()).load_trajectory(episode)
    keyframes = detect_keyframes(real)
    print(f"[stress] episode {len(real.frames)} frames, fps {real.fps:g}; "
          f"{len(keyframes)} keyframes: "
          + ", ".join(f"{kf.index}:{kf.kind}" for kf in keyframes))

    wm = connect_world_model(args.world_model, world_model_kwargs={
        "server_url": args.server_url, "domain_name": args.domain,
        "action_dim": args.action_dim, "conditioning_camera": args.camera,
    })
    action_source = _build_action_source(args, wm)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    def on_clip(clip: StressClip) -> None:
        kf = clip.keyframe
        clip_dir = out / f"clip_{kf.index:04d}_{kf.kind}"
        n = save_frame_comparison(
            clip.predicted, clip.aligned_real, clip.divergence, clip_dir / "frames",
            camera=clip.camera, start_index=clip.seed_index,
        )
        record = {
            "keyframe_index": kf.index,
            "kind": kf.kind,
            "gripper_delta": kf.gripper_delta,
            "seed_index": clip.seed_index,
            "n_frames": n,
            "metric": clip.metric,
            "divergence": clip.divergence,
            "divergence_max": max(clip.divergence) if clip.divergence else None,
            "divergence_mean": (sum(clip.divergence) / len(clip.divergence)) if clip.divergence else None,
        }
        (clip_dir / "divergence.json").write_text(json.dumps(record, indent=2))
        summary.append(record)
        (out / "summary.json").write_text(json.dumps(
            {"source": args.source, "n_actions": args.n_actions, "lead_s": args.lead_s, "clips": summary},
            indent=2,
        ))
        dmax = record["divergence_max"]
        print(f"  clip @ frame {kf.index:4d} ({kf.kind:14s}) seed {clip.seed_index:4d}  "
              f"{n} frames  div_max {dmax:.3f}  (saved)" if dmax is not None
              else f"  clip @ frame {kf.index} ({kf.kind}) — no frames", flush=True)

    clips = stress_test(
        wm, real,
        action_source=action_source,
        n_actions=args.n_actions, lead_s=args.lead_s,
        camera=args.camera, metric=args.metric,
        on_clip=on_clip,
    )

    skipped = len(keyframes) - len(clips)
    print(f"\n[stress] DONE: {len(clips)} clips -> {out}/"
          + (f"  ({skipped} keyframes skipped — too close to the episode end for a "
             f"{args.n_actions}-step rollout)" if skipped else ""))


if __name__ == "__main__":
    main()
