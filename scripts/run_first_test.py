"""End-to-end entry point for the Emboviz first test.

One command takes you from "no outputs" to "GIF + frame grid + deviation plot
+ HYPOTHESIS.md stub" in your outputs/ folder.

Usage:
    uv run python scripts/run_first_test.py --episode 0
    uv run python scripts/run_first_test.py --episode 7 --num-keyframes 8

If running on a machine without CUDA (e.g. Mac smoke-test), the script
auto-falls-back to CPU and warns.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Make the package importable when running this script directly (without uv).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emboviz.attribute import compute_attributions
from emboviz.load import load_episode, load_policy
from emboviz.replay import pick_keyframes, replay_episode
from emboviz.visualize import (
    render_deviation_plot,
    render_frame_grid_png,
    render_side_by_side_gif,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Emboviz first test")
    parser.add_argument("--episode", type=int, default=0, help="LeRobot pusht episode index")
    parser.add_argument("--num-keyframes", type=int, default=7, help="Frames to attribute & render")
    parser.add_argument("--ig-steps", type=int, default=16, help="Integrated Gradients integration steps")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("[run] WARNING: no CUDA — running on CPU. Expect 10–20 min per episode.")

    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[run] device={device}  episode={args.episode}  keyframes={args.num_keyframes}")

    print("[run] loading policy...")
    policy = load_policy(device=device)

    print("[run] loading episode...")
    episode = load_episode(episode_idx=args.episode)
    print(f"       T={episode.num_frames}  fps={episode.fps}")

    print("[run] replaying policy over episode...")
    replay = replay_episode(policy, episode, device=device)
    print(
        f"       failure frame: t={replay.failure_frame_idx}  "
        f"deviation={replay.action_deviations[replay.failure_frame_idx]:.4f}"
    )

    keyframes = pick_keyframes(replay, args.num_keyframes, episode.num_frames)
    print(f"[run] keyframes: {keyframes}")

    print("[run] computing attributions (IG + Saliency + Random)...")
    attributions = compute_attributions(
        policy, episode, keyframes, device=device, ig_steps=args.ig_steps
    )

    print("[run] rendering outputs...")
    gif_path = outdir / f"episode_{args.episode:04d}_attribution.gif"
    grid_path = outdir / f"episode_{args.episode:04d}_grid.png"
    dev_path = outdir / f"episode_{args.episode:04d}_deviation.png"

    render_side_by_side_gif(episode, attributions, gif_path, replay.failure_frame_idx)
    render_frame_grid_png(episode, attributions, grid_path, replay.failure_frame_idx)
    render_deviation_plot(replay, dev_path)

    _write_hypothesis_stub(
        outdir / "HYPOTHESIS.md",
        episode_idx=args.episode,
        failure_idx=replay.failure_frame_idx,
        max_dev=float(replay.action_deviations[replay.failure_frame_idx]),
        gif=gif_path,
        grid=grid_path,
        dev_plot=dev_path,
    )

    dt = time.time() - t0
    print(f"[run] done in {dt:.1f}s")
    print(f"       GIF  : {gif_path}")
    print(f"       grid : {grid_path}")
    print(f"       devs : {dev_path}")
    print(f"       notes: {outdir / 'HYPOTHESIS.md'}")
    return 0


def _write_hypothesis_stub(
    path: Path,
    *,
    episode_idx: int,
    failure_idx: int,
    max_dev: float,
    gif: Path,
    grid: Path,
    dev_plot: Path,
) -> None:
    """Drop a fill-in-the-blank readout next to the outputs.

    The whole point of the experiment is the human read of these images;
    forcing a written readout makes the result legible to non-authors.
    """
    body = f"""# Emboviz hypothesis check — episode {episode_idx}

**Hypothesis**: small robot policy + gradient attribution → heatmaps that
concentrate on task-relevant pixels (the T-block and the target), beating a
random-noise baseline.

## Artifacts
- GIF: `{gif.name}`
- Frame grid: `{grid.name}`
- Per-timestep action deviation: `{dev_plot.name}`

## Failure frame
- Index: **t={failure_idx}**
- ||policy − expert||: **{max_dev:.4f}**

## Eyeball test (fill in after looking)
- [ ] IG heatmaps concentrate on T-block + target across most frames
- [ ] Saliency heatmaps are at least directionally similar to IG (or clearly noisier)
- [ ] Both clearly beat the Random baseline (which should look like uniform noise)

## Failure-frame readout (3 sentences)

> _What is attribution pointing at, at t={failure_idx}? Does it suggest why
> the policy diverged from the expert?_

## Verdict
- [ ] PASS — green-light Stage B (OpenVLA + language attribution)
- [ ] MIXED — refine attribution method before Stage B
- [ ] FAIL — rethink premise
"""
    path.write_text(body)


if __name__ == "__main__":
    raise SystemExit(main())
