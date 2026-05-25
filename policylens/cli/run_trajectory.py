"""Run a Suite across every frame of a trajectory; render timelines.

Usage:
    uv run python -m policylens.cli.run_trajectory \
        --model openvla-7b \
        --suite quick_smoke \
        --trajectory bridge:0 \
        --stride 4 \
        --outdir outputs/traj_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from policylens.cli._loaders import build_suite, load_model, load_trajectory
from policylens.core.results import Severity
from policylens.reports.trajectory_timeline import (
    render_failure_tape,
    render_trajectory_timelines,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", required=True,
                        choices=["language_grounding", "visual_robustness",
                                 "full_profile", "quick_smoke"])
    parser.add_argument("--trajectory", required=True,
                        help="Trajectory spec, e.g. bridge:0")
    parser.add_argument("--stride", type=int, default=1,
                        help="Subsample every Nth frame (default: every frame)")
    parser.add_argument("--outdir", default="outputs/trajectory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[traj] model: {args.model}", flush=True)
    model = load_model(args.model)

    print(f"[traj] trajectory: {args.trajectory}", flush=True)
    traj = load_trajectory(args.trajectory)
    print(f"      raw frames: {len(traj)}  fps={traj.fps:.1f}  "
          f"instruction=\"{traj.frames[0].instruction if traj.frames else ''}\"")
    if args.stride > 1:
        print(f"      stride: {args.stride} → {len(traj) // args.stride + 1} frames", flush=True)

    print(f"[traj] running suite {args.suite!r}...", flush=True)
    suite = build_suite(args.suite)
    result = suite.run_trajectory(model, traj, stride=args.stride)

    # Console summary per axis
    print("\n[traj] === per-axis summary ===")
    for name, tr in result.results.items():
        worst = tr.worst_frame_idx
        mean = tr.mean_score
        n_fail = len(tr.failure_moments(Severity.CRITICAL))
        print(f"  {tr.axis:35s}  mean={mean:.3f}  worst@t={worst}  "
              f"critical-frames={n_fail}/{len(tr.per_frame)}")

    print("\n[traj] rendering timelines + failure tape...", flush=True)
    render_trajectory_timelines(result, outdir / "trajectory_timelines.png")
    render_failure_tape(result, outdir / "failure_tape.png")
    (outdir / "trajectory_summary.json").write_text(
        json.dumps(result.summary(), indent=2, default=lambda o: o.tolist()
                   if hasattr(o, "tolist") else str(o))
    )
    print(f"[traj] done → {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
